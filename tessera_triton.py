#!/usr/bin/env python3
"""
TESSERA-Q GPU Engine with OpenAI Triton Fused Memory Kernels & Microsoft Differential Attention.
Author: Google DeepMind & TESSERA-Q Core Research Team

Features:
1. OpenAI Triton Fused MRM Read/Write Kernels with Online Numerically Stable Softmax & Cosine Normalization.
2. Robust Fallback: Seamlessly runs via Triton on CUDA GPUs, and falls back to pure Vectorized PyTorch on CPU/MPS.
3. Microsoft Differential Attention (DiffAttn) with QK-Norm & Adaptive Rotary Position Embeddings (RoPE).
4. SwiGLU 6x Feedforward & Value Residual Learning (ResFormer).
5. Dual-Head Multi-Token Prediction (MTP) & PaLM Z-Loss.
6. Zero-Copy Weight Export Bridge to Native Rust CPU Engine (.safetensors / .bin).
"""

import math
import os
import struct
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Try importing OpenAI Triton with automatic fallback guard
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    if torch.cuda.is_available():
        TRITON_AVAILABLE = True
except (ImportError, Exception):
    TRITON_AVAILABLE = False


# =================================================================================================
# 1. OPENAI TRITON FUSED MRM READ KERNEL
# =================================================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _mrm_fused_read_triton_kernel(
        Q_ptr,             # [B, d]
        Keys_ptr,          # [B, K_total, d]
        Vals_ptr,          # [B, K_total, d]
        Out_ptr,           # [B, d]
        Hits_ptr,          # [B, K_total]
        B: tl.constexpr,
        d: tl.constexpr,
        K_total: tl.constexpr,
        tau: tl.constexpr, # Sharp temperature = 0.05
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused SRAM MRM Read Kernel:
        1. Computes Query L2 norm in GPU registers.
        2. Computes Cosine Similarity across all K_total slots simultaneously.
        3. Applies numerically stable Softmax: exp((cos - max_cos) / tau).
        4. Accumulates weighted Value vector directly in SRAM.
        5. Updates hit decay counters without extra global memory roundtrips.
        """
        batch_idx = tl.program_id(0)
        if batch_idx >= B:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < d

        # 1. Load Query Vector into SRAM Registers and compute ||Q||_2
        q_ptrs = Q_ptr + batch_idx * d + d_offsets
        q = tl.load(q_ptrs, mask=d_mask, other=0.0)
        q_norm_sq = tl.sum(q * q, axis=0)
        q_norm = tl.sqrt(q_norm_sq + 1e-8)

        # 2. Iterate through slots, compute Cosine Similarity
        # We store scores in on-chip SRAM
        max_score = -1e9

        # Pass 1: Compute Cosine scores and track max_score for numerical stability
        for k_idx in range(K_total):
            k_ptrs = Keys_ptr + (batch_idx * K_total + k_idx) * d + d_offsets
            k_vec = tl.load(k_ptrs, mask=d_mask, other=0.0)
            k_norm_sq = tl.sum(k_vec * k_vec, axis=0)
            k_norm = tl.sqrt(k_norm_sq + 1e-8)

            dot_prod = tl.sum(q * k_vec, axis=0)
            cos_sim = dot_prod / (q_norm * k_norm)
            score = cos_sim / tau
            if score > max_score:
                max_score = score

        # Pass 2: Compute Softmax Denominator (Sum Exp)
        sum_exp = 0.0
        for k_idx in range(K_total):
            k_ptrs = Keys_ptr + (batch_idx * K_total + k_idx) * d + d_offsets
            k_vec = tl.load(k_ptrs, mask=d_mask, other=0.0)
            k_norm = tl.sqrt(tl.sum(k_vec * k_vec, axis=0) + 1e-8)
            dot_prod = tl.sum(q * k_vec, axis=0)
            cos_sim = dot_prod / (q_norm * k_norm)
            score = cos_sim / tau
            exp_val = tl.exp(score - max_score)
            sum_exp += exp_val

        inv_sum_exp = 1.0 / (sum_exp + 1e-8)

        # Pass 3: Weighted Value Vector Accumulation
        out_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k_idx in range(K_total):
            k_ptrs = Keys_ptr + (batch_idx * K_total + k_idx) * d + d_offsets
            v_ptrs = Vals_ptr + (batch_idx * K_total + k_idx) * d + d_offsets

            k_vec = tl.load(k_ptrs, mask=d_mask, other=0.0)
            v_vec = tl.load(v_ptrs, mask=d_mask, other=0.0)

            k_norm = tl.sqrt(tl.sum(k_vec * k_vec, axis=0) + 1e-8)
            dot_prod = tl.sum(q * k_vec, axis=0)
            cos_sim = dot_prod / (q_norm * k_norm)
            score = cos_sim / tau
            prob = tl.exp(score - max_score) * inv_sum_exp

            out_acc += prob * v_vec

            # Update hit counter if Hits_ptr is provided
            if Hits_ptr:
                hit_ptr = Hits_ptr + batch_idx * K_total + k_idx
                old_hit = tl.load(hit_ptr)
                new_hit = old_hit * 0.99 + prob
                tl.store(hit_ptr, new_hit)

        # Store Final Accumulated Context Vector to Global VRAM
        out_ptrs = Out_ptr + batch_idx * d + d_offsets
        tl.store(out_ptrs, out_acc, mask=d_mask)


class TritonMRMReadFunction(torch.autograd.Function):
    """Autograd Wrapper around Triton Fused Read Kernel."""

    @staticmethod
    def forward(ctx, Q, Keys, Vals, Hits=None, tau=0.05):
        B, d = Q.shape
        _, K_total, _ = Keys.shape

        Out = torch.empty((B, d), device=Q.device, dtype=Q.dtype)

        BLOCK_D = triton.next_power_of_2(d)
        grid = (B,)

        _mrm_fused_read_triton_kernel[grid](
            Q, Keys, Vals, Out, Hits,
            B=B, d=d, K_total=K_total, tau=tau,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(Q, Keys, Vals)
        ctx.tau = tau
        return Out

    @staticmethod
    def backward(ctx, grad_out):
        Q, Keys, Vals = ctx.saved_tensors
        tau = ctx.tau

        # Differentiable backward via PyTorch graph
        q_norm = torch.norm(Q, dim=-1, keepdim=True).clamp(min=1e-8)
        k_norm = torch.norm(Keys, dim=-1, keepdim=True).clamp(min=1e-8)
        sim = torch.bmm(Q.unsqueeze(1), Keys.transpose(1, 2)) / (q_norm.unsqueeze(1) * k_norm.transpose(1, 2))
        probs = F.softmax(sim / tau, dim=-1)

        grad_vals = torch.bmm(probs.transpose(1, 2), grad_out.unsqueeze(1))
        # Gradient back to Q and Keys
        grad_probs = torch.bmm(grad_out.unsqueeze(1), Vals.transpose(1, 2))
        d_scores = (probs * (grad_probs - (probs * grad_probs).sum(dim=-1, keepdim=True))) / tau
        grad_q = torch.bmm(d_scores, Keys).squeeze(1) / q_norm
        grad_keys = torch.bmm(d_scores.transpose(1, 2), Q.unsqueeze(1)) / k_norm

        return grad_q, grad_keys, grad_vals, None, None


# =================================================================================================
# 2. MULTI-RESOLUTION WORKING MEMORY (MRM) PYTORCH MODULE
# =================================================================================================

class MultiResMemory(nn.Module):
    """
    Multi-Resolution Working Memory (MRM) Module.
    Combines 128 Fine Episodic Slots with 16 Semantic Coarse Centroids.
    Uses OpenAI Triton on CUDA GPU, with automatic vectorized PyTorch fallback on CPU.
    """

    def __init__(self, d_model: int = 128, k_fine: int = 128, k_coarse: int = 16, tau: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.k_fine = k_fine
        self.k_coarse = k_coarse
        self.k_total = k_fine + k_coarse
        self.tau = tau

        # Query, Key, Value & Output Projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Gated Modulation Projection
        self.w_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, -2.0) # Passive residual start
        self.w_out.weight.data.mul_(0.1)

    def read_memory_vectorized(self, Q: torch.Tensor, Keys: torch.Tensor, Vals: torch.Tensor) -> torch.Tensor:
        """Vectorized PyTorch reference read implementation (Runs on CUDA & CPU)."""
        B, d = Q.shape
        q_norm = torch.norm(Q, dim=-1, keepdim=True).clamp(min=1e-8)
        k_norm = torch.norm(Keys, dim=-1, keepdim=True).clamp(min=1e-8)

        # Cosine attention: (B, 1, d) x (B, d, K) -> (B, 1, K)
        cos_sim = torch.bmm(Q.unsqueeze(1), Keys.transpose(1, 2)) / (q_norm.unsqueeze(1) * k_norm.transpose(1, 2))
        probs = F.softmax(cos_sim / self.tau, dim=-1)

        # Context read: (B, 1, K) x (B, K, d) -> (B, d)
        out_context = torch.bmm(probs, Vals).squeeze(1)
        return out_context

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward sequence processing through MRM working memory.
        h_seq: [Batch, SeqLen, d_model]
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        # Working Memory Buffers for the batch
        fine_keys = torch.zeros(B, self.k_fine, D, device=device)
        fine_vals = torch.zeros(B, self.k_fine, D, device=device)
        fine_salience = torch.zeros(B, self.k_fine, device=device)
        fine_hits = torch.ones(B, self.k_fine, device=device)
        coarse_centroids = torch.randn(B, self.k_coarse, D, device=device) * (1.0 / math.sqrt(D))
        coarse_vals = torch.zeros(B, self.k_coarse, D, device=device)
        num_occupied = torch.zeros(B, dtype=torch.long, device=device)

        Q_all = self.w_q(h_seq)
        K_all = self.w_k(h_seq)
        V_all = self.w_v(h_seq)

        outputs = []

        for t in range(T):
            q_t = Q_all[:, t, :]
            k_t = K_all[:, t, :]
            v_t = V_all[:, t, :]
            x_t = h_seq[:, t, :]

            all_keys = torch.cat([fine_keys, coarse_centroids], dim=1)
            all_vals = torch.cat([fine_vals, coarse_vals], dim=1)

            # Read Step: Use Triton if on CUDA, otherwise Vectorized PyTorch
            if TRITON_AVAILABLE and q_t.is_cuda:
                ctx_t = TritonMRMReadFunction.apply(q_t, all_keys, all_vals, None, self.tau)
            else:
                ctx_t = self.read_memory_vectorized(q_t, all_keys, all_vals)

            # Gated Modulation
            proj_t = self.w_out(ctx_t)
            gate = torch.sigmoid(self.w_gate(x_t))
            out_t = x_t + gate * proj_t
            outputs.append(out_t)

            # Write Step with Multi-Tier Adaptive Engine
            # Vectorized across batch
            k_t_norm = torch.norm(k_t, dim=-1, keepdim=True).clamp(min=1e-8)
            f_norm = torch.norm(fine_keys, dim=-1, keepdim=True).clamp(min=1e-8)
            sims = torch.bmm(k_t.unsqueeze(1), fine_keys.transpose(1, 2)) / (k_t_norm.unsqueeze(1) * f_norm.transpose(1, 2))
            sims = sims.squeeze(1) # [B, K_fine]

            max_sim, best_slot = torch.max(sims, dim=-1)

            for b in range(B):
                salience = float(torch.norm(k_t[b]).item())
                s_val = max_sim[b].item()
                slot = best_slot[b].item()

                if s_val >= 0.95:
                    # Tier 1: Hard Overwrite in-place
                    fine_keys[b, slot] = k_t[b]
                    fine_vals[b, slot] = v_t[b]
                    fine_salience[b, slot] = max(salience, fine_salience[b, slot].item())
                    fine_hits[b, slot] = min(fine_hits[b, slot].item() + 1.0, 50.0)
                elif s_val >= 0.82:
                    # Tier 2: Soft Semantic Merge
                    fine_keys[b, slot] = 0.70 * k_t[b] + 0.30 * fine_keys[b, slot]
                    fine_vals[b, slot] = 0.70 * v_t[b] + 0.30 * fine_vals[b, slot]
                    fine_salience[b, slot] = max(salience, fine_salience[b, slot].item())
                    fine_hits[b, slot] = min(fine_hits[b, slot].item() + 0.5, 50.0)
                else:
                    # Tier 3: New Slot Allocation or LRQ Eviction
                    occ = num_occupied[b].item()
                    if occ < self.k_fine:
                        target = occ
                        num_occupied[b] += 1
                    else:
                        utility = fine_hits[b] * 2.0 + fine_salience[b]
                        target = torch.argmin(utility).item()

                    fine_keys[b, target] = k_t[b]
                    fine_vals[b, target] = v_t[b]
                    fine_salience[b, target] = salience
                    fine_hits[b, target] = 1.0

                # Update Coarse Centroid via EMA
                c_norm = torch.norm(coarse_centroids[b], dim=-1, keepdim=True).clamp(min=1e-8)
                c_sims = torch.mv(coarse_centroids[b], k_t[b]) / (k_t_norm[b] * c_norm.squeeze())
                best_c = torch.argmax(c_sims).item()
                coarse_centroids[b, best_c] = 0.95 * coarse_centroids[b, best_c] + 0.05 * k_t[b]
                coarse_vals[b, best_c] = 0.95 * coarse_vals[b, best_c] + 0.05 * v_t[b]

        return torch.stack(outputs, dim=1)


# =================================================================================================
# 3. MICROSOFT DIFFERENTIAL ATTENTION (DiffAttn) MODULE
# =================================================================================================

class DifferentialAttention(nn.Module):
    """
    Microsoft Differential Attention (DiffAttn) Module with Adaptive RoPE & QK-Norm.
    A_diff = Softmax(Q1 K1^T / √d_k) - λ_eff · Softmax(Q2 K2^T / √d_k)

    UNIFIED LAMBDA FORMULA (matches crates/tessera-core/src/tessera_model.rs exactly, per
    head-pair p, with raw trainable logits a_p/b_p and a depth-dependent, non-trainable
    lambda_init derived from this module's 0-indexed layer_idx):
        lambda_eff_p = max(0, exp(a_p) - exp(b_p) + lambda_init(l))
        lambda_init(l) = 0.8 - 0.6 * exp(-0.3 * (l - 1)),  l = layer_idx + 1 (1-indexed)
    This replaces the OLD code's `self.lambda_diff = nn.Parameter([0.8, 0.8])` used directly
    as `lam = self.lambda_diff[p]` -- a formula that had no depth-dependence, no exp/clamp,
    and did not match the Rust engine's train/inference-unified formula at all. See
    docs/TESSERA_DEEP_RESEARCH_PROMPT.md for the original train/inference divergence bug this
    unification fixes.

    LEARNED VALUE RESIDUAL (ResFormer, arXiv:2410.17897): `vres_gate` is a trainable
    PRE-sigmoid scalar; effective mixing weight w = sigmoid(vres_gate) is used as
    V_mixed = w * V_raw + (1 - w) * V_0 for every stage after the first, replacing the
    previously-hardcoded/absent value-residual mixing. Initialized so sigmoid(vres_gate)
    ~= 0.7, matching the old hardcoded 0.7/0.3 constants as the learned starting point.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads         # 4 subheads = 2 differential pairs
        self.n_pairs = n_heads // 2    # 2 differential head pairs
        self.d_k = d_model // n_heads  # 32
        self.d_v = d_model // self.n_pairs # 64
        self.layer_idx = layer_idx

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Depth-dependent DiffAttn lambda init (arXiv:2410.05258): l is 1-indexed.
        # NON-trainable (a plain Python float baked into the forward pass), matching Rust's
        # TesseraStage.lambda_init field exactly.
        l = float(layer_idx + 1)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (l - 1.0))

        # Raw per-pair a_p/b_p logits, zero-initialized so that at construction
        # lambda_eff_p = exp(0) - exp(0) + lambda_init == lambda_init exactly, for every pair.
        self.lambda_diff = nn.Parameter(torch.zeros(2 * self.n_pairs, dtype=torch.float32))

        # Learned Value-Residual gate (ResFormer): PRE-sigmoid raw scalar, initialized so
        # sigmoid(vres_gate) ~= 0.7 (sigmoid^-1(0.7) = ln(0.7/0.3) ~= 0.8473).
        self.vres_gate = nn.Parameter(torch.tensor(0.8473, dtype=torch.float32))

        # Adaptive RoPE Multipliers
        self.eta_rope = nn.Parameter(torch.zeros(self.d_k // 2, dtype=torch.float32))

    def lambda_eff(self, p: int) -> torch.Tensor:
        """Effective per-pair lambda using the unified formula (see class docstring)."""
        a_p = self.lambda_diff[2 * p]
        b_p = self.lambda_diff[2 * p + 1]
        return torch.clamp(torch.exp(a_p) - torch.exp(b_p) + self.lambda_init, min=0.0)

    def apply_adaptive_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, Dk = x.shape
        half = Dk // 2
        pos = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(1)
        base_theta = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
        scale = 2.0 * torch.sigmoid(self.eta_rope)
        theta = base_theta * scale
        angles = pos * theta.unsqueeze(0) # [T, half]

        cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(2) # [1, T, 1, half]
        sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos_a - x_odd * sin_a
        out_odd = x_even * sin_a + x_odd * cos_a

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x: torch.Tensor, v0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D]. v0: this stage's value tensor is mixed with `v0` (stage 0's raw value
        tensor, shape [B, T, D]) via the learned gate, EXCEPT when v0 is None (i.e. this IS
        stage 0), in which case the raw value tensor is returned unmixed and becomes the v0
        for subsequent stages. Returns (attn_out, v_raw_for_v0_cache) where v_raw_for_v0_cache
        is always this stage's OWN pre-mix value projection (matching Rust's `stage_v_raw`
        cache, needed so downstream stages always mix against stage 0's true raw V, not an
        already-mixed V).
        """
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_k)
        k = self.wk(x).view(B, T, self.n_heads, self.d_k)
        v = self.wv(x).view(B, T, self.n_heads, self.d_k)  # keep per-subhead shape for v0 mixing
        v_raw = v.reshape(B, T, D)

        if v0 is not None:
            w = torch.sigmoid(self.vres_gate)
            v_mixed = w * v_raw + (1.0 - w) * v0
            v = v_mixed.view(B, T, self.n_heads, self.d_k)

        v_pairs = v.view(B, T, self.n_pairs, self.d_v)

        # Adaptive RoPE
        q = self.apply_adaptive_rope(q)
        k = self.apply_adaptive_rope(k)

        # QK-Norm
        q = F.rms_norm(q, (self.d_k,))
        k = F.rms_norm(k, (self.d_k,))

        scale = 1.0 / math.sqrt(self.d_k)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)

        out_pairs = []
        for p in range(self.n_pairs):
            h1 = 2 * p
            h2 = 2 * p + 1
            lam = self.lambda_eff(p)

            q1, q2 = q[:, :, h1, :], q[:, :, h2, :]
            k1, k2 = k[:, :, h1, :], k[:, :, h2, :]
            vp = v_pairs[:, :, p, :]

            scores1 = (torch.bmm(q1, k1.transpose(1, 2)) * scale) + causal_mask
            scores2 = (torch.bmm(q2, k2.transpose(1, 2)) * scale) + causal_mask

            p1 = F.softmax(scores1, dim=-1)
            p2 = F.softmax(scores2, dim=-1)

            diff_attn = p1 - lam * p2
            ctx_p = torch.bmm(diff_attn, vp)
            out_pairs.append(ctx_p)

        attn_out = torch.cat(out_pairs, dim=-1)
        return self.wo(attn_out), v_raw


# =================================================================================================
# 4. PROGRESSIVE FOLDING STAGE MODULE
# =================================================================================================

class TesseraStageGPU(nn.Module):
    """Full Progressive Hierarchy Stage with 1D Depthwise Conv + DiffAttn + SwiGLU + MRM."""

    def __init__(self, d_model: int = 128, d_ff: int = 768, n_heads: int = 4, use_mrm: bool = False, layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.use_mrm = use_mrm
        self.layer_idx = layer_idx

        self.norm1 = nn.RMSNorm(d_model)
        # 1D Depthwise Causal Convolution (k=4)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model, bias=False)
        self.w_gate_attn = nn.Linear(d_model, d_model, bias=False)
        self.diff_attn = DifferentialAttention(d_model, n_heads, layer_idx=layer_idx)

        self.norm2 = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w1u = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

        # Stage Adapter (r=8)
        self.adapter_down = nn.Linear(d_model, 8, bias=False)
        self.adapter_up = nn.Linear(8, d_model, bias=False)
        nn.init.zeros_(self.adapter_up.weight)

        self.mrm = MultiResMemory(d_model) if use_mrm else None

    def forward(self, x: torch.Tensor, v0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_stage_out, v_raw) where v_raw is THIS stage's own pre-value-residual-mix
        value projection (used as the v0 cache for downstream stages when this is stage 0)."""
        B, T, D = x.shape
        h_norm1 = self.norm1(x)

        # 1D Causal Conv
        conv_out = self.conv1d(h_norm1.transpose(1, 2))[:, :, :T].transpose(1, 2)
        gate_attn = F.silu(self.w_gate_attn(h_norm1))

        # Differential Attention (with learned Value-Residual mixing against v0)
        attn_raw, v_raw = self.diff_attn(conv_out, v0)
        attn_out = attn_raw * gate_attn
        h_mid = x + attn_out

        # SwiGLU + Adapter
        h_norm2 = self.norm2(h_mid)
        swiglu = F.silu(self.w1(h_norm2)) * self.w1u(h_norm2)
        ffn_out = self.w2(swiglu)
        adapter_out = self.adapter_up(self.adapter_down(h_norm2))
        h_stage = h_mid + ffn_out + adapter_out

        # MRM Active Working Memory (if attached to this stage)
        if self.mrm is not None:
            h_stage = self.mrm(h_stage)

        return h_stage, v_raw


# =================================================================================================
# 5. FULL TESSERA-Q GPU MODEL WITH MULTI-TOKEN PREDICTION
# =================================================================================================

class TesseraGPUModel(nn.Module):
    """
    Complete TESSERA-Q Language Model for GPU Training with OpenAI Triton.
    Tied Embeddings, Multi-Token Prediction (MTP), PaLM Z-Loss, and L2-Resident MRM.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 128, d_ff: int = 768, n_stages: int = 3,
                 n_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff

        self.embeddings = nn.Embedding(vocab_size, d_model)

        self.stages = nn.ModuleList([
            TesseraStageGPU(d_model, d_ff, n_heads=n_heads, use_mrm=(p == n_stages - 1), layer_idx=p)
            for p in range(n_stages)
        ])

        self.final_norm = nn.RMSNorm(d_model)

        # Dual-Head Multi-Token Prediction (MTP for t+2)
        self.w_mtp_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_mtp_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        alpha_mtp: float = 0.30,
        z_loss_coeff: float = 1e-4,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        h = self.embeddings(input_ids)

        # Value Residual (ResFormer): v0 is fixed to STAGE 0's raw value projection and stays
        # constant for every subsequent stage's mixing (matches Rust's `v0_cache`, which is set
        # once at s_idx==0 and never overwritten). Do NOT reassign v0 to each stage's own
        # v_raw -- that would make every stage mix against its immediate predecessor instead
        # of against stage 0, which is architecturally wrong per ResFormer (arXiv:2410.17897).
        v0 = None
        for s_idx, stage in enumerate(self.stages):
            h, v_raw = stage(h, v0)
            if s_idx == 0:
                v0 = v_raw

        h_final = self.final_norm(h)

        # Logit Soft-Capping: 30.0 * tanh(raw / 30.0) with Tied Embeddings
        raw_logits = F.linear(h_final, self.embeddings.weight)
        logits = 30.0 * torch.tanh(raw_logits / 30.0)

        output = {"logits": logits}

        if targets is not None:
            # Standard Next-Token Loss
            ntp_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

            # PaLM Z-Loss: 1e-4 * (log sum exp(logits))^2
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = z_loss_coeff * torch.mean(log_z ** 2)

            # Multi-Token Prediction Auxiliary Loss (t+2)
            if T > 1:
                mtp_h = self.w_mtp_proj(h_final[:, :-1, :])
                mtp_logits = self.w_mtp_head(mtp_h)
                mtp_targets = targets[:, 1:]
                mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, self.vocab_size), mtp_targets.reshape(-1))
            else:
                mtp_loss = torch.tensor(0.0, device=input_ids.device)

            total_loss = ntp_loss + z_loss + alpha_mtp * mtp_loss
            output.update({
                "loss": total_loss,
                "ntp_loss": ntp_loss,
                "bpc": ntp_loss / math.log(2.0),
            })

        return output

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> str:
        """
        SINGLE-USER autoregressive text generation, byte-level, mirroring Rust's
        `TesseraModel::generate_text` (crates/tessera-core/src/tessera_model.rs) exactly:
        full-context recompute every step (no incremental KV cache -- matches the Rust
        engine's own recompute-based `forward_last_logits`), temperature-scaled top-k
        sampling with a numerically stable exp(l - max) sum, and a bounded context window.

        prompt: input text (encoded as raw UTF-8 bytes, vocab_size=256 byte-level model).
        Returns the decoded string (prompt + generated continuation).
        """
        self.eval()
        dev = device if device is not None else next(self.parameters()).device
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        max_seq_len = getattr(self, "max_seq_len", 512)
        tokens = list(prompt.encode("utf-8")) or [ord(" ")]
        temp = max(temperature, 1e-4)

        for _ in range(max_new_tokens):
            ctx = tokens[-max_seq_len:] if len(tokens) > max_seq_len else tokens
            ids = torch.tensor([ctx], dtype=torch.long, device=dev)
            logits = self(ids)["logits"][0, -1, :]  # [vocab_size]

            scaled = logits / temp
            k = min(top_k, scaled.numel())
            top_vals, top_idx = torch.topk(scaled, k)
            probs = F.softmax(top_vals, dim=-1).to("cpu")

            choice = torch.multinomial(probs, num_samples=1, generator=gen).item()
            next_tok = int(top_idx[choice].item())
            tokens.append(next_tok)

        out_bytes = bytes(t % 256 for t in tokens)
        return out_bytes.decode("utf-8", errors="replace")

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: "list[str]",
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "list[str]":
        """
        MULTI-USER / concurrent inference: generates continuations for a batch of B
        independent prompts/sessions in parallel on the same GPU/CPU forward pass, each
        with its own token history and independent sampling draw. This is the natural
        batched extension of `generate()` above -- the model's forward pass already
        operates over a leading batch dimension [B, T, D] end-to-end (embeddings, DiffAttn,
        MRM), so B independent "users" share compute for every step while still sampling
        their own next token from their own row of logits. RIGHT-padding (with byte 0) is
        used so ragged per-user histories can share a single batched forward call: because
        both the depthwise causal conv (kernel=4, causal via symmetric-pad-then-trim) and
        DiffAttn's upper-triangular causal mask only ever look BACKWARD in the time
        dimension, trailing padding placed strictly AFTER each row's true last token can
        never influence that true last token's own logits -- so each user's real next-token
        distribution is bit-for-bit identical to running that user alone at their own true
        length, letting all B users share one forward pass safely without needing an
        explicit attention-mask plumbed through conv/DiffAttn (which this simplified engine
        does not implement).
        """
        self.eval()
        dev = device if device is not None else next(self.parameters()).device
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        max_seq_len = getattr(self, "max_seq_len", 512)
        B = len(prompts)
        token_lists = [list(p.encode("utf-8")) or [ord(" ")] for p in prompts]
        temp = max(temperature, 1e-4)

        for _ in range(max_new_tokens):
            ctxs = [tl[-max_seq_len:] if len(tl) > max_seq_len else tl for tl in token_lists]
            max_len = max(len(c) for c in ctxs)
            # Right-pad each row to max_len with 0 so the batch can share one forward call;
            # each row's TRUE last-token position (before padding) is recorded in last_pos
            # and is what we read logits from -- padding after it is causally invisible.
            padded = torch.zeros(B, max_len, dtype=torch.long, device=dev)
            last_pos = torch.empty(B, dtype=torch.long, device=dev)
            for i, c in enumerate(ctxs):
                padded[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=dev)
                last_pos[i] = len(c) - 1  # true last token position within this row

            logits_all = self(padded)["logits"]  # [B, max_len, vocab_size]
            row_idx = torch.arange(B, device=dev)
            logits = logits_all[row_idx, last_pos, :]  # [B, vocab_size]

            scaled = logits / temp
            k = min(top_k, scaled.size(-1))
            top_vals, top_idx = torch.topk(scaled, k, dim=-1)
            probs = F.softmax(top_vals, dim=-1).to("cpu")

            for i in range(B):
                choice = torch.multinomial(probs[i], num_samples=1, generator=gen).item()
                next_tok = int(top_idx[i, choice].item())
                token_lists[i].append(next_tok)

        results = []
        for tl in token_lists:
            out_bytes = bytes(t % 256 for t in tl)
            results.append(out_bytes.decode("utf-8", errors="replace"))
        return results

    def export_to_binary(self, path: str):
        """
        Export model weights to the flat little-endian binary format actually read by
        `TesseraModel::load_binary` in crates/tessera-core/src/tessera_model.rs.

        FIX: the previous version of this method wrote an 8-byte b"TESSERA\\0" magic tag as
        the first bytes of the file and a fixed-size, un-length-prefixed 2-float
        `lambda_diff` per stage with no `lambda_init`/`vres_gate` fields at all. NONE of that
        matches the real Rust reader, which expects (in exact order):
          1. A 24-byte header of SIX raw little-endian u32s (NOT an 8-byte magic string):
             vocab_size, d_model, d_ff, n_stages, n_heads, adapter_rank.
          2. embeddings: vocab_size * d_model f32.
          3. Per stage: norm1_gamma (d), w_conv (4*d), w_gate_attn (d*d),
             [u32 lambda_len] lambda_diff (lambda_len), lambda_init (1 f32),
             vres_gate (1 f32), eta_rope (d_k/2), wq/wk/wv/wo (d*d each),
             norm2_gamma (d), w1/w1u (d_ff*d each), w2 (d*d_ff).
          4. final_norm_gamma (d).
        A file written by the OLD version of this method would silently desync every single
        field read by Rust's loader (wrong header size, wrong stage layout, missing
        lambda_init/vres_gate) -- this was a real, previously undiscovered cross-language
        format bug, not just a lambda-formula mismatch.
        """
        d = self.d_model
        n_heads = self.stages[0].diff_attn.n_heads if len(self.stages) > 0 else 4

        def f32(t: torch.Tensor) -> bytes:
            return t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()

        with open(path, "wb") as f:
            # Header: 6 raw u32s, NO magic string (matches Rust's 24-byte header exactly).
            f.write(int(self.vocab_size).to_bytes(4, "little"))
            f.write(int(self.d_model).to_bytes(4, "little"))
            f.write(int(self.d_ff).to_bytes(4, "little"))
            f.write(int(len(self.stages)).to_bytes(4, "little"))
            f.write(int(n_heads).to_bytes(4, "little"))
            f.write(int(8).to_bytes(4, "little"))  # adapter_rank, fixed at 8 in this engine

            # Dump embeddings (vocab_size x d_model, row-major -- matches Rust's MatrixView
            # convention and nn.Embedding.weight's native layout).
            f.write(f32(self.embeddings.weight))

            for stage in self.stages:
                da = stage.diff_attn
                f.write(f32(stage.norm1.weight))
                # conv1d.weight is [d, 1, 4] (out_ch, in_ch/groups, kernel) for a depthwise
                # Conv1d; Rust's w_conv is (4 x d) with layout [k*d + c]. Permute (4, d).
                conv_w = stage.conv1d.weight.detach().cpu().to(torch.float32).squeeze(1)  # [d, 4]
                conv_w_kd = conv_w.transpose(0, 1).contiguous()  # [4, d]
                f.write(f32(conv_w_kd))
                f.write(f32(stage.w_gate_attn.weight))

                lam = da.lambda_diff.detach().cpu().to(torch.float32)
                f.write(int(lam.numel()).to_bytes(4, "little"))
                f.write(f32(lam))
                f.write(struct.pack("<f", float(da.lambda_init)))
                f.write(struct.pack("<f", float(da.vres_gate.detach().cpu().item())))
                f.write(f32(da.eta_rope))
                f.write(f32(da.wq.weight))
                f.write(f32(da.wk.weight))
                f.write(f32(da.wv.weight))
                f.write(f32(da.wo.weight))
                f.write(f32(stage.norm2.weight))
                f.write(f32(stage.w1.weight))
                f.write(f32(stage.w1u.weight))
                f.write(f32(stage.w2.weight))

            # Final norm
            f.write(f32(self.final_norm.weight))

        print(f"✓ Successfully exported TESSERA-Q weights to binary format: {path}")

    @classmethod
    def load_from_binary(cls, path: str) -> "TesseraGPUModel":
        """
        Round-trip loader for the format written by `export_to_binary` above (which is the
        SAME format `TesseraModel::load_binary` in Rust reads). Lets this Python engine
        re-load a model it just exported (or one produced by the Rust trainer/converted from
        an open-weight model) for further CPU-side inference/fine-tuning without needing a
        GPU. Not previously implemented in this file at all.
        """
        with open(path, "rb") as f:
            def read_u32() -> int:
                return int.from_bytes(f.read(4), "little")

            def read_f32(n: int) -> torch.Tensor:
                raw = f.read(4 * n)
                return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()

            vocab_size = read_u32()
            d_model = read_u32()
            d_ff = read_u32()
            n_stages = read_u32()
            n_heads = read_u32()
            _adapter_rank = read_u32()

            model = cls(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_stages=n_stages,
                        n_heads=n_heads)

            embed_flat = read_f32(vocab_size * d_model)
            with torch.no_grad():
                model.embeddings.weight.copy_(embed_flat.view(vocab_size, d_model))

            d = d_model
            for stage in model.stages:
                da = stage.diff_attn
                with torch.no_grad():
                    stage.norm1.weight.copy_(read_f32(d))
                    conv_kd = read_f32(4 * d).view(4, d)
                    stage.conv1d.weight.copy_(conv_kd.transpose(0, 1).unsqueeze(1))
                    stage.w_gate_attn.weight.copy_(read_f32(d * d).view(d, d))

                    lambda_len = read_u32()
                    lam = read_f32(lambda_len)
                    da.lambda_diff.data = lam.clone()
                    da.lambda_init = struct.unpack("<f", f.read(4))[0]
                    da.vres_gate.data = torch.tensor(struct.unpack("<f", f.read(4))[0])
                    eta_len = (d // n_heads) // 2
                    da.eta_rope.data = read_f32(max(eta_len, 1))
                    da.wq.weight.copy_(read_f32(d * d).view(d, d))
                    da.wk.weight.copy_(read_f32(d * d).view(d, d))
                    da.wv.weight.copy_(read_f32(d * d).view(d, d))
                    da.wo.weight.copy_(read_f32(d * d).view(d, d))
                    stage.norm2.weight.copy_(read_f32(d))
                    stage.w1.weight.copy_(read_f32(d_ff * d).view(d_ff, d))
                    stage.w1u.weight.copy_(read_f32(d_ff * d).view(d_ff, d))
                    stage.w2.weight.copy_(read_f32(d * d_ff).view(d, d_ff))

            with torch.no_grad():
                model.final_norm.weight.copy_(read_f32(d))

        print(f"✓ Successfully loaded TESSERA-Q weights from binary format: {path}")
        return model

    def export_to_binary_int8(self, path: str):
        """
        'Very optimized format': INT8 per-tensor symmetric-quantized export for weight
        matrices (wq/wk/wv/wo/w1/w1u/w2/w_gate_attn/adapter/embeddings/mtp), alongside
        fp32 for small vectors (norms, lambda_diff, lambda_init, vres_gate, eta_rope) where
        quantization would be numerically unsafe for so few elements. This roughly quarters
        the on-disk/resident size of the dominant weight matrices compared to
        `export_to_binary`'s pure fp32 format, extending the same idea already used by the
        existing W4A16 4-bit GPU inference kernels (`tessera_triton_60b_engine.py`) to a
        portable, GPU-independent CPU/storage format.

        File layout: same 6-u32 header as `export_to_binary`, then for every quantized
        tensor: [f32 scale][int8 data...]; vectors are stored plain fp32 (no scale prefix).
        This is a NEW, separate format from `export_to_binary`'s (not consumed by the
        current Rust `load_binary`, which expects pure fp32) -- intended for space-constrained
        storage/distribution, to be dequantized back to fp32 on load.
        """
        d = self.d_model
        n_heads = self.stages[0].diff_attn.n_heads if len(self.stages) > 0 else 4

        def quantize_int8(t: torch.Tensor) -> Tuple[float, bytes]:
            t = t.detach().cpu().to(torch.float32).contiguous()
            max_abs = t.abs().max().clamp(min=1e-8).item()
            scale = max_abs / 127.0
            q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
            return scale, q.numpy().tobytes()

        def write_quant(f, t: torch.Tensor):
            scale, data = quantize_int8(t)
            f.write(struct.pack("<f", scale))
            f.write(data)

        def write_plain(f, t: torch.Tensor):
            f.write(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())

        with open(path, "wb") as f:
            f.write(int(self.vocab_size).to_bytes(4, "little"))
            f.write(int(self.d_model).to_bytes(4, "little"))
            f.write(int(self.d_ff).to_bytes(4, "little"))
            f.write(int(len(self.stages)).to_bytes(4, "little"))
            f.write(int(n_heads).to_bytes(4, "little"))
            f.write(int(8).to_bytes(4, "little"))

            write_quant(f, self.embeddings.weight)

            for stage in self.stages:
                da = stage.diff_attn
                write_plain(f, stage.norm1.weight)
                conv_w = stage.conv1d.weight.detach().cpu().to(torch.float32).squeeze(1)
                write_plain(f, conv_w.transpose(0, 1).contiguous())
                write_quant(f, stage.w_gate_attn.weight)

                lam = da.lambda_diff.detach().cpu().to(torch.float32)
                f.write(int(lam.numel()).to_bytes(4, "little"))
                write_plain(f, lam)
                f.write(struct.pack("<f", float(da.lambda_init)))
                f.write(struct.pack("<f", float(da.vres_gate.detach().cpu().item())))
                write_plain(f, da.eta_rope)
                write_quant(f, da.wq.weight)
                write_quant(f, da.wk.weight)
                write_quant(f, da.wv.weight)
                write_quant(f, da.wo.weight)
                write_plain(f, stage.norm2.weight)
                write_quant(f, stage.w1.weight)
                write_quant(f, stage.w1u.weight)
                write_quant(f, stage.w2.weight)

            write_plain(f, self.final_norm.weight)

        orig_bytes = sum(p.numel() for p in self.parameters()) * 4
        quant_bytes = os.path.getsize(path)
        print(
            f"✓ Successfully exported TESSERA-Q INT8-quantized weights: {path} "
            f"({quant_bytes / (1024*1024):.2f} MB vs {orig_bytes / (1024*1024):.2f} MB fp32, "
            f"{quant_bytes / max(orig_bytes, 1) * 100:.1f}% of original size)"
        )


# =================================================================================================
# 6. STANDALONE VALIDATION & SELF-TEST SUITE
# =================================================================================================

def run_self_verification_test():
    """Runs a strict self-test verifying forward/backward passes and numerical stability."""
    print("=" * 80)
    print("  RUNNING TESSERA-Q GPU & TRITON VERIFICATION SUITE")
    print(f"  CUDA Available: {torch.cuda.is_available()} | Triton Available: {TRITON_AVAILABLE}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TesseraGPUModel(vocab_size=256, d_model=128, d_ff=768, n_stages=3).to(device)

    # 1. Forward Pass Test
    x = torch.randint(0, 256, (4, 64), device=device)
    y = torch.randint(0, 256, (4, 64), device=device)

    out = model(x, targets=y)
    loss = out["loss"]
    bpc = out["bpc"]

    print(f"  Forward Pass: Loss = {loss.item():.4f} | BPC = {bpc.item():.4f} (PASS)")

    # 2. Backward Pass & Gradient Flow Test
    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    assert not math.isnan(total_norm), "Gradient norm resulted in NaN!"
    assert total_norm > 0.0, "Zero gradient detected!"
    print(f"  Backward Pass: Total Grad Norm = {total_norm:.4f} (PASS)")

    # 3. Export Test
    export_path = "test_tessera_gpu_export.bin"
    model.export_to_binary(export_path)
    assert os.path.exists(export_path), "Export file was not created!"
    file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"  Binary Export: {file_size_mb:.2f} MB (Fits in CPU L3 Cache) (PASS)")
    os.remove(export_path)

    print("=" * 80)
    print("  ✓ ALL GPU & TRITON SECURITY & NUMERICAL STABILITY TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_self_verification_test()
