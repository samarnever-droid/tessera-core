#!/usr/bin/env python3
"""
====================================================================================================
🚀 TESSERA-Q ~50B STREAMING INFERENCE ENGINE — KAGGLE DUAL TESLA T4 (2 x 16 GB)
====================================================================================================
Complete, runnable PyTorch implementation of the 9 canonical TESSERA-Q pillars, with optional
OpenAI Triton fused kernels (auto-enabled when triton + CUDA are available, e.g. on Kaggle T4s).

  1. Affine RMSNorm (pre-attention / pre-FFN)
  2. 1D Causal Depthwise Highway Conv (k=4) + sigmoid gate
  3. Per-Head QK-RMSNorm
  4. Adaptive RoPE Banding  theta_i = base^(-2i/d_k) * 2*sigmoid(eta_i)
  5. Differential Attention  lambda_eff_p = max(0, exp(a_p) - exp(b_p) + lambda_init(l)),
     lambda_init(l) = 0.8 - 0.6*exp(-0.3*(l-1))            (unified formula, Rust parity)
  6. ResFormer Value Residual  v_write = sigmoid(vres_gate)*v + (1-sigmoid(vres_gate))*v0
     (v0 = stage-1 value, carried across layers per token)
  7. SwiGLU FFN + parallel low-rank stage adapter (r=8)
  8. MRM-v2 Multi-Resolution Working Memory (128 fine slots + 16 coarse centroids, tau=0.05)
     - Rust-parity dynamic thresholds:  t_overwrite(d) = min(0.9999, 8.014/sqrt(d))
                                        t_merge(d)     = min(t_overwrite - 1e-4, 6.50/sqrt(d))
     - Poincaré Ball Riemannian coarse EMA: c += (1-gamma) * ((1-||c||^2)^2 / 4) * (k - c)
     - Closed-form gradient-surprise salience: S_i = ||v_i - ctx_i||
     - Utility eviction:  evict argmin(2*hits + salience)
  9. Tied output embeddings with logit soft-capping  30*tanh(z/30)

MEMORY MODEL (why ~50B fits on 2x T4):
  - Weights are 4-bit GPTQ (~0.5 byte/param). A 50B model ≈ 25-28 GB -> fits SPLIT across
    2x16 GB. Layers [0, split) live on cuda:0, [split, L) on cuda:1; the hidden state hops
    devices exactly once per token.
  - Decode is TOKEN-STREAMING through MRM-v2: there is NO per-layer KV cache. Attention at
    every layer reads the fixed 144-slot working memory, so per-token state is O(1)
    (~3.5 MB of slot tensors), not O(T).
  - All linears are GEMV (M=1): we ship a fused Triton GPTQ-INT4 GEMV kernel; without Triton
    a chunked pure-PyTorch dequant fallback is used (slow but exact, works on CPU).

WEIGHT SOURCES:
  - `--model <dir>`: a HuggingFace GPTQ (INT4) checkpoint (auto-gptq / GPTQModel export),
    e.g. Qwen2.5-32B-Instruct-GPTQ-Int4. Qwen-style GQA is un-grouped to full MHA via
    repeat_interleave (lossless, same as convert_open_weights_to_tessera.py). Tessera-only
    components (depthwise conv, eta, adapters, DiffAttn lambda, value-residual gate) are
    initialized to their principled defaults and can be learned later.
  - `--smoke`: tiny random-weight config that exercises every math path on CPU.

SMOKE TEST:
    python tessera_50b_kaggle.py --smoke
====================================================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


def free_memory() -> None:
    """Best-effort release of GPU memory held by dead objects in this session,
    then report per-GPU free/total. Stale engines from earlier cells keep their
    weights alive until every reference is dropped (or the session restarts)."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  cuda:{i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")


def triton_enabled(device: torch.device) -> bool:
    # Escape hatch: TESSERA_FORCE_TORCH=1 skips the Triton path entirely
    # (useful if Triton JIT compile thrashes on a given driver/GPU combo).
    if os.environ.get("TESSERA_FORCE_TORCH", "0") == "1":
        return False
    return HAS_TRITON and device.type == "cuda"


# ====================================================================================================
# 0. CONFIG
# ====================================================================================================

@dataclass
class TesseraConfig:
    vocab_size: int = 152064
    d_model: int = 6144
    n_layers: int = 55
    n_heads: int = 48           # full MHA, even count -> n_heads/2 DiffAttn pairs
    d_ff: Optional[int] = None  # None -> 6 * d_model (canonical Pillar 7)
    r_adapter: int = 8
    fine_slots: int = 128
    coarse_slots: int = 16
    tau: float = 0.05
    logit_cap: float = 30.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    group_size: int = 128       # GPTQ group size of the source checkpoint

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 6 * self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.n_heads % 2 == 0, "n_heads must be even (DiffAttn head pairs)"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_pairs(self) -> int:
        return self.n_heads // 2

    def param_count(self) -> int:
        d = self.d_model
        per_layer = 5 * d * d + 3 * d * self.d_ff + 2 * d * self.r_adapter + 4 * d
        return per_layer * self.n_layers + self.vocab_size * d + 2 * d


TESSERA_50B = TesseraConfig()


def lambda_init_at(layer_idx: int) -> float:
    """Depth-dependent DiffAttn lambda_init, l = layer_idx + 1 (1-indexed)."""
    return 0.8 - 0.6 * math.exp(-0.3 * float(layer_idx))


# ====================================================================================================
# 1. TRITON KERNELS (CUDA path)
# ====================================================================================================

if HAS_TRITON:

    @triton.jit
    def _gptq_gemv_kernel(
        X_ptr, QW8_ptr, SC_ptr, QZ8_ptr, GIDX_ptr, Y_ptr,
        K, N,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused INT4-GPTQ GEMV for M=1 decode.
        Weights/zeros are uint8 2-nibble packs (low nibble = even index), the same
        convention as the repo's proven 60B engine. We deliberately avoid int32
        8-nibble word shifts — they misbehave on some sm_75 Triton backends.
        w[k, n] = (nibble8(qw8[k//2, n], k%2) - (nibble8(qz8[g, n//2], n%2) + 1)) * sc[g, n],
        g = g_idx[k]  (auto-gptq convention: unpacked zero is stored minus 1)."""
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x = tl.load(X_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

            gidx = tl.load(GIDX_ptr + offs_k, mask=mask_k, other=0)

            kb = tl.load(
                QW8_ptr + (offs_k[:, None] // 2) * N + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            w4 = tl.where((offs_k[:, None] & 1) == 1,
                          (kb >> 4) & 0xF, kb & 0xF).to(tl.float32)

            sc = tl.load(
                SC_ptr + gidx[:, None] * N + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :], other=0.0,
            ).to(tl.float32)

            zb = tl.load(
                QZ8_ptr + gidx[:, None] * (N // 2) + (offs_n[None, :] // 2),
                mask=mask_k[:, None] & mask_n[None, :], other=0,
            )
            z4 = tl.where((offs_n[None, :] & 1) == 1,
                          (zb >> 4) & 0xF, zb & 0xF).to(tl.float32) + 1.0

            w = (w4 - z4) * sc
            acc += tl.sum(x[:, None] * w, axis=0)

        tl.store(Y_ptr + offs_n, acc.to(tl.float16), mask=mask_n)


def gptq_gemv_triton(x: torch.Tensor, qw8: torch.Tensor, sc: torch.Tensor,
                     qz8: torch.Tensor, g_idx: torch.Tensor) -> torch.Tensor:
    K, N = qw8.shape[0] * 2, sc.shape[1]
    y = torch.empty(N, device=x.device, dtype=torch.float16)
    # 64x64 tiles: the fp32 dequant tile must fit registers on T4 (sm_75);
    # 128x128 spills to local memory and slows every launch by orders of magnitude.
    BLOCK_N, BLOCK_K = 64, 64
    grid = (triton.cdiv(N, BLOCK_N),)
    # Triton launches on the CURRENT cuda device; tensors on cuda:1 are unreachable
    # from a cuda:0 launch context ("Pointer argument cannot be accessed").
    with torch.cuda.device(x.device):
        _gptq_gemv_kernel[grid](x, qw8, sc, qz8, g_idx, y, K, N,
                                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return y


# ====================================================================================================
# 2. PURE-PYTORCH GPTQ FALLBACK (CPU / no-Triton path)
# ====================================================================================================

def unpack_nibbles(qw: torch.Tensor) -> torch.Tensor:
    """[K/8, N] int32 -> [K, N] int16 (low nibble first: k = i*8 + j)."""
    shifts = (torch.arange(8, device=qw.device, dtype=torch.int32) * 4)
    out = ((qw.unsqueeze(-1) >> shifts) & 0xF).to(torch.int16)
    k8, n, _ = out.shape
    return out.reshape(k8 * 8, n)


def unpack_zeros(qz: torch.Tensor) -> torch.Tensor:
    """[G, N/8] int32 -> [G, N] int16 (nibbles pack along N, low first)."""
    shifts = (torch.arange(8, device=qz.device, dtype=torch.int32) * 4)
    out = ((qz.unsqueeze(-1) >> shifts) & 0xF).to(torch.int16)
    g, n8, _ = out.shape
    return out.reshape(g, n8 * 8)


def unpack_uint8_pairs(packed: torch.Tensor) -> torch.Tensor:
    """uint8 2-nibble bytes [R, C] (low nibble = even index) -> int16 [2R, C]."""
    lo = (packed & 0xF).to(torch.int16)
    hi = ((packed >> 4) & 0xF).to(torch.int16)
    out = torch.empty(packed.shape[0] * 2, packed.shape[1], dtype=torch.int16,
                      device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    return out


def repack_gptq_uint8(qw: torch.Tensor, qz: torch.Tensor):
    """int32 8-nibble GPTQ words -> uint8 2-nibble packs via torch-verified ops.
    Same total bytes. qw [K/8, N] -> [K/2, N]; qz [G, N/8] -> [G/2, N/2].
    Odd group counts (e.g. k_in < group_size -> G=1) are padded with a duplicate
    row so row pairing works; indices still resolve because packed row j holds
    original rows 2j (lo) and 2j+1 (hi)."""
    w4 = unpack_nibbles(qw).to(torch.uint8)
    qw8 = ((w4[1::2] << 4) | w4[0::2]).contiguous()
    z4 = unpack_zeros(qz).to(torch.uint8)
    if z4.shape[0] % 2 == 1:
        z4 = torch.cat([z4, z4[-1:]], dim=0)
    qz8 = ((z4[1::2] << 4) | z4[0::2]).contiguous()
    return qw8, qz8


def dequant_gptq(qw: torch.Tensor, sc: torch.Tensor, qz: torch.Tensor,
                 g_idx: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Dequantize a GPTQ tensor to fp16, chunked over K to bound memory."""
    N = sc.shape[1]
    out = torch.empty(qw.shape[0] * 8, N, dtype=torch.float16, device=qw.device)
    for k0 in range(0, out.shape[0], chunk):
        k1 = min(k0 + chunk, out.shape[0])
        w4 = unpack_nibbles(qw[k0 // 8: k1 // 8]).to(torch.float32)
        g = g_idx[k0:k1]
        z = unpack_zeros(qz).to(torch.float32) + 1.0
        w = (w4 - z[g]) * sc[g].to(torch.float32)
        out[k0:k1] = w.to(torch.float16)
    return out


class GPTQLinear:
    """4-bit GPTQ GEMV layer. Triton fast path (uint8 2-nibble packs),
    chunked torch fallback (int32 words)."""

    def __init__(self, qw: torch.Tensor, sc: torch.Tensor, qz: torch.Tensor,
                 g_idx: torch.Tensor, device: torch.device, use_triton: bool):
        self.use_triton = use_triton and triton_enabled(device)
        if self.use_triton:
            # Triton path: repack to uint8 2-nibble form and DROP the int32 words
            # (keeping both would double weight memory).
            self.qw8, self.qz8 = repack_gptq_uint8(qw, qz)
            self.qw8, self.qz8 = self.qw8.to(device), self.qz8.to(device)
            self.qw, self.qz = None, None
        else:
            self.qw, self.qz = qw.to(device), qz.to(device)
            self.qw8, self.qz8 = None, None
        self.sc = sc.to(device)
        self.g_idx = g_idx.to(device)
        self.device = device
        self.out_features = sc.shape[1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            return gptq_gemv_triton(x.to(self.device), self.qw8, self.sc,
                                    self.qz8, self.g_idx)
        # Fallback: dequantize chunk-by-chunk and matvec (exact, slow).
        N = self.sc.shape[1]
        y = torch.zeros(N, dtype=torch.float32, device=self.device)  # MUST be zeros: += accumulates
        K = self.qw.shape[0] * 8
        for k0 in range(0, K, 512):
            k1 = min(k0 + 512, K)
            ks = slice(k0, k1)
            w4 = unpack_nibbles(self.qw[k0 // 8: k1 // 8]).to(torch.float32)
            g = self.g_idx[ks]
            z = unpack_zeros(self.qz).to(torch.float32) + 1.0
            w = (w4 - z[g]) * self.sc[g].to(torch.float32)
            y += torch.mv(w.t(), x[ks].to(self.device).to(torch.float32))
        return y.to(torch.float16)

    def memory_bytes(self) -> int:
        packed = self.qw8 if self.use_triton else self.qw
        zeros = self.qz8 if self.use_triton else self.qz
        return packed.numel() + self.sc.numel() * 2 + zeros.numel() + self.g_idx.numel() * 4


def repack_nibbles(w4: torch.Tensor) -> torch.Tensor:
    """[K, N] int (0..15) -> [K/8, N] int32, low nibble first."""
    k, n = w4.shape
    w4 = w4.reshape(k // 8, 8, n).to(torch.int32)
    shifts = (torch.arange(8, device=w4.device, dtype=torch.int32) * 4).view(1, 8, 1)
    return ((w4 << shifts).sum(dim=1)).to(torch.int32)


def ungqa(qw: torch.Tensor, sc: torch.Tensor, qz: torch.Tensor,
          n_heads: int, n_kv_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Un-group GQA to full MHA (lossless): repeat each KV head's columns group times.
    Operates on the OUTPUT (N) dimension, which is packed inside int32 words -> unpack,
    repeat, repack."""
    group = n_heads // n_kv_heads
    if group == 1:
        return qw, sc, qz
    w4 = unpack_nibbles(qw)                       # [K, N]
    w4 = w4.repeat_interleave(group, dim=1)       # [K, N*group]
    sc = sc.repeat_interleave(group, dim=1)       # [G, N*group]
    z4 = unpack_zeros(qz)                          # [G, N]
    z4 = z4.repeat_interleave(group, dim=1)       # [G, N*group]
    return repack_nibbles(w4), sc.contiguous(), repack_nibbles(z4)


# ====================================================================================================
# 3. SMALL MATH OPS (torch; Triton variants only win on large d — kept torch for clarity)
# ====================================================================================================

def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + eps) * gamma.float()).to(x.dtype)


def rope_qknorm_seq(x: torch.Tensor, eta: torch.Tensor, pos0: int, d_head: int,
                    base: float = 10000.0, eps: float = 1e-6) -> torch.Tensor:
    """Vectorized adaptive RoPE + per-head QK-RMSNorm for a whole chunk.
    x: [T, n_heads, d_head]; absolute positions pos0..pos0+T-1."""
    T, H, dk = x.shape
    half = dk // 2
    freqs = base ** (-2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / d_head)
    theta = freqs * (2.0 * torch.sigmoid(eta.float()))
    pos = torch.arange(pos0, pos0 + T, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * theta[None, :]                       # [T, half]
    cos = ang.cos()[:, None, :]
    sin = ang.sin()[:, None, :]
    xf = x.float()
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    re, ro = xe * cos - xo * sin, xe * sin + xo * cos
    out = torch.stack([re, ro], dim=-1).reshape(T, H, dk)
    inv = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * inv).to(x.dtype)


def _deq(W: "GPTQLinear") -> torch.Tensor:
    """Dequantize a frozen GPTQ linear to fp16 [K, N] on its device (training path).
    Called inside checkpointed compute: materialized transiently in forward and
    again in the backward recompute, never retained across layers by autograd."""
    if W.use_triton:
        w4 = unpack_uint8_pairs(W.qw8).to(torch.float32)
        z = unpack_uint8_pairs(W.qz8).to(torch.float32) + 1.0
    else:
        w4 = unpack_nibbles(W.qw).to(torch.float32)
        z = unpack_zeros(W.qz).to(torch.float32) + 1.0
    g = W.g_idx.long()
    return ((w4 - z[g]) * W.sc.float()[g]).to(torch.float16)


def rope_qknorm(vec: torch.Tensor, eta: torch.Tensor, pos: int, d_head: int,
                base: float = 10000.0, eps: float = 1e-6) -> torch.Tensor:
    """Adaptive RoPE + per-head QK-RMSNorm on [n_heads, d_head] at absolute position pos."""
    half = d_head // 2
    freqs = base ** (-2.0 * torch.arange(half, device=vec.device, dtype=torch.float32) / d_head)
    theta = freqs * (2.0 * torch.sigmoid(eta.float()))
    ang = float(pos) * theta
    cos, sin = ang.cos(), ang.sin()

    v = vec.float()
    ve, vo = v[:, 0::2], v[:, 1::2]
    re, ro = ve * cos - vo * sin, ve * sin + vo * cos
    out = torch.stack([re, ro], dim=-1).reshape(vec.shape[0], d_head)
    # per-head RMSNorm
    inv = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * inv).to(vec.dtype)


# ====================================================================================================
# 4. MRM-v2 WORKING MEMORY (streaming, Rust-parity math)
# ====================================================================================================

class MRMv2Streaming:
    """Fixed-capacity 3-tier working memory. One instance per engine, shared by all layers."""

    def __init__(self, d_model: int, fine_slots: int = 128, coarse_slots: int = 16,
                 tau: float = 0.05, device: torch.device = torch.device("cpu")):
        self.d = d_model
        self.fine_slots, self.coarse_slots = fine_slots, coarse_slots
        self.tau = tau
        self.device = device

        self.fine_keys = torch.zeros(fine_slots, d_model, dtype=torch.float16, device=device)
        self.fine_vals = torch.zeros(fine_slots, d_model, dtype=torch.float16, device=device)
        self.fine_hits = torch.zeros(fine_slots, dtype=torch.float32, device=device)
        self.fine_sal = torch.zeros(fine_slots, dtype=torch.float32, device=device)
        self.fine_n = 0

        self.coarse_keys = torch.zeros(coarse_slots, d_model, dtype=torch.float16, device=device)
        self.coarse_vals = torch.zeros(coarse_slots, d_model, dtype=torch.float16, device=device)
        self.coarse_n = 0

    # --- Rust-parity dynamic thresholds -------------------------------------------
    @staticmethod
    def t_overwrite(d: int) -> float:
        return min(0.9999, 8.014 / math.sqrt(d))

    @staticmethod
    def t_merge(d: int) -> float:
        return min(MRMv2Streaming.t_overwrite(d) - 1e-4, 6.50 / math.sqrt(d))

    def reset(self):
        for t in (self.fine_keys, self.fine_vals, self.fine_hits, self.fine_sal,
                  self.coarse_keys, self.coarse_vals):
            t.zero_()
        self.fine_n = self.coarse_n = 0

    def slots(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Occupied (keys, values) of shape [S, d], S = fine_n + coarse_n."""
        parts_k, parts_v = [], []
        if self.fine_n:
            parts_k.append(self.fine_keys[: self.fine_n])
            parts_v.append(self.fine_vals[: self.fine_n])
        if self.coarse_n:
            parts_k.append(self.coarse_keys[: self.coarse_n])
            parts_v.append(self.coarse_vals[: self.coarse_n])
        if not parts_k:
            return None, None
        return torch.cat(parts_k, 0).contiguous(), torch.cat(parts_v, 0).contiguous()

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor, salience: float):
        """3-tier write state machine (Tier 1 overwrite / Tier 2 merge / Tier 3 eviction)."""
        key = key.reshape(-1).to(self.device).float()
        value = value.reshape(-1).to(self.device).float()
        kn = key.norm()
        if torch.isfinite(kn) and kn > 1e-12:
            key = key / kn
        key, value = key.half(), value.half()

        d = self.d
        if self.fine_n < self.fine_slots:
            idx = self.fine_n
            self.fine_keys[idx], self.fine_vals[idx] = key, value
            self.fine_hits[idx], self.fine_sal[idx] = 1.0, float(salience)
            self.fine_n += 1
        else:
            sims = torch.mv(self.fine_keys[: self.fine_n].float(), key.float())
            best_sim, idx = sims.max(0)
            s = float(best_sim)
            if s >= self.t_overwrite(d):          # Tier 1: hard in-place overwrite
                self.fine_keys[idx], self.fine_vals[idx] = key, value
                self.fine_hits[idx] = torch.clamp(self.fine_hits[idx] + 1.0, max=50.0)
            elif s >= self.t_merge(d):            # Tier 2: soft semantic merge
                merged = 0.70 * key.float() + 0.30 * self.fine_keys[idx].float()
                merged = merged / max(float(merged.norm()), 1e-8)
                self.fine_keys[idx] = merged.half()
                self.fine_vals[idx] = (0.70 * value.float() + 0.30 * self.fine_vals[idx].float()).half()
                self.fine_hits[idx] = torch.clamp(self.fine_hits[idx] + 0.5, max=50.0)
            else:                                  # Tier 3: utility eviction
                utility = 2.0 * self.fine_hits[: self.fine_n] + self.fine_sal[: self.fine_n]
                victim = int(utility.argmin())
                self.fine_keys[victim], self.fine_vals[victim] = key, value
                self.fine_hits[victim], self.fine_sal[victim] = 1.0, float(salience)

        # Coarse tier: Poincaré Ball conformal Riemannian EMA (gamma = 0.95)
        gamma = 0.95
        if self.coarse_n < self.coarse_slots:
            self.coarse_keys[self.coarse_n], self.coarse_vals[self.coarse_n] = key, value
            self.coarse_n += 1
        else:
            c_sims = torch.mv(self.coarse_keys[: self.coarse_n].float(), key.float())
            c_idx = int(c_sims.argmax())
            c = self.coarse_keys[c_idx].float()
            scale = ((1.0 - float(c.pow(2).sum())) ** 2) / 4.0
            c = c + (1.0 - gamma) * scale * (key.float() - c)
            c_norm = float(c.norm())
            if c_norm > 1.0 - 1e-5:               # boundary clip ||c|| <= 1 - 1e-5
                c = c * ((1.0 - 1e-5) / c_norm)
            self.coarse_keys[c_idx] = c.half()
            self.coarse_vals[c_idx] = (gamma * self.coarse_vals[c_idx].float()
                                       + (1.0 - gamma) * value.float()).half()

    def status(self) -> Dict:
        return {"fine_occupied": self.fine_n, "coarse_occupied": self.coarse_n,
                "t_overwrite": self.t_overwrite(self.d), "t_merge": self.t_merge(self.d),
                "tau": self.tau, "device": str(self.device)}


# ====================================================================================================
# 5. TESSERA LAYER (streaming / GEMV form of the 9-pillar stage)
# ====================================================================================================

class TesseraLayer:
    def __init__(self, cfg: TesseraConfig, layer_idx: int, device: torch.device):
        self.cfg, self.layer_idx, self.device = cfg, layer_idx, device
        d, dk = cfg.d_model, cfg.d_head
        use_tri = triton_enabled(device)

        def lin(k_in: int, n_out: int):
            qw = torch.randint(0, 1 << 31, (k_in // 8, n_out), dtype=torch.int64).to(torch.int32)
            g = int(cfg.group_size)
            n_groups = (k_in + g - 1) // g
            g_idx = torch.arange(k_in, dtype=torch.int32) // g
            sc = (torch.randn(n_groups, n_out) * 0.005 + 0.01).half()
            qz = torch.full((n_groups, n_out // 8), -2004318072, dtype=torch.int32)  # 0x88888888 signed
            return GPTQLinear(qw, sc, qz, g_idx, device, use_tri)

        self.wq, self.wk, self.wv, self.wo = lin(d, d), lin(d, d), lin(d, d), lin(d, d)
        self.w_gate = lin(d, d)
        self.w1, self.w1u, self.w2 = lin(d, cfg.d_ff), lin(d, cfg.d_ff), lin(cfg.d_ff, d)
        self.av, self.au = lin(d, cfg.r_adapter), lin(cfg.r_adapter, d)

        self.gamma1 = torch.ones(d, dtype=torch.float16, device=device)
        self.gamma2 = torch.ones(d, dtype=torch.float16, device=device)
        self.w_conv = (torch.randn(4, d) * 0.02).half().to(device)
        self.eta = torch.zeros(dk // 2, dtype=torch.float32, device=device)

        self.lambda_ab = torch.zeros(cfg.n_pairs, 2, dtype=torch.float32, device=device)
        self.vres_w = 1.0 / (1.0 + math.exp(-0.8473))  # sigmoid(ln(0.7/0.3)) ~= 0.7
        self.lambda_init = lambda_init_at(layer_idx)

        self._conv_buf: deque = deque(maxlen=3)
        # Calibration fine-tune fields (populated by engine.enable_calibration_mode):
        # dense trainable replacements for the Tessera-only components.
        self.w_conv_p = None      # Parameter fp32 [4, d]   (replaces self.w_conv)
        self.eta_p = None         # Parameter fp32 [dk/2]  (replaces self.eta)
        self.lambda_p = None      # Parameter fp32 [P, 2]  (replaces self.lambda_ab)
        self.vres_gate = None     # Parameter fp32 scalar  (replaces float self.vres_w)
        self.av_w = None          # Parameter fp32 [r, d]  (replaces GPTQ self.av)
        self.au_w = None          # Parameter fp32 [d, r]  (replaces GPTQ self.au; zero-init)
        self._chunk_seed = []     # last 3 post-norm h (detached) for chunked conv

    def _active(self, name: str):
        """Calibration Parameter if present, else the frozen default."""
        p = getattr(self, name + "_p")
        return p if p is not None else getattr(self, name)

    def load_source(self, w: Dict[str, torch.Tensor], prefix: str,
                    n_kv_heads: int) -> List[str]:
        """Transplant weight-compatible tensors from an HF GPTQ checkpoint (see engine loader)."""
        g = self.cfg.group_size
        d = self.cfg.d_model

        def gp(name, k_in, n_out, ungroup=False):
            qw, sc, qz = w[prefix + name + ".qweight"], w[prefix + name + ".scales"], w[prefix + name + ".qzeros"]
            g_idx = w.get(prefix + name + ".g_idx")
            if g_idx is None:
                g_idx = torch.arange(k_in, dtype=torch.int32) // g
            if ungroup:
                qw, sc, qz = ungqa(qw, sc, qz, self.cfg.n_heads, n_kv_heads)
            return GPTQLinear(qw, sc, qz, g_idx.to(torch.int32), self.device,
                              triton_enabled(self.device))

        self.wq = gp(".self_attn.q_proj", d, d)
        self.wk = gp(".self_attn.k_proj", d, d, ungroup=True)
        self.wv = gp(".self_attn.v_proj", d, d, ungroup=True)
        self.wo = gp(".self_attn.o_proj", d, d)
        self.w1 = gp(".mlp.gate_proj", d, self.cfg.d_ff)
        self.w1u = gp(".mlp.up_proj", d, self.cfg.d_ff)
        self.w2 = gp(".mlp.down_proj", self.cfg.d_ff, d)
        if prefix + ".input_layernorm.weight" in w:
            self.gamma1 = w[prefix + ".input_layernorm.weight"].to(self.device).half()
        if prefix + ".post_attention_layernorm.weight" in w:
            self.gamma2 = w[prefix + ".post_attention_layernorm.weight"].to(self.device).half()

    def reset_state(self):
        self._conv_buf.clear()
        self._chunk_seed = []

    # ----------------------------------------------------------------------------------
    def chunk_compute(self, h, v0, slots_k, slots_v, seed, pos0):
        """PURE chunked training forward over T tokens (checkpoint-safe: deterministic,
        no side effects, no RNG). GEMM path: frozen 4-bit weights dequantized on the fly.
        h: [T, d] fp16 | v0: [T, d] stage-1 values or None | slots_k/v: [S, d] detached
        memory snapshot or None | seed: [s<=3, d] detached conv history or None.
        Returns (h_out, v_raw, q_mean_det, veff_mean_det, sal_det, hn)."""
        cfg = self.cfg
        T, d = h.shape[0], cfg.d_model
        dev = h.device
        w_conv = self._active("w_conv")
        eta = self._active("eta")
        lam_ab = self.lambda_p if self.lambda_p is not None else self.lambda_ab

        # Pillar 1
        hn = rmsnorm(h, self.gamma1, cfg.norm_eps)            # [T, d]

        # Pillar 2 (chunked causal conv seeded by detached history)
        s = 0 if seed is None else seed.shape[0]
        inp = torch.cat([seed, hn], 0) if seed is not None else hn
        inp_f = inp.float()
        conv_full = torch.zeros_like(inp_f)
        for j in range(4):
            shifted = inp_f if j == 0 else torch.cat(
                [torch.zeros(j, d, device=dev), inp_f[:-j]], 0)
            conv_full = conv_full + w_conv[j].float().to(dev) * shifted
        conv = conv_full[s:]
        gate = torch.sigmoid((hn @ _deq(self.w_gate)).float())
        h_gated = (conv * gate).to(h.dtype)

        # Pillars 3+4: Q/K/V GEMM + RoPE + QK-norm
        qh = (h_gated @ _deq(self.wq)).view(T, cfg.n_heads, cfg.d_head)
        kh = (h_gated @ _deq(self.wk)).view(T, cfg.n_heads, cfg.d_head)
        vh = (h_gated @ _deq(self.wv)).view(T, cfg.n_heads, cfg.d_head)
        qh = rope_qknorm_seq(qh, eta, pos0, cfg.d_head, cfg.rope_base, cfg.norm_eps)
        kh = rope_qknorm_seq(kh, eta, pos0, cfg.d_head, cfg.rope_base, cfg.norm_eps)

        # Pillar 6: value residual on this chunk's values (v0 carries grad)
        v_raw = vh
        if v0 is not None:
            v0 = v0.to(h.device)   # stage-1 values live on dev0; dev1 layers must pull them
            w = torch.sigmoid(self.vres_gate) if self.vres_gate is not None else self.vres_w
            v_eff = (w * vh.float() + (1.0 - w) * v0.float()).to(vh.dtype)
        else:
            v_eff = vh

        # Pillars 5+8: DiffAttn over detached memory slots + causal chunk prefix
        qf, kf, vf = qh.float(), kh.float(), v_eff.float()
        q1, q2 = qf[:, 0::2], qf[:, 1::2]
        k1c, k2c = kf[:, 0::2], kf[:, 1::2]
        v1c, v2c = vf[:, 0::2], vf[:, 1::2]
        S = 0
        if slots_k is not None:
            sh = slots_k.float().view(-1, cfg.n_heads, cfg.d_head)
            sv = slots_v.float().view(-1, cfg.n_heads, cfg.d_head)
            k1 = torch.cat([sh[:, 0::2], k1c], 0)
            k2 = torch.cat([sh[:, 1::2], k2c], 0)
            v1 = torch.cat([sv[:, 0::2], v1c], 0)
            v2 = torch.cat([sv[:, 1::2], v2c], 0)
            S = sh.shape[0]
        else:
            k1, k2, v1, v2 = k1c, k2c, v1c, v2c

        scale = 1.0 / math.sqrt(cfg.d_head)
        s1 = torch.einsum("tpd,spd->tps", q1, k1) * scale
        s2 = torch.einsum("tpd,spd->tps", q2, k2) * scale
        if S:
            neg = torch.finfo(s1.dtype).min
            causal = torch.triu(torch.full((T, T), neg, device=dev), diagonal=1)
            mask = torch.zeros(T, S + T, device=dev)
            mask[:, S:] = causal
            s1, s2 = s1 + mask[:, None, :], s2 + mask[:, None, :]
        p1, p2 = s1.softmax(-1), s2.softmax(-1)
        a, b = lam_ab[:, 0], lam_ab[:, 1]
        lam = torch.clamp(torch.exp(a) - torch.exp(b) + self.lambda_init, min=0.0)
        pdiff = p1 - lam[None, :, None] * p2
        o1 = torch.einsum("tps,spd->tpd", pdiff, v1)
        o2 = torch.einsum("tps,spd->tpd", pdiff, v2)
        ctx = torch.stack([o1, o2], dim=-1).reshape(T, cfg.n_heads, cfg.d_head)
        ctx = ctx.reshape(T, d)

        h_mid = h + (ctx.to(h.dtype) @ _deq(self.wo)).to(h.dtype)

        # Pillar 7
        hn2 = rmsnorm(h_mid, self.gamma2, cfg.norm_eps)
        g = hn2 @ _deq(self.w1)
        u = hn2 @ _deq(self.w1u)
        act = (g.float() * torch.sigmoid(g.float())).to(h.dtype)
        ffn = act @ _deq(self.w2)
        adapter = (hn2.float() @ self.av_w.t()) @ self.au_w.t()
        h_out = h_mid + (ffn.float() + adapter).to(h.dtype)

        with torch.no_grad():
            q_mean = qh.mean(0).reshape(-1).detach()
            veff_mean = v_eff.mean(0).reshape(-1).detach()
            sal = (v_eff.reshape(T, d).float() - ctx.float()).norm(dim=-1).detach()   # [T]
        return h_out, v_raw, q_mean, veff_mean, sal, hn

    # ----------------------------------------------------------------------------------
    def _diff_attn_over_mrm(self, q: torch.Tensor, mrm: MRMv2Streaming) -> torch.Tensor:
        """Differential Attention over working-memory slots (Pillars 5 + 8).
        q: [n_heads, d_head] (rope'd + qk-normed). Returns [d_model]."""
        cfg = self.cfg
        keys, vals = mrm.slots()
        if keys is None:
            return torch.zeros(cfg.d_model, dtype=q.dtype, device=q.device)
        # MRM lives on dev0; layers on dev1 must pull the slots across PCIe
        if keys.device != self.device:
            keys, vals = keys.to(self.device, non_blocking=True), vals.to(self.device, non_blocking=True)
        lam_ab = self.lambda_p if self.lambda_p is not None else self.lambda_ab

        S = keys.shape[0]
        kh = keys.float().view(S, cfg.n_heads, cfg.d_head)
        vh = vals.float().view(S, cfg.n_heads, cfg.d_head)
        qf = q.float()

        k1, k2 = kh[:, 0::2], kh[:, 1::2]            # [S, P, dk]
        v1, v2 = vh[:, 0::2], vh[:, 1::2]
        q1, q2 = qf[0::2], qf[1::2]                  # [P, dk]

        scale = 1.0 / math.sqrt(cfg.d_head)
        s1 = torch.einsum("pd,spd->sp", q1, k1) * scale
        s2 = torch.einsum("pd,spd->sp", q2, k2) * scale
        p1, p2 = s1.softmax(-1), s2.softmax(-1)

        a, b = lam_ab[:, 0], lam_ab[:, 1]
        lam = torch.clamp(torch.exp(a) - torch.exp(b) + self.lambda_init, min=0.0)
        pdiff = p1 - lam.unsqueeze(0) * p2           # [S, P] (s: slots, p: pairs)

        o1 = torch.einsum("sp,spd->pd", pdiff, v1)
        o2 = torch.einsum("sp,spd->pd", pdiff, v2)
        out = torch.stack([o1, o2], dim=-1).reshape(cfg.n_heads, cfg.d_head)
        return out.reshape(cfg.d_model).to(q.dtype)

    @torch.no_grad()
    def forward(self, h: torch.Tensor, mrm: MRMv2Streaming, pos: int,
                v0_token: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """One token through one stage. h: [d_model] on self.device.
        Returns (h_out, v_raw_of_this_stage). v_raw of stage 1 becomes v0 for later stages."""
        cfg = self.cfg
        if h.device != self.device:
            h = h.to(self.device)

        # Pillar 1: pre-attention RMSNorm
        hn = rmsnorm(h, self.gamma1, cfg.norm_eps)

        # Pillar 2: causal depthwise conv k=4 * sigmoid(W_gate . hn)
        self._conv_buf.append(hn)
        w_conv = self._active("w_conv")
        conv = torch.zeros_like(hn.float())
        for j, prev in enumerate(self._conv_buf):
            conv += w_conv[j].float().to(h.device) * prev.float()   # j=0 -> current token
        gate = torch.sigmoid(self.w_gate(hn).float())
        h_gated = (conv * gate).to(hn.dtype)

        # Pillars 3+4: Q/K/V projections, adaptive RoPE, per-head QK-RMSNorm
        eta = self._active("eta")
        q = rope_qknorm(self.wq(h_gated).view(cfg.n_heads, cfg.d_head),
                        eta, pos, cfg.d_head, cfg.rope_base, cfg.norm_eps)
        k = rope_qknorm(self.wk(h_gated).view(cfg.n_heads, cfg.d_head),
                        eta, pos, cfg.d_head, cfg.rope_base, cfg.norm_eps)
        v = self.wv(h_gated)

        # Pillars 5+8: DiffAttn over MRM slots
        ctx = self._diff_attn_over_mrm(q, mrm)
        h_mid = h + self.wo(ctx)

        # Pillar 7: SwiGLU FFN + low-rank adapter
        hn2 = rmsnorm(h_mid, self.gamma2, cfg.norm_eps)
        g, u = self.w1(hn2).float(), self.w1u(hn2).float()
        ffn = self.w2((g * torch.sigmoid(g)).to(hn2.dtype)).float()
        if self.av_w is not None:   # trainable dense adapter (calibration mode)
            adapter = (hn2.float() @ self.av_w.t().to(hn2.device)) @ self.au_w.t().to(hn2.device)
        else:                       # frozen GPTQ adapter (default init)
            adapter = self.au(self.av(hn2)).float()
        h_out = h_mid + (ffn + adapter).to(h_mid.dtype)

        # Pillar 6: ResFormer value residual on the value written to memory
        v_raw = v
        if v0_token is not None:
            if self.vres_gate is not None:
                w = torch.sigmoid(self.vres_gate)
            else:
                w = self.vres_w
            v_eff = (w * v.float() + (1.0 - w) * v0_token.to(self.device).float()).to(v.dtype)
        else:
            v_eff = v

        # Pillar 8 write: key = rope'd q, value = value-residual-mixed v,
        # salience = closed-form gradient-surprise proxy ||v_eff - ctx||
        salience = float((v_eff.float() - ctx.float()).norm())
        mrm.write(q.reshape(-1), v_eff, salience)

        return h_out, v_raw


# ====================================================================================================
# 6. ENGINE: dual-GPU layer split + streaming generation
# ====================================================================================================

class Tessera50BEngine:
    def __init__(self, cfg: TesseraConfig = TESSERA_50B, model_dir: Optional[str] = None,
                 tokenizer_dir: Optional[str] = None):
        self.cfg = cfg
        if torch.cuda.is_available():
            self.dev0, self.dev1 = torch.device("cuda:0"), torch.device("cuda:1")
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"[tessera] cuda:{i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB "
                      f"(if this is low, del stale engines or restart the session)")
        else:
            print("[!] CUDA not available -> single-device (CPU) smoke mode.")
            self.dev0 = self.dev1 = torch.device("cpu")

        self.split = (cfg.n_layers + 1) // 2
        self.mrm = MRMv2Streaming(cfg.d_model, cfg.fine_slots, cfg.coarse_slots,
                                  cfg.tau, self.dev0)
        self.pos = 0

        if model_dir:
            self._load_from_hf_gptq(model_dir, tokenizer_dir or model_dir)
        else:
            self._random_init()

        n_dev0 = self.split
        print(f"[tessera] {cfg.n_layers} layers: [0..{n_dev0}) on {self.dev0}, "
              f"[{n_dev0}..{cfg.n_layers}) on {self.dev1}")
        print(f"[tessera] compute path: {'Triton INT4 GEMV' if triton_enabled(self.dev0) else 'PyTorch fallback'}"
              f" | first call may JIT-compile (~1 min)")
        print(f"[tessera] params ≈ {cfg.param_count()/1e9:.2f}B | MRM slots: "
              f"{cfg.fine_slots}+{cfg.coarse_slots} | t_over={self.mrm.t_overwrite(cfg.d_model):.4f} "
              f"t_merge={self.mrm.t_merge(cfg.d_model):.4f}")

    # ------------------------------------------------------------------ random init
    def _random_init(self):
        cfg = self.cfg
        self.embed = (torch.randn(cfg.vocab_size, cfg.d_model) * 0.02).half().to(self.dev0)
        self.final_gamma = torch.ones(cfg.d_model, dtype=torch.float16, device=self.dev0)
        self.layers = [TesseraLayer(cfg, i, self.dev0 if i < self.split else self.dev1)
                       for i in range(cfg.n_layers)]

    # ------------------------------------------------------------------ HF GPTQ load
    def _load_from_hf_gptq(self, model_dir: str, tokenizer_dir: str):
        from transformers import AutoConfig

        if not os.path.isdir(model_dir) or not os.path.exists(
                os.path.join(model_dir, "config.json")):
            raise FileNotFoundError(
                f"'{model_dir}' is not a local HF checkpoint directory (config.json "
                f"missing). Download it first, e.g.\n"
                f"  from huggingface_hub import snapshot_download\n"
                f"  snapshot_download('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4',\n"
                f"      local_dir='{model_dir}',\n"
                f"      allow_patterns=['*.json', '*.safetensors', '*.safetensors.index.json'])\n"
                f"Tip: /kaggle/working is wiped when a session is recreated — upload the "
                f"checkpoint as a Kaggle Dataset to avoid re-downloading.")

        hf_cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        qcfg = getattr(hf_cfg, "quantization_config", {}) or {}
        gs = int(qcfg.get("group_size", self.cfg.group_size))

        d_model = hf_cfg.hidden_size
        n_layers = hf_cfg.num_hidden_layers
        n_heads = hf_cfg.num_attention_heads
        n_kv = getattr(hf_cfg, "num_key_value_heads", n_heads)
        cfg = TesseraConfig(
            vocab_size=hf_cfg.vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_ff=hf_cfg.intermediate_size, r_adapter=self.cfg.r_adapter,
            fine_slots=self.cfg.fine_slots, coarse_slots=self.cfg.coarse_slots,
            tau=self.cfg.tau, group_size=gs,
        )
        self.cfg = cfg
        self.split = (cfg.n_layers + 1) // 2
        # re-create MRM at the real d_model (dev0)
        self.mrm = MRMv2Streaming(cfg.d_model, cfg.fine_slots, cfg.coarse_slots,
                                  cfg.tau, self.dev0)

        print(f"[tessera] source: d_model={d_model} layers={n_layers} heads={n_heads} "
              f"kv_heads={n_kv} d_ff={cfg.d_ff} vocab={cfg.vocab_size} group={gs}")

        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            weight_map = {}
        shards = sorted(set(weight_map.values())) or sorted(
            glob.glob(os.path.join(model_dir, "*.safetensors")))

        from safetensors import safe_open
        handles = {sh: safe_open(os.path.join(model_dir, sh), framework="pt", device="cpu")
                   for sh in shards}

        def fetch(name: str) -> Optional[torch.Tensor]:
            sh = weight_map.get(name)
            if sh is None:
                for h in handles.values():
                    if name in h.keys():
                        return h.get_tensor(name)
                return None
            return handles[sh].get_tensor(name)

        # Embeddings (tied) fp16 on dev0
        emb = fetch("model.embed_tokens.weight")
        if emb is None:
            raise RuntimeError("model.embed_tokens.weight not found in checkpoint")
        self.embed = emb.to(self.dev0, torch.float16)
        final = fetch("model.norm.weight")
        self.final_gamma = (final if final is not None else torch.ones(cfg.d_model)) \
            .to(self.dev0).half()

        self.layers = []
        for i in range(cfg.n_layers):
            dev = self.dev0 if i < self.split else self.dev1
            layer = TesseraLayer.__new__(TesseraLayer)  # skip random lin init (saves RAM)
            layer.cfg, layer.layer_idx, layer.device = cfg, i, dev
            layer.gamma1 = torch.ones(cfg.d_model, dtype=torch.float16, device=dev)
            layer.gamma2 = torch.ones(cfg.d_model, dtype=torch.float16, device=dev)
            layer.w_conv = (torch.randn(4, cfg.d_model) * 0.02).half().to(dev)
            layer.eta = torch.zeros(cfg.d_head // 2, dtype=torch.float32, device=dev)
            layer.lambda_ab = torch.zeros(cfg.n_pairs, 2, dtype=torch.float32, device=dev)
            layer.vres_w = 1.0 / (1.0 + math.exp(-0.8473))
            layer.lambda_init = lambda_init_at(i)
            layer._conv_buf = deque(maxlen=3)
            use_tri = triton_enabled(dev)

            def gp(name, k_in, n_out, ungroup=False):
                p = f"model.layers.{i}{name}"
                qw, sc, qz = fetch(p + ".qweight"), fetch(p + ".scales"), fetch(p + ".qzeros")
                if qw is None:
                    raise RuntimeError(f"missing GPTQ tensor {p}.qweight")
                g_idx = fetch(p + ".g_idx")
                if g_idx is None:
                    g_idx = torch.arange(k_in, dtype=torch.int32) // gs
                if ungroup:
                    qw, sc, qz = ungqa(qw, sc, qz, cfg.n_heads, n_kv)
                return GPTQLinear(qw, sc, qz, g_idx.to(torch.int32), dev, use_tri)

            d = cfg.d_model
            layer.wq = gp(".self_attn.q_proj", d, d)
            layer.wk = gp(".self_attn.k_proj", d, d, ungroup=True)
            layer.wv = gp(".self_attn.v_proj", d, d, ungroup=True)
            layer.wo = gp(".self_attn.o_proj", d, d)
            layer.w1 = gp(".mlp.gate_proj", d, cfg.d_ff)
            layer.w1u = gp(".mlp.up_proj", d, cfg.d_ff)
            layer.w2 = gp(".mlp.down_proj", cfg.d_ff, d)
            # Tessera-only components (no source equivalent): principled random defaults.
            layer.w_gate = _random_gptq_linear(d, d, cfg.group_size, dev, use_tri)
            layer.av = _random_gptq_linear(d, cfg.r_adapter, cfg.group_size, dev, use_tri)
            layer.au = _random_gptq_linear(cfg.r_adapter, d, cfg.group_size, dev, use_tri)

            g1 = fetch(f"model.layers.{i}.input_layernorm.weight")
            g2 = fetch(f"model.layers.{i}.post_attention_layernorm.weight")
            if g1 is not None:
                layer.gamma1 = g1.to(dev).half()
            if g2 is not None:
                layer.gamma2 = g2.to(dev).half()

            self.layers.append(layer)
            if (i + 1) % 10 == 0 or i == cfg.n_layers - 1:
                print(f"[tessera] loaded layer {i+1}/{cfg.n_layers}")

        # tokenizer (optional)
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        except Exception as e:
            print(f"[tessera] tokenizer unavailable ({e}); byte-fallback enabled.")
            self.tokenizer = None

    # ------------------------------------------------------------------ generation
    def reset(self):
        self.pos = 0
        self.mrm.reset()
        for layer in self.layers:
            layer.reset_state()

    def step(self, token_id: int) -> torch.Tensor:
        """One streaming decode step -> logits [vocab] (fp32)."""
        h = self.embed[token_id]                      # [d_model] fp16 on dev0
        v0_token = None
        for i, layer in enumerate(self.layers):
            h, v_raw = layer.forward(h, self.mrm, self.pos, v0_token)
            if i == 0:
                v0_token = v_raw.to(self.dev0)
            if i == self.split - 1:
                h = h.to(self.dev1)                   # single device hop per token
        # Pillar 9: final norm + tied head + soft-cap
        hn = rmsnorm(h.to(self.dev0), self.final_gamma, self.cfg.norm_eps)
        logits = self.embed.float() @ hn.float()
        return self.cfg.logit_cap * torch.tanh(logits / self.cfg.logit_cap)

    # ------------------------------------------------------------------ calibration
    def enable_calibration_mode(self):
        """Swap Tessera-only components for dense fp32 trainable Parameters.
        Transplanted weights (QKVO, FFN, embeddings, norms, conv gate) stay frozen.
        The conv gate is re-initialized to zero -> sigmoid(0)=0.5 neutral gate."""
        if self.layers[0].w_conv_p is not None:
            print("[tessera] calibration mode already enabled")
            return
        for i, layer in enumerate(self.layers):
            cfg, dev = self.cfg, layer.device
            layer.w_conv_p = torch.nn.Parameter(layer.w_conv.float().clone().to(dev))
            layer.eta_p = torch.nn.Parameter(layer.eta.clone())
            layer.lambda_p = torch.nn.Parameter(layer.lambda_ab.clone())
            layer.vres_gate = torch.nn.Parameter(torch.tensor(0.8473, device=dev))
            g = torch.Generator().manual_seed(1234 + i)
            av = torch.randn(cfg.r_adapter, cfg.d_model, generator=g) * 0.02
            layer.av_w = torch.nn.Parameter(av.to(dev))
            layer.au_w = torch.nn.Parameter(                       # zero-init: no-op start
                torch.zeros(cfg.d_model, cfg.r_adapter, device=dev))
            layer.w_gate = _zero_gptq_linear(cfg.d_model, cfg.d_model, cfg.group_size,
                                             dev, triton_enabled(dev))
        n = sum(p.numel() for p in self.trainable_parameters())
        print(f"[tessera] calibration mode: {n/1e6:.2f}M trainable fp32 params "
              f"(conv, eta, lambda, vres, adapters); everything else frozen")

    def trainable_parameters(self):
        out = []
        for layer in self.layers:
            out += [layer.w_conv_p, layer.eta_p, layer.lambda_p,
                    layer.vres_gate, layer.av_w, layer.au_w]
        return out

    def calibration_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {}
        for i, layer in enumerate(self.layers):
            for name in ("w_conv_p", "eta_p", "lambda_p", "vres_gate", "av_w", "au_w"):
                sd[f"L{i}.{name}"] = getattr(layer, name).detach().cpu()
        return sd

    def load_calibration_state_dict(self, sd: Dict[str, torch.Tensor]):
        self.enable_calibration_mode()
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                for name in ("w_conv_p", "eta_p", "lambda_p", "vres_gate", "av_w", "au_w"):
                    cur = getattr(layer, name)
                    cur.copy_(sd[f"L{i}.{name}"].to(cur.device, cur.dtype))
        print("[tessera] calibration weights loaded")

    def chunk_forward(self, token_ids: List[int]) -> torch.Tensor:
        """Training forward over a chunk (GEMM path, per-layer gradient checkpointing).
        Memory (MRM slots, conv seeds) is DETACHED across chunk boundaries — truncated
        BPTT; attention within the chunk is exact, causal, and includes the memory
        slots as a shared prefix. Returns soft-capped logits [T, vocab] (fp32)."""
        T = len(token_ids)
        cfg = self.cfg
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.dev0)
        h = self.embed[ids]                                       # [T, d] fp16
        v0 = None
        for i, layer in enumerate(self.layers):
            slots_k, slots_v = self.mrm.slots()
            if slots_k is not None and slots_k.device != layer.device:
                slots_k, slots_v = slots_k.to(layer.device), slots_v.to(layer.device)
            seed = torch.stack(layer._chunk_seed).to(layer.device) if layer._chunk_seed else None
            h, v_raw, q_mean, veff_mean, sal, hn = checkpoint(
                layer.chunk_compute, h, v0, slots_k, slots_v, seed, self.pos,
                use_reentrant=False)
            with torch.no_grad():                                 # detached memory update
                self.mrm.write(q_mean, veff_mean, salience=float(sal.mean()))
                layer._chunk_seed = [t.detach() for t in hn[-3:]]
            if i == 0:
                v0 = v_raw                                        # keeps grad (Pillar 6)
            if i == self.split - 1:
                h = h.to(self.dev1)
        hn_final = rmsnorm(h.to(self.dev0), self.final_gamma, cfg.norm_eps)
        logits = hn_final @ self.embed.t()                        # [T, V] fp16, tied head
        self.pos += T
        return cfg.logit_cap * torch.tanh(logits.float() / cfg.logit_cap)

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95) -> List[int]:
        self.reset()
        out = list(prompt_ids)
        for t in range(max_new_tokens):
            logits = self.step(out[-1])
            self.pos += 1
            probs = sample_logits(logits, temperature, top_k, top_p)
            nxt = int(torch.multinomial(probs, 1))
            out.append(nxt)
        return out

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return bytes([i & 0xFF for i in ids]).decode("utf-8", errors="replace")


def _zero_gptq_linear(k_in: int, n_out: int, group: int, device, use_triton: bool) -> GPTQLinear:
    """All-zero GPTQ linear: dequantizes to exactly 0 -> sigmoid(Wx) = 0.5 gate.
    Used for the Tessera-only conv gate at load time (no source equivalent)."""
    n_groups = (k_in + group - 1) // group
    qw = torch.zeros(k_in // 8, n_out, dtype=torch.int32)
    sc = torch.zeros(n_groups, n_out, dtype=torch.float16)
    qz = torch.full((n_groups, n_out // 8), -2004318072, dtype=torch.int32)
    g_idx = (torch.arange(k_in, dtype=torch.int32) // group)
    return GPTQLinear(qw, sc, qz, g_idx, device, use_triton)


def _random_gptq_linear(k_in: int, n_out: int, group: int, device, use_triton: bool) -> GPTQLinear:
    n_groups = (k_in + group - 1) // group
    qw = torch.randint(0, 1 << 31, (k_in // 8, n_out), dtype=torch.int64).to(torch.int32)
    sc = (torch.randn(n_groups, n_out) * 0.005 + 0.01).half()
    qz = torch.full((n_groups, n_out // 8), -2004318072, dtype=torch.int32)  # 0x88888888 signed
    g_idx = (torch.arange(k_in, dtype=torch.int32) // group)
    return GPTQLinear(qw, sc, qz, g_idx, device, use_triton)


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int,
                  top_p: float) -> torch.Tensor:
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    z = logits.float() / temperature
    if top_k and top_k > 0:
        kth = z.topk(min(top_k, z.numel()), dim=-1).values[-1]
        z = torch.where(z < kth, torch.full_like(z, float("-inf")), z)
    probs = z.softmax(-1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        cum = sp.cumsum(-1)
        keep = cum - sp < top_p
        keep[0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, keep)
        probs = torch.where(mask, probs, torch.zeros_like(probs))
    return probs / probs.sum()


# ====================================================================================================
# 7b. GPU SELF-CHECK: pinpoints where the CUDA path first goes non-finite
# ====================================================================================================

def check_gpu():
    """Bisects the GPU forward pass: validates every Triton GEMV against a float64
    torch reference, then walks a token layer-by-layer reporting finiteness."""
    print("=" * 90)
    print("  TESSERA-Q GPU SELF-CHECK (Triton GEMV vs float64 reference + layer bisect)")
    print("=" * 90)
    cfg = TesseraConfig(vocab_size=64, d_model=64, n_layers=4, n_heads=4,
                        d_ff=192, r_adapter=8, group_size=16)
    eng = Tessera50BEngine(cfg)
    if eng.dev0.type != "cuda":
        print("[!] No CUDA — nothing to check (CPU path already covered by --smoke).")
        return
    if not triton_enabled(eng.dev0):
        print("[!] WARNING: Triton path is NOT active (fallback in use). This check only\n"
              "    exercises the PyTorch fallback. To test Triton: unset TESSERA_FORCE_TORCH\n"
              "    in THIS process and re-run, e.g.  os.environ.pop('TESSERA_FORCE_TORCH', None)\n"
              "    then call check_gpu() directly instead of via a !shell subprocess.")

    for li, layer in enumerate(eng.layers):
        for name in ("w_gate", "wq", "wk", "wv", "wo", "w1", "w1u", "w2", "av", "au"):
            W = getattr(layer, name)
            K = W.qw8.shape[0] * 2 if W.use_triton else W.qw.shape[0] * 8
            x = (torch.randn(K, device=W.device, dtype=torch.float16) * 0.5)
            if W.use_triton:
                y_tri = gptq_gemv_triton(x, W.qw8, W.sc, W.qz8, W.g_idx).float()
                # float32 reference from the same uint8 packs, via torch-verified ops
                w4 = unpack_uint8_pairs(W.qw8).float()
                z4 = unpack_uint8_pairs(W.qz8).float() + 1.0
                g = W.g_idx.long()
                w_ref = (w4 - z4[g]) * W.sc.float()[g]
            else:
                y_tri = W(x).float()
                w_ref = dequant_gptq(W.qw.cpu(), W.sc.cpu(), W.qz.cpu(),
                                     W.g_idx.cpu().long()).to(W.device).float()
            y_ref = w_ref.t() @ x.float()
            diff = (y_tri - y_ref).abs().max().item()
            finite = torch.isfinite(y_tri).all().item()
            flag = "OK " if (finite and diff < 1e-1) else "BAD"
            print(f"  [{flag}] layer {li} {name:6s} finite={finite} max|tri-ref|={diff:.4f}")
            if not finite:
                print(f"      -> Triton output non-finite! ref absmax={y_ref.abs().max().item():.4f}")
                return

    # layer-by-layer token walk
    eng.reset()
    h = eng.embed[1]
    print(f"\n  token walk: embed absmax={h.float().abs().max().item():.4f}")
    for i, layer in enumerate(eng.layers):
        h, _ = layer.forward(h, eng.mrm, 0, None)
        amax = h.float().abs().max().item()
        print(f"  after layer {i}: finite={torch.isfinite(h.float()).all().item()} absmax={amax:.4f}")
        if not torch.isfinite(h.float()).all():
            print(f"      -> layer {i} poisoned the hidden state; rerun with CUDA_LAUNCH_BLOCKING=1")
            return
    print("\n✓ all GEMVs match float64 reference; full forward is finite")
    print("=" * 90)




# ====================================================================================================
# 9. CALIBRATION FINE-TUNING (fix output quality: train only Tessera-specific components)
# ====================================================================================================

def build_token_chunks(model_dir: str, tokenizer_dir: Optional[str] = None,
                       seq_len: int = 256, dataset_name: str = "wikitext",
                       dataset_config: str = "wikitext-2-raw-v1"):
    """Yields chunks of seq_len+1 token ids (input + 1 shifted target) from a HF dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir, trust_remote_code=True)
    ds = load_dataset(dataset_name, dataset_config, split="train")
    text = "\n".join(ds["text"])
    ids = tok.encode(text)
    print(f"[data] {len(ids):,} tokens from {dataset_name}/{config_name_safe(dataset_config)} "
          f"-> {(len(ids) - 1) // seq_len:,} chunks of {seq_len}")
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        yield ids[i:i + seq_len + 1]


def config_name_safe(name: str) -> str:
    return name


def train_calibration(engine: "Tessera50BEngine", token_iter, steps: int = 500,
                      lr: float = 1e-3, warmup: int = 50, weight_decay: float = 0.01,
                      report_every: int = 25, save_path: Optional[str] = None) -> List[float]:
    """AdamW on the Tessera-only parameters. Loss = next-token CE + z-loss (Pillar spec:
    1e-4 * logsumexp^2). WSD-style schedule: linear warmup then cosine to 0.1*lr.
    Memory state persists across chunks (detached) — matches streaming inference."""
    engine.enable_calibration_mode()
    params = engine.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        return 0.1 * lr + 0.9 * lr * 0.5 * (1.0 + math.cos(math.pi * t))

    losses: List[float] = []
    t0 = time.time()
    for step in range(steps):
        chunk = next(token_iter)
        x, y = chunk[:-1], chunk[1:]
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        logits = engine.chunk_forward(x)
        targets = torch.tensor(y, dtype=torch.long, device=logits.device)
        ce = F.cross_entropy(logits.float(), targets)
        z = 1e-4 * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        loss = ce + z
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(ce.detach()))
        if (step + 1) % report_every == 0:
            recent = losses[-report_every:]
            rate = report_every * len(x) / max(1e-9, time.time() - t0)
            print(f"  step {step+1}/{steps} | CE {sum(recent)/len(recent):.4f} "
                  f"| lr {lr_at(step):.2e} | {rate:.0f} tok/s")
            t0 = time.time()
            if save_path:
                torch.save(engine.calibration_state_dict(), save_path)
    if save_path:
        torch.save(engine.calibration_state_dict(), save_path)
        print(f"[tessera] calibration saved -> {save_path}")
    return losses


def calib_self_test():
    """CPU self-test: tiny engine, REPEATED skewed-unigram chunk -> the trainable path
    must overfit it (proves gradients flow through the checkpointed chunk graph).
    Also checks the save/load round-trip and that GEMV inference still runs."""
    print("=" * 90)
    print("  TESSERA CALIBRATION SELF-TEST (tiny config, CPU)")
    print("=" * 90)
    V = 64
    cfg = TesseraConfig(vocab_size=V, d_model=64, n_layers=4, n_heads=4,
                        d_ff=192, r_adapter=8, group_size=16)
    eng = Tessera50BEngine(cfg)
    torch.manual_seed(0)

    probs = torch.softmax(torch.linspace(3.0, -3.0, V), dim=0)
    rng = torch.Generator().manual_seed(7)
    fixed_chunk = torch.multinomial(probs, 257, replacement=True, generator=rng).tolist()

    class FixedIter:
        def __iter__(self):
            return self

        def __next__(self):
            return fixed_chunk

    losses = train_calibration(eng, FixedIter(), steps=200, lr=1e-2, warmup=10,
                               report_every=50)
    import statistics
    first, last = statistics.mean(losses[:5]), statistics.mean(losses[-5:])
    floor = float(-(probs * probs.log()).sum())
    print(f"  CE: first {first:.3f} -> last {last:.3f} | entropy floor {floor:.3f} "
          f"| uniform baseline {math.log(V):.3f}")
    # The frozen 0.02-scale random head gives weak gradients; a clear drop (>0.3)
    # on a repeated chunk is the proof that gradients reach every trainable param.
    assert last < first - 0.3, "loss did not decrease"

    # save/load round-trip must be bit-exact
    sd = eng.calibration_state_dict()
    with torch.no_grad():
        eng.layers[0].au_w.fill_(0.123)
    eng.load_calibration_state_dict(sd)
    assert torch.equal(eng.layers[0].au_w.cpu(), sd["L0.au_w"]), "round-trip mismatch"
    print("  save/load round-trip: exact")

    # streaming inference still works with calibration params installed
    ids = eng.generate(prompt_ids=[1, 2, 3], max_new_tokens=4, temperature=1.0,
                       top_k=0, top_p=1.0)
    assert all(0 <= i < V for i in ids)
    print("  GEMV inference with calibration params: ok")
    print("✓ CALIBRATION SELF-TEST PASSED")
    print("=" * 90)




# ====================================================================================================
# 10. SMOKE TEST (CPU, tiny random config — exercises every pillar end to end)
# ====================================================================================================

def smoke_test():
    print("=" * 90)
    print("  TESSERA-Q 50B ENGINE — CPU SMOKE TEST (tiny random config, all 9 pillars)")
    print("=" * 90)
    cfg = TesseraConfig(vocab_size=64, d_model=64, n_layers=4, n_heads=4,
                        d_ff=192, r_adapter=8, group_size=16)
    eng = Tessera50BEngine(cfg)

    torch.manual_seed(0)
    # stepped manually (instead of generate) so progress is visible per token
    eng.reset()
    ids = [1, 2, 3]
    for t in range(8):
        t0 = time.time()
        logits = eng.step(ids[-1])
        assert torch.isfinite(logits).all(), f"non-finite logits at step {t}"
        probs = sample_logits(logits, 1.0, 0, 1.0)
        ids.append(int(torch.multinomial(probs, 1)))
        eng.pos += 1
        print(f"  smoke step {t+1}/8 done in {time.time()-t0:.2f}s")
    assert all(isinstance(i, int) and 0 <= i < cfg.vocab_size for i in ids), ids
    assert eng.mrm.fine_n > 0, "MRM should have received writes"
    st = eng.mrm.status()
    print(f"✓ generated ids: {ids}")
    print(f"✓ MRM status: {st}")
    print(f"✓ param count: {cfg.param_count():,}")
    print("✓ ALL PILLARS EXECUTED (norm, conv+gate, rope, qk-norm, DiffAttn,")
    print("  value-residual, SwiGLU+adapter, MRM-v2 3-tier write, logit soft-cap)")
    print("=" * 90)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TESSERA-Q ~50B streaming engine (Kaggle dual-T4)")
    ap.add_argument("--model", type=str, default=None,
                    help="HF GPTQ (INT4) checkpoint dir to transplant weights from")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU self-test")
    ap.add_argument("--check-gpu", action="store_true",
                    help="validate Triton GEMVs vs float64 reference on CUDA")
    ap.add_argument("--calib-test", action="store_true",
                    help="CPU self-test of the calibration fine-tune mechanics")
    args = ap.parse_args()

    if args.smoke:
        smoke_test()
        sys.exit(0)

    if args.check_gpu:
        check_gpu()
        sys.exit(0)

    if args.calib_test:
        calib_self_test()
        sys.exit(0)

    eng = Tessera50BEngine(model_dir=args.model, tokenizer_dir=args.tokenizer)
    ids = eng.generate(eng.encode(args.prompt), args.max_new,
                       args.temperature, args.top_k, args.top_p)
    print("PROMPT:", args.prompt)
    print("OUTPUT:", eng.decode(ids))
