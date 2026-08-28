#!/usr/bin/env python3
"""
TESSERA-Q Frontier PyTorch Reference Implementation.
Architectural Parity with the Native Rust Engine:
1. Microsoft Differential Attention (DiffAttn) with learnable lambda cancelation.
2. Adaptive Rotary Position Embeddings (RoPE) with learnable per-band scale (eta).
3. Per-head QK-RMSNorm preventing attention entropy collapse.
4. GAU-style 1D Depthwise Causal Convolution (kernel_size=4).
5. Value Residual Connection across progressive hierarchy stages.
6. SwiGLU FFN with Low-Rank Stage Adapters (r=8).
7. Multi-Resolution Working Memory (MRM-v2) with sharp cosine temperature softmax.
8. Tied Output Logits with Tanh Soft-Capping (cap=30.0).
9. PaLM Auxiliary Z-Loss and DeepSeek-V3 Multi-Token Prediction (MTP, t+2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.gamma


class AdaptiveRoPE(nn.Module):
    def __init__(self, d_head: int, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.base = base
        half = d_head // 2
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2 / d_head))
        self.register_buffer("base_freqs", freqs, persistent=False)
        self.eta = nn.Parameter(torch.zeros(half))

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        b, h, t, d = x.shape
        half = d // 2
        scale = 2.0 * torch.sigmoid(self.eta)
        freqs = self.base_freqs * scale
        t_idx = torch.arange(t, device=x.device, dtype=torch.float32)
        angles = torch.outer(t_idx, freqs) # (t, half)
        cos = angles.cos().view(1, 1, t, half)
        sin = angles.sin().view(1, 1, t, half)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rot_x1
        out[..., 1::2] = rot_x2
        return out


class DifferentialAttention(nn.Module):
    """Microsoft Differential Attention (DiffAttn) with QK-Norm."""

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert n_heads % 2 == 0, "n_heads must be even for Differential Attention head pairs"
        self.n_pairs = n_heads // 2

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.rope = AdaptiveRoPE(self.d_head)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

        # Learnable lambda parameters: lambda = exp(l_q1 * l_k1) - exp(l_q2 * l_k2) + lambda_init
        self.lambda_q1 = nn.Parameter(torch.zeros(self.n_pairs, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.n_pairs, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.n_pairs, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.n_pairs, self.d_head))
        self.lambda_init = 0.8

    def forward(self, x: torch.Tensor, v0_residual: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        # Apply Adaptive RoPE
        q = self.rope(q, t)
        k = self.rope(k, t)

        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Value Residual Connection: V_s = 0.7 * V_s + 0.3 * V_0
        if v0_residual is not None:
            v = 0.7 * v + 0.3 * v0_residual
        current_v = v

        # Split into Differential head pairs (Q1, K1) and (Q2, K2)
        scale = 1.0 / math.sqrt(self.d_head)
        q1 = q[:, 0::2]
        q2 = q[:, 1::2]
        k1 = k[:, 0::2]
        k2 = k[:, 1::2]
        v_pairs = v[:, 0::2]

        scores1 = torch.matmul(q1, k1.transpose(-1, -2)) * scale
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) * scale

        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        scores1 = scores1 + causal_mask.unsqueeze(0).unsqueeze(0)
        scores2 = scores2 + causal_mask.unsqueeze(0).unsqueeze(0)

        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)

        # Compute dynamic lambda per head pair
        l1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1)).view(1, self.n_pairs, 1, 1)
        l2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1)).view(1, self.n_pairs, 1, 1)
        lam = l1 - l2 + self.lambda_init

        diff_attn = attn1 - lam * attn2
        ctx = torch.matmul(diff_attn, v_pairs) # (b, n_pairs, t, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, self.n_pairs * self.d_head)

        # Project output to d_model
        if self.n_pairs * self.d_head != d:
            ctx = F.pad(ctx, (0, d - self.n_pairs * self.d_head))
        out = self.wo(ctx)
        return out, current_v


class MultiResMemoryV2(nn.Module):
    """Multi-Resolution Working Memory (MRM-v2) in PyTorch with sharp Cosine Temperature Softmax."""

    def __init__(self, d_model: int = 128, k_fine: int = 128, k_coarse: int = 16, tau: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.k_fine = k_fine
        self.k_coarse = k_coarse
        self.tau = tau

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w_gate = nn.Parameter(torch.zeros(d_model))

        nn.init.zeros_(self.w_o.weight) # Warm start with identity residual

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        b, t, d = h_seq.shape
        q = self.w_q(h_seq)
        k = self.w_k(h_seq)
        v = self.w_v(h_seq)

        # Sharp Cosine Softmax Attention
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        cos_sim = torch.bmm(q_norm, k_norm.transpose(1, 2)) / self.tau

        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=h_seq.device), diagonal=1)
        cos_sim = cos_sim + causal_mask.unsqueeze(0)
        attn_weights = F.softmax(cos_sim, dim=-1)
        mem_ctx = torch.bmm(attn_weights, v)

        proj_out = self.w_o(mem_ctx)
        gate_raw = (h_seq * self.w_gate).sum(dim=-1, keepdim=True) - 2.0
        gate_sig = torch.sigmoid(gate_raw)

        return h_seq + gate_sig * proj_out


class TesseraStagePyTorch(nn.Module):
    """Frontier Progressive Folding Stage."""

    def __init__(self, d_model: int = 128, d_ffn: int = 512, n_heads: int = 4, r_adapter: int = 8, use_mrm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model, bias=False)
        self.gate_attn = nn.Linear(d_model, d_model, bias=False)
        self.diff_attn = DifferentialAttention(d_model, n_heads)

        self.norm2 = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w1u = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

        self.adapter_v = nn.Linear(d_model, r_adapter, bias=False)
        self.adapter_u = nn.Linear(r_adapter, d_model, bias=False)
        nn.init.zeros_(self.adapter_u.weight)

        self.mrm = MultiResMemoryV2(d_model) if use_mrm else None

    def forward(self, x: torch.Tensor, v0_res: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        h = self.norm1(x)

        # 1D Depthwise Causal Conv
        h_conv = self.conv1d(h.transpose(1, 2))[:, :, :t].transpose(1, 2)
        gate = torch.sigmoid(self.gate_attn(h))
        h_gated = h_conv * gate

        # Differential Attention with Value Residual
        attn_out, current_v = self.diff_attn(h_gated, v0_res)
        h_mid = x + attn_out

        # SwiGLU + LoRA Adapter
        h_norm2 = self.norm2(h_mid)
        swiglu = F.silu(self.w1(h_norm2)) * self.w1u(h_norm2)
        ffn_out = self.w2(swiglu)
        adapter_out = self.adapter_u(self.adapter_v(h_norm2))
        h_stage_out = h_mid + ffn_out + adapter_out

        # MRM-v2 Working Memory
        if self.mrm is not None:
            h_stage_out = self.mrm(h_stage_out)

        return h_stage_out, current_v


class TesseraModelPyTorch(nn.Module):
    """Full TESSERA-Q Frontier Model in PyTorch."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        d_ffn: int = 512,
        n_heads: int = 4,
        n_stages: int = 3,
        r_adapter: int = 8,
        use_mrm: bool = True,
        logit_cap: float = 30.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.logit_cap = logit_cap

        # Tied Embeddings
        self.embeddings = nn.Embedding(vocab_size, d_model)

        self.stages = nn.ModuleList([
            TesseraStagePyTorch(d_model, d_ffn, n_heads, r_adapter, use_mrm)
            for _ in range(n_stages)
        ])

        self.final_norm = RMSNorm(d_model)

        # Multi-Token Prediction (MTP) auxiliary branch (t+2 prediction)
        self.mtp_proj = nn.Linear(d_model, d_model, bias=False)
        self.mtp_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embeddings(input_ids)
        v0 = None

        for idx, stage in enumerate(self.stages):
            h, stage_v = stage(h, v0)
            if idx == 0:
                v0 = stage_v

        h_norm = self.final_norm(h)

        # Tied logits with tanh soft-capping: logits = 30 * tanh(W_e * h / 30)
        raw_logits = F.linear(h_norm, self.embeddings.weight)
        logits = self.logit_cap * torch.tanh(raw_logits / self.logit_cap)

        # MTP Auxiliary Prediction
        mtp_h = self.mtp_proj(h_norm)
        mtp_logits = self.mtp_head(mtp_h)

        return logits, mtp_logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        mtp_logits: torch.Tensor,
        targets: torch.Tensor,
        z_loss_weight: float = 1e-4,
        mtp_loss_weight: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, v = logits.shape
        ntp_loss = F.cross_entropy(logits.view(-1, v), targets.view(-1))

        # PaLM Z-Loss
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = z_loss_weight * (log_z ** 2).mean()

        # Multi-Token Prediction loss on t+2
        if t > 1:
            mtp_targets = targets[:, 1:]
            mtp_preds = mtp_logits[:, :-1]
            mtp_loss = mtp_loss_weight * F.cross_entropy(mtp_preds.reshape(-1, v), mtp_targets.reshape(-1))
        else:
            mtp_loss = 0.0

        total_loss = ntp_loss + z_loss + mtp_loss
        return total_loss, ntp_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TesseraModelPyTorch().to(device)
    dummy_input = torch.randint(0, 256, (2, 64), device=device)
    dummy_target = torch.randint(0, 256, (2, 64), device=device)

    logits, mtp_logits = model(dummy_input)
    total_loss, ntp_loss = model.compute_loss(logits, mtp_logits, dummy_target)
    total_loss.backward()

    print(f"PyTorch TESSERA-Q initialized successfully on {device}!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Forward-backward complete. Total Loss: {total_loss.item():.4f}, NTP Loss: {ntp_loss.item():.4f}")
