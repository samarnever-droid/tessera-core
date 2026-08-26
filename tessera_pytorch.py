#!/usr/bin/env python3
"""PyTorch implementation of TESSERA-Q with Progressive Folding Attention + MRM-v2."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResMemoryV2PyTorch(nn.Module):
    """PyTorch implementation of Multi-Resolution Working Memory (MRM-v2) with LRQ eviction."""

    def __init__(self, d_model=128, k_fine=128, k_coarse=16):
        super().__init__()
        self.d_model = d_model
        self.k_fine = k_fine
        self.k_coarse = k_coarse

        # Projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.w_gate = nn.Linear(d_model * 2, 1, bias=True)
        self.w_salience = nn.Linear(d_model, 1, bias=True)

        # Initialize gate bias with negative value to stabilize early training
        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, -2.0)
        self.w_out.weight.data.mul_(0.1)

    def forward(self, h_seq):
        b, t, d = h_seq.shape
        q = self.w_q(h_seq)
        k = self.w_k(h_seq)
        v = self.w_v(h_seq)

        # Causal Attention within working buffer
        scale = 1.0 / math.sqrt(d)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=h_seq.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0)
        attn_weights = F.softmax(scores, dim=-1)
        m_retrieved = torch.bmm(attn_weights, v)

        m_proj = self.w_out(m_retrieved)
        gate_input = torch.cat([h_seq, m_proj], dim=-1)
        gate = torch.sigmoid(self.w_gate(gate_input))

        return h_seq + gate * m_proj


class TesseraStagePyTorch(nn.Module):
    """TESSERA Folding Stage with Causal Attention + SwiGLU + Adapter."""

    def __init__(self, d_model=128, d_ffn=512, r_adapter=8, window_size=64):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # Causal Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.RMSNorm(d_model)

        # SwiGLU FFN
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w1u = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)

        # Low-rank stage adapter
        self.adapter_down = nn.Linear(d_model, r_adapter, bias=False)
        self.adapter_up = nn.Linear(r_adapter, d_model, bias=False)
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, x):
        b, t, d = x.shape
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        scale = 1.0 / math.sqrt(d)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, v)
        h_mid = x + self.o_proj(ctx)

        # SwiGLU + Adapter
        h_norm = self.norm2(h_mid)
        swiglu = F.silu(self.w1(h_norm)) * self.w1u(h_norm)
        ffn_out = self.w2(swiglu)
        adapter_out = self.adapter_up(self.adapter_down(h_norm))

        return h_mid + ffn_out + adapter_out


class TesseraModelPyTorch(nn.Module):
    """Full TESSERA-Q Model in PyTorch."""

    def __init__(self, vocab_size=256, d_model=128, d_ffn=512, n_stages=2, r_adapter=8, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model) * (1.0 / math.sqrt(d_model)))

        self.stages = nn.ModuleList([
            TesseraStagePyTorch(d_model, d_ffn, r_adapter) for _ in range(n_stages)
        ])
        self.mrm = MultiResMemoryV2PyTorch(d_model=d_model, k_fine=128, k_coarse=16)
        self.norm_final = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.embed(x) + self.pos_embed[pos]

        for stage in self.stages:
            h = stage(h)

        h = self.mrm(h)
        h = self.norm_final(h)
        return self.head(h)
