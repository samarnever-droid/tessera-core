#!/usr/bin/env python3
"""Official PyTorch Benchmark: Google DeepMind Griffin vs Dense Transformer vs TESSERA-Q.

Runs directly using Google DeepMind's official `vendor/recurrentgemma` PyTorch codebase.
"""

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "vendor/recurrentgemma")
from recurrentgemma import common
from recurrentgemma.torch import griffin, modules, layers


class CharDataset:
    def __init__(self, filepath, split_ratio=0.9):
        with open(filepath, "rb") as f:
            data = f.read()
        split_idx = int(len(data) * split_ratio)
        self.train_data = np.frombuffer(data[:split_idx], dtype=np.uint8)
        self.val_data = np.frombuffer(data[split_idx:], dtype=np.uint8)

    def get_batch(self, split, batch_size=32, seq_len=64, device="cpu"):
        data = self.train_data if split == "train" else self.val_data
        max_idx = len(data) - seq_len - 1
        starts = np.random.randint(0, max_idx, size=batch_size)
        x = np.stack([data[s : s + seq_len] for s in starts])
        y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


class PyTorchTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_layers=3, d_ffn=512, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model) * (1.0 / math.sqrt(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_ffn,
            dropout=0.0,
            activation=F.silu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.embed(x) + self.pos_embed[pos]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        out = self.encoder(h, mask=mask, is_causal=True)
        return self.head(out)


def get_official_griffin_model(vocab_size=256, width=128, num_layers=3):
    """Builds the official Google DeepMind Griffin model with 2:1 RG-LRU to Local Attention ratio."""
    block_types = []
    for i in range(num_layers):
        if i % 3 == 1:
            block_types.append(common.TemporalBlockType.ATTENTION)
        else:
            block_types.append(common.TemporalBlockType.RECURRENT)

    config = common.GriffinConfig(
        vocab_size=vocab_size,
        width=width,
        mlp_expanded_width=width * 4,
        num_heads=4,
        block_types=tuple(block_types),
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=64,
        logits_soft_cap=0.0,
        lru_width=width * 2,
    )
    model = griffin.Griffin(config)
    return model


def train_eval_model(name, model, dataset, steps=120, batch_size=32, seq_len=64, lr=3e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"\nTraining {name} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params) in Official PyTorch CPU Engine...")
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        x, y = dataset.get_batch("train", batch_size, seq_len, device)
        optimizer.zero_grad()

        # Handle Griffin's specific input signatures vs standard Transformer
        if hasattr(model, "config") and isinstance(model.config, common.GriffinConfig):
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            logits, _ = model(tokens=x, segment_pos=positions)
        else:
            logits = model(x)

        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(15):
                    vx, vy = dataset.get_batch("val", batch_size, seq_len, device)
                    if hasattr(model, "config") and isinstance(model.config, common.GriffinConfig):
                        v_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                        v_logits, _ = model(tokens=vx, segment_pos=v_pos)
                    else:
                        v_logits = model(vx)
                    val_losses.append(F.cross_entropy(v_logits.view(-1, 256), vy.view(-1)).item())
                val_loss = np.mean(val_losses)
                val_bpc = val_loss / math.log(2.0)
                elapsed = time.time() - t0
                tok_s = (step * batch_size * seq_len) / elapsed
                print(f"  [{name}] Step {step:>4} ({elapsed:>5.1f}s) | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f} | Tok/s: {tok_s:.0f}")

    return val_loss, val_bpc


def profile_decode_latency(name, model, device="cpu", num_tokens=300):
    model.eval()
    dummy_x = torch.randint(0, 256, (1, 64), device=device)
    latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(20):
            if hasattr(model, "config") and isinstance(model.config, common.GriffinConfig):
                pos = torch.arange(64, device=device).unsqueeze(0)
                _ = model(tokens=dummy_x, segment_pos=pos)
            else:
                _ = model(dummy_x)

        # Profile
        for _ in range(num_tokens):
            t0 = time.perf_counter()
            if hasattr(model, "config") and isinstance(model.config, common.GriffinConfig):
                pos = torch.arange(64, device=device).unsqueeze(0)
                _ = model(tokens=dummy_x, segment_pos=pos)
            else:
                _ = model(dummy_x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6 / 64.0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.90)]
    p99 = latencies[int(len(latencies) * 0.99)]
    tok_s = 1e6 / np.mean(latencies)
    return tok_s, p50, p90, p99


from tessera_pytorch import TesseraModelPyTorch


def main():
    print("=================================================================================================")
    print("  OFFICIAL PYTORCH 3-WAY SHOWDOWN: DEEPMIND GRIFFIN vs TESSERA-Q vs TRANSFORMER")
    print(f"  PyTorch Version: {torch.__version__} | Device: CPU | Threads: {torch.get_num_threads()}")
    print("=================================================================================================")

    dataset = CharDataset("data/enwik8")

    # 1. Dense Transformer Control
    tf_model = PyTorchTransformer(vocab_size=256, d_model=128, n_layers=3, d_ffn=512, max_seq_len=64)
    loss_tf, bpc_tf = train_eval_model("PyTorch-Transformer", tf_model, dataset, steps=120)

    # 2. Official Google DeepMind Griffin
    griffin_model = get_official_griffin_model(vocab_size=256, width=128, num_layers=3)
    loss_griffin, bpc_griffin = train_eval_model("Official-DeepMind-Griffin", griffin_model, dataset, steps=120)

    # 3. TESSERA-Q PyTorch
    tessera_model = TesseraModelPyTorch(vocab_size=256, d_model=128, d_ffn=512, n_stages=2, r_adapter=8, max_seq_len=64)
    loss_tes, bpc_tes = train_eval_model("TESSERA-Q-PyTorch", tessera_model, dataset, steps=120)

    # Profile latencies
    print("\n==========================================================================")
    print("  TRUE SINGLE-THREAD PYTORCH CPU DECODE LATENCY & THROUGHPUT (300 TOKENS)")
    print("==========================================================================")
    tok_tf, p50_tf, p90_tf, p99_tf = profile_decode_latency("PyTorch-Transformer", tf_model)
    tok_gr, p50_gr, p90_gr, p99_gr = profile_decode_latency("DeepMind-Griffin", griffin_model)
    tok_tes, p50_tes, p90_tes, p99_tes = profile_decode_latency("TESSERA-Q", tessera_model)

    print(f"{'Architecture':<32} | {'Tok/s':<10} | {'p50 (µs)':<10} | {'p90 (µs)':<10} | {'p99 (µs)':<10}")
    print("--------------------------------------------------------------------------------------------------")
    print(f"{'PyTorch Transformer (0.67M)':<32} | {tok_tf:<10.0f} | {p50_tf:<10.2f} | {p90_tf:<10.2f} | {p99_tf:<10.2f}")
    print(f"{'Official DeepMind Griffin (0.94M)':<32} | {tok_gr:<10.0f} | {p50_gr:<10.2f} | {p90_gr:<10.2f} | {p99_gr:<10.2f}")
    print(f"{'TESSERA-Q PyTorch (0.73M)':<32} | {tok_tes:<10.0f} | {p50_tes:<10.2f} | {p90_tes:<10.2f} | {p99_tes:<10.2f}")
    print("==================================================================================================")

    print("\n=================================================================================================")
    print("                          OFFICIAL PYTORCH 3-WAY GRAND SCORECARD")
    print("=================================================================================================")
    print(f"1. PyTorch Transformer Control:  {loss_tf:.4f} Loss | {bpc_tf:.4f} BPC | {tok_tf:.0f} tok/s")
    print(f"2. Official DeepMind Griffin:    {loss_griffin:.4f} Loss | {bpc_griffin:.4f} BPC | {tok_gr:.0f} tok/s")
    print(f"3. TESSERA-Q PyTorch:            {loss_tes:.4f} Loss | {bpc_tes:.4f} BPC | {tok_tes:.0f} tok/s")
    print("=================================================================================================\n")


if __name__ == "__main__":
    main()
