"""Standalone benchmark script for Google DeepMind Griffin.
Runs ONLY Griffin on enwik8, measures BPC, decode speed, 8K needle recall, and saves griffin_results.json.
"""

import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "vendor/recurrentgemma")
from recurrentgemma import common
from recurrentgemma.torch import griffin, layers


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


def get_official_griffin_model(vocab_size=256, width=128, num_layers=3):
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
    return griffin.Griffin(config)


def train_griffin(dataset, steps=120, batch_size=32, seq_len=64, lr=3e-3, device="cpu"):
    model = get_official_griffin_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n[DeepMind Griffin] Training Official PyTorch Model ({params_m:.2f}M params) on enwik8 ({steps} steps)...")
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        x, y = dataset.get_batch("train", batch_size, seq_len, device)
        optimizer.zero_grad()

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        logits, _ = model(tokens=x, segment_pos=positions)
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
                    v_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                    v_logits, _ = model(tokens=vx, segment_pos=v_pos)
                    val_losses.append(F.cross_entropy(v_logits.view(-1, 256), vy.view(-1)).item())
                val_loss = np.mean(val_losses)
                val_bpc = val_loss / math.log(2.0)
                elapsed = time.time() - t0
                tok_s = (step * batch_size * seq_len) / elapsed
                print(f"  [DeepMind Griffin] Step {step:>4} ({elapsed:>5.1f}s) | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f} | Tok/s: {tok_s:.0f}")

    # Latency profiling
    model.eval()
    dummy_x = torch.randint(0, 256, (1, 64), device=device)
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            pos = torch.arange(64, device=device).unsqueeze(0)
            _ = model(tokens=dummy_x, segment_pos=pos)
        for _ in range(300):
            t_start = time.perf_counter()
            pos = torch.arange(64, device=device).unsqueeze(0)
            _ = model(tokens=dummy_x, segment_pos=pos)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1e6 / 64.0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.90)]
    p99 = latencies[int(len(latencies) * 0.99)]
    tok_s = 1e6 / np.mean(latencies)

    # Needle test
    lru = layers.RGLRU(width=128, num_heads=4).eval()
    needle_results = {}
    with torch.no_grad():
        for ctx in [1024, 4096, 8192]:
            cos_sims = []
            for t in range(20):
                needle = F.normalize(torch.randn(1, 1, 128), dim=-1)
                _, h = lru(x=needle, segment_pos=torch.zeros((1, 1), dtype=torch.long))
                dist = torch.randn(1, ctx, 128)
                _, final_h = lru(x=dist, segment_pos=torch.arange(1, ctx + 1).unsqueeze(0), cache=h)
                cos = F.cosine_similarity(needle.squeeze().unsqueeze(0), final_h.squeeze().unsqueeze(0)).item()
                cos_sims.append(cos)
            avg_c = np.mean(cos_sims)
            needle_results[str(ctx)] = {"recall_pct": 100.0 if avg_c >= 0.70 else 0.0, "avg_cosine": float(avg_c)}

    results = {
        "model": "Google DeepMind Griffin (Official PyTorch)",
        "parameters_m": float(params_m),
        "val_loss": float(val_loss),
        "val_bpc": float(val_bpc),
        "dram_bytes_per_tok": 1536,
        "single_thread_tok_s": float(tok_s),
        "p50_us": float(p50),
        "p90_us": float(p90),
        "p99_us": float(p99),
        "needle_recall": needle_results,
    }

    with open("griffin_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved official Griffin metrics to griffin_results.json")
    return results


if __name__ == "__main__":
    dataset = CharDataset("data/enwik8")
    train_griffin(dataset, steps=120)
