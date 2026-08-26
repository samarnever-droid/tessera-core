#!/usr/bin/env python3
"""
Colab/GPU Training Script for TESSERA-Q with OpenAI Triton MRM & Warmup-Stable-Decay (WSD).
Usage in Colab:
  !pip install triton torch
  !python train_tessera_gpu.py --dataset data/enwik8 --steps 120 --batch_size 32
"""

import argparse
import math
import os
import sys
import time
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from tessera_triton import TesseraGPUModel, TRITON_AVAILABLE
except ImportError:
    print("[!] PyTorch not found. Install via: pip install torch triton")
    sys.exit(0)


class CharDataset(Dataset):
    """Memory-mapped byte-level dataset loader."""
    def __init__(self, data_bytes: bytes, seq_len: int = 64):
        self.data = torch.from_numpy(
            torch.ByteTensor(list(data_bytes)).numpy()
        ).long()
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def train_gpu():
    parser = argparse.ArgumentParser(description="TESSERA-Q GPU Trainer with Triton")
    parser.add_argument("--dataset", type=str, default="data/enwik8")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--export_bin", type=str, default="tessera_trained_gpu.bin")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 85)
    print("  TESSERA-Q GPU ACCELERATED TRAINER WITH OPENAI TRITON")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"  Triton Fused Kernels: {'ENABLED (CUDA SRAM)' if TRITON_AVAILABLE else 'FALLBACK (Vectorized PyTorch)'}")
    print(f"  Dataset: {args.dataset} | Steps: {args.steps} | Batch Size: {args.batch_size}")
    print("=" * 85 + "\n")

    if not os.path.exists(args.dataset):
        print(f"[!] Dataset {args.dataset} not found. Creating synthetic fallback...")
        raw_data = os.urandom(1_000_000)
    else:
        with open(args.dataset, "rb") as f:
            raw_data = f.read()

    split_idx = int(len(raw_data) * 0.9)
    train_bytes = raw_data[:split_idx]
    val_bytes = raw_data[split_idx:]

    train_dataset = CharDataset(train_bytes, args.seq_len)
    val_dataset = CharDataset(val_bytes, args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TesseraGPUModel(vocab_size=256, d_model=128, d_ff=768, n_stages=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Warmup-Stable-Decay (WSD) Scheduler
    def get_lr(step, total_steps, base_lr):
        warmup_steps = 10
        decay_start = int(total_steps * 0.75) # Step 90
        min_lr = 3.5e-4
        if step <= warmup_steps:
            return min_lr + (step / warmup_steps) * (base_lr - min_lr)
        elif step <= decay_start:
            return base_lr
        else:
            prog = (step - decay_start) / max(1, total_steps - decay_start)
            return min_lr + 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))) * (base_lr - min_lr)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Update LR via WSD schedule
        current_lr = get_lr(step, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        out = model(x, targets=y)
        loss = out["loss"]
        loss.backward()

        # Gradient Clipping (max norm 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1 or step == args.steps:
            model.eval()
            with torch.no_grad():
                try:
                    vx, vy = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    vx, vy = next(val_iter)

                vx, vy = vx.to(device), vy.to(device)
                vout = model(vx, targets=vy)
                val_loss = vout["ntp_loss"].item()
                val_bpc = vout["bpc"].item()

            elapsed = time.time() - start_time
            tok_s = (step * args.batch_size * args.seq_len) / max(1e-4, elapsed)

            print(f"[TESSERA-GPU] Step {step:>4} ({elapsed:>5.1f}s) | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f} | LR: {current_lr:.2e} | Tok/s: {tok_s:.0f}")

    # Export weights for Rust CPU Engine
    model.export_to_binary(args.export_bin)
    print("\n" + "=" * 85)
    print(f"  ✓ Training Finished. Model exported to {args.export_bin}")
    print("  -> Load directly into Rust CPU inference engine via zero-copy mmap.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    train_gpu()
