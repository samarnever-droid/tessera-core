#!/usr/bin/env python3
"""Needle Recall Probe for Official Google DeepMind Griffin PyTorch RGLRU Layer.

Tests the official `RGLRU` implementation from `vendor/recurrentgemma` across 1K, 4K, and 8K tokens.
"""

import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "vendor/recurrentgemma")
from recurrentgemma.torch import layers


def probe_official_rglru_needle(context_length, num_trials=20, width=128, num_heads=4):
    lru = layers.RGLRU(width=width, num_heads=num_heads)
    lru.eval()

    successes = 0
    cosine_sims = []

    with torch.no_grad():
        for trial in range(num_trials):
            torch.manual_seed(10000 + trial)
            # 1. Target needle vector
            needle = torch.randn(1, 1, width)
            needle = F.normalize(needle, dim=-1)

            # Pass needle through RGLRU to initialize recurrent state
            # RGLRU forward: (x, segment_pos, cache) -> (output, new_cache)
            _, h_state = lru(x=needle, segment_pos=torch.zeros((1, 1), dtype=torch.long))

            # 2. Stream N distraction tokens through official RGLRU recurrence
            distraction = torch.randn(1, context_length, width)
            segment_pos = torch.arange(1, context_length + 1).unsqueeze(0)
            
            # Pass distraction through RGLRU initialized with needle state
            _, final_state = lru(
                x=distraction,
                segment_pos=segment_pos,
                cache=h_state,
            )

            # 3. Measure cosine similarity between original needle and final state vector
            needle_vec = needle.squeeze()
            state_vec = final_state.squeeze()
            
            cos = F.cosine_similarity(needle_vec.unsqueeze(0), state_vec.unsqueeze(0)).item()
            cosine_sims.append(cos)
            if cos >= 0.70:
                successes += 1

    avg_cos = np.mean(cosine_sims)
    recall_pct = (successes / num_trials) * 100.0
    return recall_pct, avg_cos


def main():
    print("=================================================================================================")
    print("  OFFICIAL GOOGLE DEEPMIND GRIFFIN (RGLRU) PYTORCH NEEDLE RECALL PROBE")
    print("=================================================================================================")
    print(f"{'Context Length':<16} | {'Official DeepMind Griffin Recall':<32} | {'Average Cosine Similarity':<25}")
    print("-------------------------------------------------------------------------------------------------")

    for ctx in [1024, 4096, 8192]:
        recall_pct, avg_cos = probe_official_rglru_needle(ctx)
        print(f"{ctx:<11} tok   | {recall_pct:>6.1f}%                          | {avg_cos:>8.4f}")

    print("=================================================================================================\n")


if __name__ == "__main__":
    main()
