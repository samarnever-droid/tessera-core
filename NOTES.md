# Research Log & Phase Gate Results

## Part 1: AXIOM Zero-Cross-Layer Architecture Evaluation
**Summary**:
- **Phase 0 (Kernels)**: Microbenchmarks revealed dense CPU latency gaps (~9.45ms/layer at d=1024), projecting single-stream sequential inference at ~13-26 tok/s.
- **Phase 1 (Single Layer)**: Pure local-in-time loss plateaued at 5.14 BPC (unigram/bigram floor). Adding within-layer BPTT reached 3.0967 BPC on `enwik8` (matching the matched 128-dim GRU baseline at 3.1303 BPC, training at ~8.2k tok/s).
- **Phase 2 (Multi-Layer)**: Simultaneous decoupled 4-layer training failed Gate C2 (monotonicity inverted: L1=3.82 BPC vs L4=3.86 BPC) and Gate C7 (1,686 tok/s vs Transformer 1,478 tok/s, ~parity rather than 10-50x speedup). Diagnosed as greedy local representation collapse and upstream drift.

---

## Part 2: STRATUM Pre-Build Falsification Test (Experiment F4)
**Date**: 2026-08-26
**Commit / Status**: COMPLETED (Falsification Criteria Evaluated)

### 1. Controlled Experiment Matrix on enwik8 (d=128, batch=32, T=64, 120 steps / 245.7k tokens)

| Model Configuration | Total Params ($P_{\text{total}}$) | Active Params / Tok ($P_{\text{active}}$) | Val Loss | Val BPC | Bytes Read / Tok | Tok/s | Peak RAM |
|---|---|---|---|---|---|---|---|
| **Dense Transformer (Control)** | **0.47 M** | 0.47 M | **2.9908** | **4.3148** | 1.88 MB | 3,901 | 239.7 MB |
| **STRATUM ($N=256$ slots, $m=16$)** | **0.21 M** | 0.14 M | 3.2431 | **4.6788** | 0.56 MB | 23,477 | 232.7 MB |
| **STRATUM ($N=4,096$ slots, $m=64$)** | **0.70 M** | 0.14 M | 3.2332 | **4.6645** | 0.56 MB | 18,727 | 236.4 MB |
| **STRATUM ($N=65,536$ slots, $m=256$)** | **8.59 M** | 0.17 M | 3.2322 | **4.6631** | 0.68 MB | 7,444 | 305.6 MB |

### 2. Sparse Routing Diagnostics & Slot Hit-Count Histogram

| Model Arm | Total Slots ($N$) | Active $k$ | Slot Utilization | Mean Routing Entropy | Median / Mean Update Ratio | Diagnostic Status |
|---|---|---|---|---|---|---|
| **STRATUM-N256** | 256 | 16 | **100.0%** | 0.91 | **0.1571** | Healthy |
| **STRATUM-N4096** | 4,096 | 16 | **70.0%** | 1.09 | **0.1143** | 30% dead slots |
| **STRATUM-N65536** | 65,536 | 16 | **10.1%** | 0.96 | **0.0000** | **Catastrophic routing collapse** (median=0) |

- **$N=65,536$ Histogram**:
  - $0\text{–}371\text{ updates}$: $65,500\text{ slots}$ ($99.94\%$ of slot store received near-zero updates)
  - $>371\text{ updates}$: $36\text{ slots}$ (Top ~30 slots absorbed all gradient updates)

### 3. Pre-Registered Decision Rule & Parameter Multiplier
- **Pre-Registered Ladder**:
  - $<1.5\times$: Strong win
  - $1.5\text{–}2.5\times$: Plausible economic advantage
  - $2.5\text{–}8\times$: Marginal
  - $>8\times$: **Capacity thesis failed**
- **Measured Result**:
  - Dense Transformer ($0.47\text{M}$ params) = **4.3148 BPC**.
  - STRATUM-65k ($8.59\text{M}$ params) = **4.6631 BPC**.
  - Parameter Multiplier for Equal Loss: **$> 18.4\times$ (Fails to intersect)**.
  - Scaling $N$ from $256 \to 65,536$ ($40\times$ parameter growth) reduced BPC by only $0.015\text{ BPC}$ ($0.3\%$).
- **Verdict**: **[>8.0x] CORE CAPACITY THESIS FALSIFIED**.
