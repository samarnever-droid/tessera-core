# Running a ~50B TESSERA-Q model on Kaggle 2× T4

**Engine:** [tessera_50b_kaggle.py](tessera_50b_kaggle.py) — full 9-pillar TESSERA-Q forward
pass in PyTorch + Triton, 4-bit GPTQ weights, layers split across both T4s, token-streaming
decode through MRM-v2 working memory (no KV cache).

## Why ~50B fits on 2× T4 (the memory math)

| Item | Size |
|---|---|
| Kaggle dual T4 | 2 × 16 GB = **32 GB** VRAM, ~30 GB system RAM, ~57 GB disk |
| 50B params @ INT4 GPTQ | ~0.5 byte/param ≈ **25 GB** + <2% scale overhead |
| Tied embedding (152k × 6144, fp16) | ~1.9 GB |
| Activations (GEMV, M=1 streaming) | < 0.5 GB |
| MRM-v2 state (128+16 slots × 6144) | ~3.5 MB — **constant**, independent of context |

Total ≈ **27 GB**, so with layers `[0..27)` on `cuda:0` and `[27..55)` on `cuda:1` each GPU
holds ~13.5 GB and has ~2.5 GB of headroom. Decode has **no KV cache at all**: Tessera's
Pillar-8 working memory replaces it with a fixed 144-slot Hopfield memory, so context length
costs nothing.

**A note on "50B":** no mainstream open dense model is exactly 50B. The engine derives its
config from whatever GPTQ checkpoint you give it (any layer count / head count / GQA). At
INT4 the resident ceiling on 2× T4 is roughly **52B dense params**. Practical picks:

- `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` (~18 GB, comfortable, verified shape family)
- `TheBloke/Llama-2-70B-chat-GPTQ` → **does NOT fit** (needs ~38 GB) — don't use on this setup
- Any future ~50B dense GPTQ-Int4 release → fits at the limit with this engine

## Kaggle notebook, cell by cell

**Cell 1 — GPU + deps** (Settings → Accelerator → **GPU T4 ×2** first):

```python
!nvidia-smi   # must show 2 x Tesla T4
!pip -q install safetensors transformers accelerate
```
(Triton and torch are already in Kaggle's image; `triton_enabled()` auto-detects them.)

**Cell 2 — get the engine + weights:**

```python
# Option A: this repo
!git clone <your-repo-url> repo
%cd repo

# Option B: download a GPTQ checkpoint (dataset-mounted is faster & persists across sessions)
from huggingface_hub import snapshot_download
MODEL_DIR = snapshot_download("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
                              local_dir="/kaggle/working/qwen32b-gptq",
                              allow_patterns=["*.json", "*.safetensors", "*.safetensors.index.json"])
```
> Recommended: upload the checkpoint once as a **private Kaggle Dataset** and attach it —
> `/kaggle/input/<dataset>/` reads count against disk streaming, not your 30 GB RAM.

**Cell 3 — smoke test (CPU, 10 seconds, validates all 9 pillars):**

```python
!python tessera_50b_kaggle.py --smoke
```

**Cell 4 — run generation on both T4s:**

```python
import tessera_50b_kaggle as tk, torch, time

eng = tk.Tessera50BEngine(model_dir="/kaggle/working/qwen32b-gptq")
# engine prints the layer split, param count, and Rust-parity MRM thresholds

ids = eng.encode("Explain what a multi-resolution working memory is, in one paragraph.")
t0 = time.time()
out = eng.generate(ids, max_new_tokens=128, temperature=0.8, top_k=50, top_p=0.95)
dt = time.time() - t0
print(eng.decode(out))
print(f"\n{128/dt:.2f} tok/s | MRM: {eng.mrm.status()}")
```

**Cell 5 (optional) — watch the working memory tiers fire:**

```python
eng.reset()
for tok in eng.encode("The Eiffel Tower is in Paris.")[:8]:
    eng.step(tok)
print(eng.mrm.status())   # fine_occupied / coarse_occupied grow, thresholds shrink with d
```

## What to expect

- **First run**: Triton JIT-compiles the GPTQ GEMV kernel per shape (~1–2 min one-off).
- **Throughput**: GEMV decode on T4s with 4-bit weights lands around **1–4 tok/s** for a
  32–50B class model. The bottleneck is memory bandwidth (~300 GB/s per T4, and every one
  of the ~55 layer visits streams ~0.4 GB of packed weights per token).
- **Prefill** is token-by-token (streaming) — for long prompts, expect proportional wait.
- **Quality caveat, stated honestly**: the Q/K/V/O and FFN weights are transplanted from the
  source model; TESSERA-only components (depthwise conv, adaptive-RoPE eta, low-rank
  adapters, DiffAttn λ, value-residual gate) start at their principled defaults. Output
  quality will be below the source model until those components are fine-tuned
  (see the note in `convert_open_weights_to_tessera.py`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` on load | Use a smaller checkpoint; the split is printed at load — a >52B INT4 model cannot fit resident |
| `NameError: dataclass` in `tessera_qwen_72b_triton_kaggle.py` | Known bug in the older engine (missing `from dataclasses import dataclass`, `import torch.nn.functional as F`) — use this engine instead |
| Triton compile error on T4 | Engine auto-falls back to the pure-PyTorch GEMV path (slower, identical math) |
| Tokenizer weirdness | Pass `--tokenizer` / `tokenizer_dir=` pointing at the original HF repo |
