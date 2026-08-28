# TESSERA MRM-v2 — Deep-Level Source-Grounded Research Prompt

> **Purpose of this document**: A self-contained "research prompt" you can hand to an AI
> research assistant (or use yourself) to push `MultiResMemoryV2` ("MRM-v2") toward its
> theoretical and practical peak. Every claim below is grounded in the actual source at
> `crates/tessera-core/src/mrm_v2.rs` (516 lines, read in full) and its call sites in
> `tessera_model.rs`, `tessera_trainer.rs`, and `exp_tessera_0.rs`. This is not a generic
> "memory-augmented transformer" essay — it is a file/line-anchored audit plus a prioritized
> research agenda, written in the same falsificationist style as this repo's `NOTES.md`.
>
> **How to use it**: Sections 1–2 are context (skip if you already know the code). Section 3
> is the core deliverable — 11 concrete, verified findings, ranked by how much they gate
> "peak" performance. Section 4 turns each finding into a research question with a proposed
> experiment and a pass/fail kill criterion. Section 5 is a literal prompt block you can paste
> into another AI session to continue this investigation with full context.

---

## 1. What MRM-v2 Is (One-Paragraph Grounding)

MRM-v2 is a **per-token, single-layer, differentiable working-memory module** attached to the
*last* TESSERA stage only (`tessera_model.rs:436-437`, `565`: `config.use_mrm_v2 && p == n_stages - 1`).
It projects the stage's hidden state `x_t` into `q,k,v` via learned `d×d` matrices
(`w_q, w_k, w_v`), **reads** a context vector from memory using a sharp-temperature
(`tau=0.05`) cosine-softmax over a **dual-resolution store** — `k_fine` exact per-token slots
plus `k_coarse` EMA "centroid" summaries — projects that context back through `w_o`, gates it
with a learned scalar (`sigmoid(dot(x_t, w_gate) - 2.0)`) as a residual add, and only *then*
**writes** `(k_t, v_t)` into the store via a three-tier policy. This all happens causally
within `forward_sequence()` (`mrm_v2.rs:315-363`), one token at a time, entirely inside a
single call — there is no persistent state across separate `forward_sequence` invocations.

## 2. Architecture Anchors (File:Line Reference Table)

| Component | Location | Notes |
|---|---|---|
| `MultiResMemoryV2` struct | `mrm_v2.rs:77-96` | `w_q/w_k/w_v/w_o` are `d×d`; `w_gate` is `d`; fine store is `k_fine×d` keys+vals+salience+hits; coarse store is `k_coarse×d` centroids+vals |
| Tier 1 — Hard Overwrite (`sim≥0.95`) | `mrm_v2.rs:191-197` | in-place replace, `fine_hits += 1.0` (cap 50) |
| Tier 2 — Soft Merge (`0.82≤sim<0.95`) | `mrm_v2.rs:198-209` | EMA blend `α=0.70` toward new vector, `fine_hits += 0.5` |
| Tier 3 — New Insert / LRQ Eviction (`sim<0.82`) | `mrm_v2.rs:210-234` | fills empty slots first; once full, evicts `argmin(fine_hits*2.0 + fine_salience)` |
| Coarse EMA centroid update | `mrm_v2.rs:236-256` | every write nudges the *nearest* centroid by `γ=0.95`, regardless of tier |
| Read (sharp cosine softmax) | `mrm_v2.rs:261-312` | `tau=0.05`; only `num_occupied_slots` fine slots are scored (`read_memory_const:269`) |
| Forward over sequence | `mrm_v2.rs:315-363` | order per token: project → **read** (pre-write state) → gate-residual → **write** |
| Analytical backward | `mrm_v2.rs:366-474` | "re-simulates" forward per step via `self.read_memory_const` — **see Finding 1** |
| Needle-in-haystack probe | `mrm_v2.rs:477-515` | isolated `MultiResMemoryV2` instance, not the trained model's stage memory |
| Attachment point in model | `tessera_model.rs:322-323, 436-438, 565-567` | only the **last** stage gets an `Option<MultiResMemoryV2>` |
| Optimizer coverage | `tessera_trainer.rs:187-208` (moments), `354-364` (update) | AdamW updates `w_q/w_k/w_v/w_o/w_gate` only — **never** touches `fine_keys/fine_vals/coarse_centroids/fine_salience/fine_hits/num_occupied_slots` (correct — those are non-parametric state, not learned weights) |
| Empirical harness | `exp_tessera_0.rs` | Arms A (dense control), B (trunk, no MRM), C (trunk+MRM), D (30-trial needle recall); kill criteria K1 (quality parity ≤+0.10 BPC), K2 (causal MRM gain ≥+0.08 BPC), K3 (needle recall ≥75%), K5 (≥50x DRAM byte reduction) |

**Distinct from Meridian.** `tessera_meridian_engine.rs`'s `InbuiltMeridianMemory` is an
HNSW-based *long-term* vector store with its own `NeuralMemoryGate`, wired in as `use_meridian`
at the *model* level, separate from MRM's per-stage working memory. Do not conflate the two in
any research write-up — they solve different problems (episodic working memory vs. persistent
retrieval index) and have completely different code paths.

---

## 3. Deep Findings (Ranked by Severity / Peak-Limiting Impact)

These were found by tracing **actual execution**, not by reading docstrings. Several directly
contradict their own inline comments — that mismatch is itself the finding.

### F1 — 🔴 CRITICAL: The analytical backward pass reads *post-sequence* memory state, not the historical per-timestep state (likely gradient-correctness bug)

`backward_sequence(&self, ...)` (`mrm_v2.rs:366`) is called **after** `forward_sequence` has
already run to completion and mutated `self` via `write_token` at every one of its `seq_len`
steps (see call site in `tessera_model.rs:1333-1341`, using `stage_h_pre_mrm` captured
*before* the stage's MRM call — the memory object itself is not snapshotted per-t). Inside
`backward_sequence`'s loop (`mrm_v2.rs:392-473`), the comment at line 397 says
`// Re-simulate forward at step t`, and it does recompute `q` and call
`self.read_memory_const(&q, ...)` — but `self` at that point is the **final**, fully-written
memory object from the end of the whole forward pass, not a checkpoint of memory-as-it-existed
just before token `t` wrote. Consequently:

- `self.fine_keys`/`self.coarse_centroids` used to score token `t`'s backward pass may already
  contain **Tier-1/Tier-2 overwrites from tokens `t+1..seq_len-1`** that occurred *after* `t`'s
  original forward read.
- `self.num_occupied_slots` used as `k_fine` in the backward loop (`mrm_v2.rs:422`) is the
  **final** occupancy count for every `t`, not the occupancy that existed at time `t`.
- This is not a corner case: with byte-level LM training (`CharDataset`, high-frequency bytes
  like `' '`, `'e'`, `'t'`), Tier-1 hard-overwrites of the *same* semantic slot happen
  repeatedly within any 64-token training window (`seq_len=64` in `exp_tessera_0.rs`), so the
  mismatch between "state at write-time" and "state at backward-recompute-time" is the common
  case, not the exception.
- **Net effect**: gradients flowing into `grad_wq` (and transitively into `d_in`, i.e. the
  residual stream gradient feeding the rest of the model) are computed against the wrong
  memory contents for most tokens in the sequence. This does not necessarily prevent the loss
  from decreasing (it's still a locally-consistent, differentiable-ish function of *something*),
  but it means the "MRM contribution" measured by kill-criterion K2 in `exp_tessera_0.rs` may
  be training against a systematically biased gradient signal — a plausible root cause if K2
  (`bpc_b - bpc_c ≥ 0.08`) is ever observed to fail or be noisy across seeds.

### F2 — 🟠 `fine_hits` is mislabeled and does not do what its own comment says

The struct comment at `mrm_v2.rs:91` calls it a **"query hit decay counter"**. In the actual
code:
- It is **never decremented/decayed** anywhere in the file — only `.min(50.0)`-clamped, so it
  is monotonically non-decreasing until capped. "Decay counter" is factually wrong.
- It is **never touched by `read_memory`/`read_memory_const`** — i.e. it does not track
  *queries* at all. It is incremented only inside `write_token`'s Tier 1 (+1.0) and Tier 2
  (+0.5) branches (`mrm_v2.rs:197, 209`) — i.e. it tracks **write-time deduplication matches**,
  not retrieval usage.
- Consequence: a slot that is the single most *retrieved* memory (highest softmax weight in
  every read for many tokens) but happens not to be write-matched again will have the exact
  same `fine_hits` as a slot that was never retrieved at all. The eviction utility function
  (`fine_hits*2.0 + fine_salience`, `mrm_v2.rs:221`) is therefore **decoupled from actual
  retrieval value** — the one signal you'd most want an LRU/LFU-style eviction policy to use.

### F3 — 🟠 Eviction "salience" is raw, untrained key-norm — not a learned importance signal

In production (`forward_sequence`, `mrm_v2.rs:360`): `let salience = dot(&k,&k).sqrt();` — the
L2 norm of the projected key vector. This is **not** a learned scalar (there is no salience
head with its own weights and gradient), so eviction protection is effectively "protect
whichever tokens happen to produce large-magnitude key projections," which is a function of
`w_k`'s current scale/direction, not of semantic importance. Compare this to the synthetic
probe (`probe_needle_recall_with_salience`, `mrm_v2.rs:482-515`), which **hand-assigns**
`needle_salience=100.0` vs `distractor_salience=1.0` — a 100x gap that has no analog in real
forward passes, where all salience values come from the same `‖w_k x_t‖` distribution and will
be far more homogeneous. **This means kill-criterion K3 (needle recall ≥75%) in
`exp_tessera_0.rs` is measuring a best-case, artificially-separable scenario that likely
overstates what the trained model's real eviction policy can guarantee in deployment.**

### F4 — 🟠 The LRQ eviction branch is architecturally almost never exercised during standard training

`nano_default()` sets `k_fine_slots=128` (`tessera_model.rs:159`); `exp_tessera_0.rs` trains
with `seq_len=64`. Since eviction (`mrm_v2.rs:212-228`, the `else` branch once
`num_occupied_slots >= k_fine`) only triggers once all 128 fine slots are filled, and a single
training sequence only ever performs ≤64 writes, **the eviction code path is dead during
`train_tessera`'s inner loop for the default config** — it is only reachable in the isolated,
never-integrated-into-training `probe_needle_recall` call (context_len=1024 distractors,
`exp_tessera_0.rs`'s Arm D, run as a *separate* freshly-constructed `MultiResMemoryV2`, not the
one trained in Arm C). **The eviction utility weights (implicitly, `w_k`'s influence on
`fine_salience`, and the fixed `2.0`/`1.0` mixing coefficients) receive zero gradient signal
from real training data.** If eviction quality matters for "peak" performance, it is currently
untested by gradient descent and untested by any benchmark that runs on the *same* memory
instance that was trained.

### F5 — 🟡 MRM's learned memory *content* never persists across training steps — only the projection weights do

`train_tessera` (`tessera_trainer.rs:296-311`) does `let model_ref = model.clone();` then, per
batch item, `let mut local_model = model_ref.clone();` and calls
`local_model.forward_backward_sequence(...)`. Each `local_model` (and thus its
`stage.mrm.fine_keys/fine_vals/coarse_centroids/...`) is **discarded after the batch item**;
only the *gradients* are extracted and later applied via `optimizer.step()`, which updates
`stage.mrm.w_q/w_k/w_v/w_o/w_gate` (`tessera_trainer.rs:354-364`) but **not** the memory
content arrays (correct, since those aren't learned parameters — but the implication is
important). The master `model`'s own `stage.mrm` is therefore never written to during training
at all; it stays at all-zero/`num_occupied_slots=0` for the entire run. This means:
- Every training example's MRM starts **empty** and builds up from scratch within ≤64 tokens.
- The projection weights are trained exclusively for the "cold start, low-occupancy" regime —
  they never see gradient signal from a memory that is warm, saturated, or has undergone
  eviction, drift-merge cascades, or coarse-centroid overflow, because within any 64-step
  window the fine store is at most half full (64/128) and Tier-3 eviction never fires (F4).
- If the intended deployment/inference regime involves long contexts (the very "1K-context
  needle" scenario the model is being benchmarked against in Arm D), the trained weights are
  optimized for a regime (short, unsaturated) that does not match the regime they're evaluated
  in (long, saturated, with eviction). This is a train/eval distribution mismatch, and it is
  the most actionable "why isn't K2/K3 higher" lever available.

### F6 — 🟡 Parameter-count reporting is decoupled from actual memory footprint

`param_count()` (`mrm_v2.rs:159-161`) returns `4*(d*d) + d` — a constant with respect to
`k_fine`/`k_coarse`. Meanwhile `memory_footprint_bytes()` (`mrm_v2.rs:163-165`) *does* scale
with `k_fine`/`k_coarse`, and `parameter_metrics()`'s separate `dram_bytes_per_token`
(`tessera_model.rs:621-623`) also scales with them. So in the Arm A/B/C comparison table
printed by `exp_tessera_0.rs`, the **"Total P" column** is invariant to how large you make the
memory (you could set `k_fine=100,000` and "Total P" wouldn't move), while the **DRAM
bytes/token column** would explode. Any "scaling law" style argument built from the "Total P"
column alone would silently ignore memory capacity — a methodological trap for future
experiments that try to report "params vs. quality" curves for MRM configurations.

### F7 — 🟡 Backward gradients into `w_k`/`w_v` are explicit heuristic surrogates, not true derivatives

The comment at `mrm_v2.rs:470` literally says **"Key and Value smooth alignment"**, and the
code (`mrm_v2.rs:471-472`) computes `outer_product_accumulate(&d_q, x_t, 0.05, &mut gwk)` and
`outer_product_accumulate(&d_ctx, x_t, 0.05, &mut gwv)` — a fixed `0.05` scale applied to
proxies (`d_q`, `d_ctx`) rather than a derivative through the discrete, branching `write_token`
(which involves an `argmax` over similarities and hard tier selection — genuinely
non-differentiable). This is a defensible design choice (true gradients through discrete
memory addressing require REINFORCE/Gumbel-softmax/straight-through machinery this file does
not implement), but it should be treated as an **explicit, tunable hyperparameter** (currently
hardcoded `0.05`) and documented as a surrogate, not silently trusted as exact.

### F8 — 🟢 Correct: causal ordering of read-before-write is right

`forward_sequence` computes `q,k,v`, calls `self.read_memory` (state from tokens `0..t-1`
only), *then* calls `self.write_token(&k,&v,...)`. Token `t` cannot see its own key/value in
memory during its own read. This is correctly implemented and worth preserving in any
refactor.

### F9 — 🟢 `new_hardware_adaptive`/`compute_optimal_slots` is dead code

`mrm_v2.rs:139-157` defines a cache-budget-aware sizing function (80/20 fine/coarse split
targeting a byte budget), but `grep`-ing the entire `crates/` tree shows it is **never called**
from `tessera_model.rs`, `tessera_trainer.rs`, or any binary. `TesseraConfig` still hardcodes
`k_fine_slots=128, k_coarse_slots=16` (`tessera_model.rs:159-160`). This is a ready-to-use lever
for hardware-aware peak-tuning (e.g. sizing MRM to fill L2/L3 on this sandbox's Cascade-Lake
core) that is currently unused.

### F10 — 🟢 Only the last stage gets MRM — an unexplored design axis

`config.use_mrm_v2 && p == n_stages - 1` (`tessera_model.rs:436-437, 565`) is a hard,
single-attachment-point design. There's no config knob to attach MRM to more than one stage,
or to a different stage index. Whether "peak" MRM benefit is maximized at the last stage vs.
an earlier/middle stage (or multiple stages with shared vs. independent memory) is an open,
cheaply-testable question given the existing `TesseraConfig` plumbing.

### F11 — 🟢 Sharp-softmax temperature (`tau=0.05`) and Tier thresholds (`0.95`/`0.82`) are global constants, untuned per-`d_model`

`tau=0.05` appears identically in `read_memory_const` (`mrm_v2.rs:272`) and in
`backward_sequence` (`mrm_v2.rs:375`); Tier thresholds `0.95`/`0.82` are hardcoded literals at
`mrm_v2.rs:191, 198`. Cosine-similarity distributions are known to concentrate differently as
dimensionality `d` changes (higher `d` → cosine similarities of random vectors cluster tighter
around 0). Since `d_model` varies across configs (`nano_default`=128 vs. the abandoned
48M-param probe's `d_model=672`), a fixed `0.82`/`0.95` threshold pair tuned for `d=128` is not
guaranteed to preserve the same *effective* selectivity at `d=672`. No dimension-aware
recalibration exists.

---

## 4. Research Agenda: Turning Findings into Falsifiable Experiments

Following this repo's own methodology (pre-registered kill criteria, same-run controls), each
item below is written as **hypothesis → experiment → kill criterion**, directly extending
`exp_tessera_0.rs`'s Arms A–D.

| # | Hypothesis | Experiment | Kill Criterion (pre-register before running) |
|---|---|---|---|
| E1 (fixes F1) | The backward pass's stale-memory-state bug measurably corrupts the gradient signal into `w_q`/`w_k` | Snapshot `fine_keys/fine_vals/coarse_centroids/num_occupied_slots` at every timestep during forward (cheap: `seq_len × k_fine × d` floats, or store only the *delta* per step); rewrite `backward_sequence` to index into the per-t snapshot instead of `self`'s final state. Re-run Arm C vs. a "buggy-backward" control on identical seeds. | If corrected-backward Arm C's `bpc_c` improves by **≥0.02 BPC** over buggy-backward Arm C (same steps/seed), the bug is real and worth permanently fixing. If no material difference, F1 is theoretically real but empirically inert at current `seq_len=64` — deprioritize. |
| E2 (fixes F4, F5) | Training with `seq_len > k_fine` (forcing real eviction + a warm/saturated memory during gradient descent) improves K2/K3 vs. the current short-`seq_len` regime | Run Arm C with `seq_len=256` and `k_fine=64` (forcing saturation partway through every training sequence) vs. current `seq_len=64,k_fine=128` (no saturation). Compare K2 (causal MRM gain) and a K3-style probe run *on the actually-trained model's own stage MRM*, not an isolated fresh instance. | K2 improves by ≥0.03 BPC under the saturating regime → confirms F4/F5's train/eval mismatch is real and fixable by curriculum. No improvement → the mismatch may not matter at this scale; revisit at larger `d_model`. |
| E3 (fixes F2, F3) | Replacing `fine_hits` (write-dedup counter) with a true retrieval-frequency counter incremented inside `read_memory`, and/or replacing raw-norm `salience` with a small learned scalar head, improves eviction quality | Add a `retrieval_hits: Vec<f32>` incremented by softmax weight `probs[i]` inside `read_memory`; use `retrieval_hits*w1 + fine_salience*w2` (learned or grid-searched `w1,w2`) as the eviction utility. Re-run Arm D's needle probe but **with realistic (non-100x-inflated) salience** for both needle and distractors, executed against the model's own post-training MRM instance. | If needle recall (realistic salience) rises from a currently-measured baseline by ≥15 percentage points, retrieval-aware eviction is a validated peak-lever. If flat, the eviction policy isn't the bottleneck — look at F1/F5 instead. |
| E4 (fixes F6, F9) | Wiring `compute_optimal_slots`/`new_hardware_adaptive` into `TesseraConfig` to size `k_fine/k_coarse` from a byte budget (e.g. this sandbox's measured L2/L3) yields better BPC-per-DRAM-byte than the hardcoded 128/16 default | Sweep `buffer_mb ∈ {1, 4, 16, 32}` via `compute_optimal_slots(d_model, buffer_mb)`, train Arm C variants, plot `val_bpc` vs. `memory_footprint_bytes()` (not "Total P", per F6) to get a real capacity/quality Pareto curve. | A monotonic (or near-monotonic) BPC improvement with buffer size, with diminishing returns identifiable → gives a principled default recommendation. Non-monotonic/no relationship → capacity isn't the limiting factor; look at F1/F7 (gradient fidelity) instead. |
| E5 (fixes F10) | Attaching MRM to an earlier or middle stage (or two stages) instead of only the last yields a better K2 gain per unit of added params/compute | Add a `mrm_stage_indices: Vec<usize>` config field (generalizing the current `bool`), test `{last}`, `{first}`, `{middle}`, `{first,last}` on identical param/compute budgets (reduce `k_fine` per stage if multiple stages have MRM, to hold `memory_footprint_bytes()` roughly constant). | Any variant beats `{last}`-only by ≥0.02 BPC at equal memory footprint → informs the "peak" default attachment point. |
| E6 (fixes F7) | A softer, differentiable write-addressing scheme (e.g., soft top-k blend instead of hard argmax `best_sim_slot`, or a straight-through estimator) improves gradient quality enough to matter for K2 | Implement a soft-write variant behind a feature flag: instead of writing to the single argmax slot, blend the write across the top-2 slots weighted by softmax(sim/τ_write); keep hard eviction logic unchanged for slot *selection* under overflow. Compare against current hard-argmax on identical seeds/steps. | ≥0.02 BPC improvement on Arm C validates investing in differentiable addressing; no change → F1 (temporal bug) is likely the dominant gradient-fidelity issue, not the discreteness of addressing itself — fix F1 first. |
| E7 (fixes F11) | Tier thresholds (`0.95`/`0.82`) and softmax `tau=0.05` need to scale with `d_model` to preserve equivalent selectivity | Empirically measure the distribution of cosine similarities between random `d`-dimensional unit vectors for `d ∈ {128, 256, 672}` (closed form: concentrates like `N(0, 1/d)`), derive `d`-dependent thresholds (e.g. threshold = `k · 1/sqrt(d)` calibrated to preserve a target selectivity), and re-run Arm C at `d_model=672` (the previously-selected 48M-param config) with recalibrated vs. fixed thresholds. | Recalibrated thresholds change which fraction of writes land in Tier1/2/3 by a large margin (e.g. >20 percentage-point shift in Tier distribution) at `d=672` vs. `d=128` → thresholds must be made `d`-aware before scaling up. Small shift → current fixed constants are fine to keep as-is when scaling. |

**Suggested order of attack** (highest confidence / lowest cost first): **E1 → E5 → E4 → E2 →
E3 → E7 → E6.** E1 is a pure correctness fix with no architecture change and should be done
regardless of whether it moves benchmark numbers (it's simply wrong as written). E5 and E4 are
cheap config sweeps using code that already exists (`compute_optimal_slots`) or trivial to add.
E2/E3 require a bit more plumbing (curriculum, new hit counters). E6/E7 are the deepest,
highest-effort, most research-y changes — save for last once the "free" bugs are fixed and the
low-hanging config sweeps are exhausted, so that any measured gain can be attributed correctly.

---

## 5. Ready-to-Paste Prompt Block (for handing to another AI session)

```
You are working in a Rust workspace at crates/tessera-core/src/mrm_v2.rs implementing
"MRM-v2" (Multi-Resolution Memory), a per-token working-memory module for the TESSERA
language model. Read mrm_v2.rs in full (516 lines), plus its call sites in
tessera_model.rs (search "mrm") and tessera_trainer.rs (search "mrm"), plus the
falsification harness in exp_tessera_0.rs.

Your task: push MRM-v2 toward peak retrieval/quality performance while preserving its
CPU-first, zero-GPU, low-RAM design constraints (this sandbox: 2 vCPU, ~1.9GB RAM, no
GPU, AVX-512 available). Work through these verified findings in priority order,
treating each as a falsifiable hypothesis with a pre-registered kill criterion before
you touch code:

1. [CORRECTNESS BUG] backward_sequence() re-simulates forward per-timestep by calling
   self.read_memory_const() against `self` AFTER forward_sequence has already fully
   mutated `self` via write_token() for every t. This means backward for token t is
   scored against the FINAL post-sequence memory state, not the state that existed at
   time t. Verify this by instrumenting/snapshotting memory state per-t during forward,
   then compare against what backward currently reads. Fix by threading a per-t
   snapshot (or per-t diff) into backward_sequence instead of relying on `self`.
   Kill criterion: does fixing this change val_bpc by >=0.02 on Arm C (exp_tessera_0.rs)
   at matched seed/steps? If yes, it's a real bug worth permanently fixing. If no,
   deprioritize but still fix for correctness.

2. [ARCHITECTURE] fine_hits is documented as a "query hit decay counter" (mrm_v2.rs:91)
   but is (a) never decayed, only clamped, and (b) only incremented by write_token's
   Tier1/Tier2 dedup matches, never by read_memory. It is therefore decoupled from
   actual retrieval usage, which is what an eviction policy should protect. Propose and
   test a retrieval-weighted hit counter incremented inside read_memory by softmax
   attention weight, used alongside (or replacing) the write-dedup counter in the
   Tier-3 eviction utility function (currently `fine_hits*2.0 + fine_salience`).

3. [TRAIN/EVAL MISMATCH] Because train_tessera clones the whole model per batch item
   and discards each clone after extracting gradients, and because seq_len=64 <
   k_fine=128 in the default config, MRM's eviction branch (Tier 3 overflow case) NEVER
   fires during training — it is only exercised in the isolated probe_needle_recall
   call, on a freshly-constructed memory instance never seen during training. Propose
   a training curriculum where seq_len exceeds k_fine (forcing real saturation and
   eviction during gradient descent) and measure whether K2 (causal MRM BPC gain,
   threshold >=0.08 in exp_tessera_0.rs) improves.

4. [BENCHMARK VALIDITY] probe_needle_recall_with_salience hand-assigns needle_salience
   =100.0 vs distractor salience=1.0, a 100x gap that never occurs in real forward
   passes (where salience = ||w_k @ x_t||, an untrained raw norm with a much narrower,
   uncontrolled distribution). Re-run the K3 needle-recall kill criterion (threshold
   >=75%) using realistic, non-inflated salience values sampled from the actual
   trained model's own key-projection distribution, and report whether the 75%
   threshold still passes.

5. [DEAD CODE / HARDWARE AWARENESS] compute_optimal_slots()/new_hardware_adaptive() in
   mrm_v2.rs are implemented but never called anywhere in the codebase. TesseraConfig
   still hardcodes k_fine_slots=128, k_coarse_slots=16. Wire the hardware-adaptive
   sizing into TesseraConfig behind a buffer_mb knob, sweep buffer_mb against this
   sandbox's real L2/L3 sizes, and plot val_bpc vs memory_footprint_bytes() (NOT the
   param_count(), which is fixed at 4*d*d+d regardless of k_fine/k_coarse — see finding
   F6) to find the actual capacity/quality Pareto frontier.

For every change, follow this repo's existing falsification discipline: state the
hypothesis, pre-register a numeric kill criterion BEFORE running, use the same-run
Arm A/B/C/D comparative structure in exp_tessera_0.rs, and report GREEN/YELLOW/RED
verdicts exactly as that file does. Do not claim an improvement without a same-seed,
same-step-count control run.
```

---

## 6. What This Document Deliberately Does *Not* Cover

- The JS-side `MeridianEngine` in `tessera-loader-app/` (BM25 + dense hybrid) — a different
  system in a different language, unrelated to `mrm_v2.rs`.
- AXIOM/STRATUM/MNEME/FORGE — separate architecture lineages in this repo with their own
  falsification history (`NOTES.md`); not memory-relevant to MRM-v2 specifically.
- Actually running training — per your explicit instruction, no training was (re-)started
  while producing this document. Every finding above was obtained by static source reading
  and cross-referencing call sites, not by executing new experiments. Sections 3–4 are
  designed so you (or another AI session) can decide which experiments are worth the compute
  before running anything.
