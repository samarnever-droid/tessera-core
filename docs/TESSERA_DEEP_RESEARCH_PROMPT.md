# TESSERA-Q — Deep Research Prompt (paste into AI Deep Research mode)

```
I am the sole developer/researcher on "TESSERA-Q", a from-scratch Rust
implementation of a hybrid recurrent+attention language model architecture,
running CPU-only (no GPU, 2 vCPU, ~1.9GB RAM, AVX-512 available) as a
falsification-driven research project (pre-registered kill criteria, same-run
A/B/C/D arm comparisons — similar in spirit to how DeepMind/Meta run ablation
studies before committing to an architecture at scale).

I need a DEEP RESEARCH report, not a shallow summary. Prioritize primary
sources (arXiv papers, official model tech reports, source code of open
implementations) over blog posts. For every claim, cite the specific paper/
model/commit it comes from. Where evidence conflicts across papers, say so
explicitly rather than picking one side. Where something is unresolved in the
literature, say "this is an open question" rather than guessing with false
confidence.

============================================================
CURRENT ARCHITECTURE (ground truth — do not need to re-derive this)
============================================================
Per "progressive hierarchy stage" (P=3 stages by default, d_model=128,
d_ff=768, n_heads=4 giving 2 differential-attention head-pairs,
adapter_rank=8):

1. Affine RMSNorm (gamma-scaled, no bias) pre-attention norm.
2. Depthwise causal 1D convolution (kernel width 4) applied to the normed
   stream before attention projections — a "local mixing" step before global
   attention.
3. QK-Norm: RMSNorm applied per-head to Q and K vectors before the dot
   product (no learnable affine on this norm, standard QK-Norm as in
   Gemma2/many 2024+ LLMs).
4. Adaptive RoPE "banding": instead of one fixed RoPE frequency schedule
   across all dims, there is a learned per-frequency-band scalar multiplier
   eta (length d_k/2) applied to rotation angles — intended to let the model
   learn which rotational frequencies matter most, rather than using the
   fixed geometric schedule from the original RoPE paper.
5. Differential Attention (DiffAttn, following Microsoft's "Differential
   Transformer", arXiv 2410.05258): heads are processed in pairs; each pair
   computes two independent softmax attention distributions (probs1, probs2)
   over the SAME keys/values but with different Q/K sub-projections, and the
   final attention weight is probs1 - lambda_eff * probs2, where
   lambda_eff = exp(lambda1) - exp(lambda2) + 0.8 (clamped >= 0), initialized
   with lambda1=lambda2=0.8. This is meant to cancel attention noise/
   "attention sink" artifacts by having the second softmax subtract out a
   learned baseline.
6. Output projection via a dense d x d matrix (no separate "Gated Attention
   Unit" multiplicative gate is currently applied to the attention output
   itself in the main forward path — there IS a separate learned d x d
   "Gated Temporal Unit" weight matrix (w_gate_attn) in the parameter struct,
   but I need you to help me figure out from the GAU/Gated Attention Unit
   literature whether my current wiring under-uses this gate compared to
   how papers like "Transformer Quality in Linear Time" (GAU, arXiv
   2202.10447) or newer gated-attention papers (2024-2025) actually use it).
7. Residual add, second Affine RMSNorm, then a SwiGLU-style FFN (gate proj
   w1, up proj w1u, down proj w2, d_ff = 6x d_model expansion) PLUS a
   parallel low-rank "adapter" branch (down-project d->r via adapter_v,
   up-project r->d via adapter_u, r=8) added into the same residual stream —
   my own design, inspired loosely by LoRA-style adapters but applied as a
   permanent parallel path rather than a fine-tuning-only add-on.
8. Value Residual connections (per "Value Residual Learning" / ResFormer-
   style ideas circulating in 2024): I referenced this in my design intent
   but need you to help verify what the ACTUAL cited mechanism is (does
   ResFormer add the previous layer's V directly to the current layer's V
   before attention, or does it mix V across layers with a learned gate?)
   and whether my implementation is a faithful version of it.
9. Only the LAST stage additionally has "MRM-v2" (Multi-Resolution Memory): a
   per-token working-memory module with dual-resolution storage — 128 exact
   "fine" key/value slots plus 16 EMA-decayed "coarse" centroid summaries —
   using a 3-tier write policy (hard overwrite for cosine sim >= 0.95, soft
   EMA merge for 0.82-0.95, new-insert-or-evict below 0.82) and a sharp
   cosine-softmax read (temperature 0.05). This is NOT the same as the
   model's separate long-term HNSW-based vector memory ("Meridian").
10. Model-level (not per-stage): Dual-Head Multi-Token Prediction (MTP,
    following DeepSeek-V3's MTP module, arXiv 2412.19437) — predicts the
    next TWO tokens per position instead of one, tied embedding/output
    weights, a Z-loss auxiliary term for logit-magnitude regularization
    (following PaLM/Chinchilla-lineage stabilization tricks), and a
    Warmup-Stable-Decay (WSD) learning-rate schedule (following MiniCPM's
    WSD paper, arXiv 2404.06395) for the AdamW optimizer.
11. Optional separate long-term memory ("Meridian"): an HNSW-based
    approximate-nearest-neighbor vector index with a differentiable
    "NeuralMemoryGate" (identity-initialized projection + sigmoid-gated
    residual fusion of retrieved memory into the hidden state).

Empirically, I validate this against a dense Transformer baseline using
same-run controlled arms (Arm A = dense Transformer control, Arm B = my
trunk without MRM, Arm C = trunk + MRM, Arm D = isolated needle-in-haystack
recall probe) with PRE-REGISTERED kill criteria: (K1) quality parity vs
dense control within +0.10 BPC, (K2) MRM must causally contribute >=0.08 BPC
over the no-MRM trunk, (K3) needle recall >=75% at 1024-token distractor
context, (K5) >=50x DRAM-bytes-per-token reduction vs dense attention's KV
cache. I already separately benchmark MRM-v2 against a Griffin-style
RG-LRU recurrent memory baseline and a plain FIFO buffer baseline in an
adversarial suite.

============================================================
WHAT I ALREADY SUSPECT ARE WEAKNESSES (verified by reading my own source,
not guessed) — please research literature-backed fixes/alternatives for
these SPECIFICALLY, not generic "improve your transformer" advice:
============================================================

W1. My MRM-v2's analytical backward pass re-derives per-token gradients by
    re-reading memory state AFTER the full forward pass over the sequence
    has already finished writing to that same memory object — meaning early
    tokens' backward pass sees a memory state contaminated by later tokens'
    writes, not the state that existed when that token actually ran. This
    smells like the same class of problem that motivated truncated
    backpropagation-through-time (BPTT) and gradient checkpointing
    strategies in RNN/SSM literature, and also resembles issues that
    differentiable-memory papers (e.g. Neural Turing Machines, Differentiable
    Neural Computers, Memorizing Transformers arXiv 2203.08913,
    Larimar/other "memory as parameter" 2024 papers) had to solve explicitly.
    RESEARCH QUESTION: What are the standard, established techniques from
    the differentiable-memory / memory-augmented-neural-network literature
    (NTM, DNC, Memorizing Transformers, Recurrent Memory Transformer,
    Infini-attention (arXiv 2404.07143), Titans (Google, 2024/2025)) for
    correctly backpropagating through a memory store that is mutated
    causally at every timestep within the same forward pass? Is
    "snapshot-per-timestep" the standard answer, or do these papers use
    stop-gradient tricks / detach the memory read from the memory write
    entirely (treating writes as non-differentiable state updates, only
    backpropagating through reads) to avoid this problem altogether? Which
    approach is cheaper and how do the major implementations (Infini-
    attention, Titans especially, since both explicitly deal with per-
    segment/per-token memory updates during training) handle this?

W2. My eviction policy for the 128-slot "fine" memory uses a hand-tuned
    heuristic utility score (retrieval/write-hit count * 2.0 + raw key-norm
    "salience") with NO learned/gradient-trained component deciding what to
    evict, and separately, I've found the "salience" signal used in my own
    published benchmark's needle-in-haystack probe uses an artificially
    100x-inflated salience value for the needle vs. distractors that does
    not match the salience distribution actually produced during real
    training (raw key-projection L2 norms are much more homogeneous).
    RESEARCH QUESTION: In fixed-slot / fixed-capacity memory architectures
    (Griffin's RG-LRU, Infini-attention's linear-attention memory, Titans'
    "surprise"-gated memory update rule, DNC's usage-weighted allocation,
    product-key memory / PKM papers), what mechanisms are used to decide
    WHAT to evict/overwrite, and are any of them LEARNED (trained via
    gradient descent to predict eviction-worthiness) rather than hand-tuned
    heuristics? Specifically look at Titans' "surprise metric" (gradient-
    based per-token importance) since it seems most directly transferable
    to my hand-rolled "salience" concept, and tell me exactly how it's
    computed and whether it requires a second backward pass or can be
    derived from the same forward-pass activations.

W3. My model only attaches the working-memory module (MRM-v2) to the LAST
    of 3 progressive stages, and there's no principled justification for
    this choice beyond "seemed reasonable" — I have not tested attaching it
    to an earlier or middle stage, or to multiple stages simultaneously.
    RESEARCH QUESTION: Across memory-augmented architectures that stack
    multiple layers/blocks (Infini-attention, Titans' MAC/MAG/MAL variants,
    Block-Recurrent Transformers, Recurrent Memory Transformer), is there
    empirical or theoretical guidance on WHICH layer(s) benefit most from
    attached persistent/working memory — early layers (closer to raw
    input, more "surface" features), late layers (closer to output, more
    "semantic" features), or a distributed placement across many layers with
    smaller memory each? Cite specific ablations if papers report them.

W4. My Differential Attention lambda parameters (lambda1, lambda2, both
    initialized to 0.8, giving lambda_eff = exp(0.8)-exp(0.8)+0.8 = 0.8 at
    init) are per-STAGE scalars shared across all head-pairs within that
    stage, whereas I recall the original Differential Transformer paper
    (Microsoft, arXiv 2410.05258) may use a more granular per-layer-and-
    per-head-pair (or even depth-dependent) lambda initialization schedule.
    RESEARCH QUESTION: What EXACTLY is the lambda initialization and
    parameterization scheme in the original Differential Transformer paper
    (is lambda_init depth-dependent, e.g. lambda_init = 0.8 - 0.6 *
    exp(-0.3*layer_idx) as I vaguely recall, or a fixed constant)? Does the
    paper's ablation show sensitivity to this choice? Also: does the paper's
    reported quality/robustness gain (reduced attention noise, better long-
    context retrieval, resistance to "attention sink") replicate at SMALL
    scale (~1-50M parameters, the scale I operate at), or is DiffAttn's
    benefit primarily reported at >1B parameter scale where it may not be
    representative of my regime?

W5. My "Value Residual" mechanism (inspired loosely by ResFormer-style
    ideas) — I am not fully certain I've implemented the mechanism the
    literature actually describes.
    RESEARCH QUESTION: Precisely describe the Value Residual Learning
    mechanism (find the exact paper — I believe circa 2024, possibly titled
    something like "Value Residual Learning for Alleviating Attention
    Concentration" or similar, or it may be part of ResFormer). Does it (a)
    add the PREVIOUS LAYER's value vectors directly into the CURRENT layer's
    value vectors before the attention weighted-sum, (b) use a learned gate
    to mix current vs. previous-layer values, or (c) something else
    entirely? What problem does it solve (attention entropy collapse in
    deep networks?) and is there an ablation showing the effect size at
    shallow depth (3-6 layers, my regime) vs. deep networks (24+ layers,
    where most such papers benchmark)?

W6. I use a fixed sharp-softmax temperature (tau=0.05) for MRM-v2's memory
    read, and fixed Tier thresholds (cosine sim >= 0.95 / >= 0.82) for its
    write-merge policy, both hardcoded regardless of d_model. Cosine
    similarity between random vectors concentrates more tightly around 0 as
    dimensionality increases (concentration of measure), so a threshold
    tuned for d_model=128 may not preserve the same "selectivity" at
    d_model=672 or larger.
    RESEARCH QUESTION: In nearest-neighbor / vector-memory / retrieval
    literature (HNSW tuning guides, dense retrieval papers, hash-based
    memory papers), is there an established formula or heuristic for scaling
    similarity thresholds or softmax temperature as a function of embedding
    dimensionality to preserve constant "selectivity" or "discrimination
    power"? Is temperature scaling as 1/sqrt(d) (as in standard attention
    scaling) the right analogy here, or does memory-read softmax over a
    small fixed set of KEYS (as opposed to attention over all positions in
    a sequence) behave differently?

W7. My model reports "total parameters" via a formula that is CONSTANT with
    respect to the size of the MRM-v2 memory store (i.e., param_count()
    doesn't change if I make the memory store 10x bigger), while a separate
    "DRAM bytes per token" metric DOES scale with memory size. This means my
    own benchmark table's "Total Params" column is not comparable across
    memory-size configurations, which is a methodological trap if I ever
    try to plot a "quality vs. total capacity" scaling curve.
    RESEARCH QUESTION: In papers that report scaling laws or capacity/
    quality tradeoffs for memory-augmented models (Product-Key Memory
    papers, Memorizing Transformers, Titans, kNN-LM, RETRO), how do they
    define "model size" or "capacity" for the purposes of a scaling-law
    x-axis when part of the model is a non-parametric or semi-parametric
    memory store rather than learned weights? Is there a standard convention
    (e.g., report memory bytes and learned-parameter count as two SEPARATE
    axes rather than trying to unify them into one "total params" number)?

============================================================
BROADER OPEN QUESTIONS I WANT YOUR RESEARCH TO ANSWER
============================================================

Q1. Given a hybrid architecture that already combines: differential
    attention, adaptive RoPE banding, depthwise causal convolution
    (local token mixing), QK-norm, a per-stage low-rank adapter branch
    parallel to the FFN, multi-token prediction, AND a dual-resolution
    working-memory module — is there evidence in the literature (ablation
    studies from any of the source papers, or from architecture-search /
    "kitchen sink" combination papers) about WHICH of these components tend
    to have DIMINISHING or even NEGATIVE returns when stacked together
    (component interaction effects), versus which combine cleanly/
    additively? I want to know if I'm likely wasting parameters/compute on
    redundant mechanisms that solve overlapping problems (e.g., does
    DiffAttn's noise-cancellation make QK-Norm partially redundant, or vice
    versa? Does the depthwise causal conv make the adaptive RoPE banding
    less necessary because local mixing already captures what banding was
    trying to help with?).

Q2. At my actual operating scale (3-6 transformer-equivalent "stages",
    d_model 128-672, total params in the 1M-50M range, CPU-only training,
    no GPU, byte-level tokenization on small corpora like tiny_shakespeare/
    enwik8), which of the "frontier-scale" techniques I've borrowed
    (DeepSeek's MTP, Differential Attention, WSD schedule, Z-loss) have
    published or informally-reported evidence of actually helping at THIS
    small scale, versus techniques that are known/suspected to only pay off
    at >1B parameter scale and may be net-negative (added complexity/
    compute for no quality gain, or even quality LOSS due to e.g. MTP
    diluting gradient signal per auxiliary head when the trunk is already
    tiny) at my scale? Please be skeptical and specific — cite any small-
    scale ablations you can find, and flag clearly if a technique has ONLY
    ever been validated at large scale with no small-scale evidence either
    way.

Q3. My MRM-v2 module is deliberately positioned as a middle ground between
    (a) full quadratic-cost attention/KV-cache (unlimited exact recall, but
    O(n) memory growth) and (b) fixed-size recurrent state (Griffin RG-LRU,
    Mamba/SSM state, O(1) memory but lossy compression of all history into
    one fixed-size state). Survey the CURRENT (2024-2026) state of the art
    specifically for this middle-ground category: bounded-size but non-
    trivial (multi-slot, addressable) working memory for transformers/
    hybrids — Titans (Google), Infini-attention (Google), Memorizing
    Transformers, RETRO/RETRO++, kNN-LM, Recurrent Memory Transformer,
    HGRN2, product-key memory layers, and anything more recent you find.
    For EACH, tell me: (a) how they decide what to write into memory,
    (b) how they decide what to evict/overwrite when memory is full, (c)
    how they read/retrieve from memory (exact nearest-neighbor? soft
    attention? learned gating?), (d) whether they backprop through the
    memory write operation or treat writes as non-differentiable, and (e)
    what their reported needle-in-haystack / long-context recall numbers
    were and at what context length. I want a comparison table I can use to
    see exactly where my dual-tier (hard-overwrite / soft-merge / evict)
    design sits relative to the state of the art, and which specific paper's
    mechanism I should consider adopting or adapting next.

Q4. Is there recent (2024-2026) research on COMBINING multi-token
    prediction (predicting >1 future token per forward pass, as in
    DeepSeek-V3) with a persistent/working-memory module like mine? I'm
    specifically wondering whether predicting multiple future tokens
    changes what SHOULD be written into working memory at each step (should
    the memory write be keyed on the token that was actually generated, or
    could it usefully incorporate information about the multi-token
    prediction target as an additional training signal for what to
    memorize?). If no such combination has been published, say so plainly —
    don't invent a citation.

Q5. Given I am CPU-only with no GPU (2 vCPU, AVX-512 available, ~1.9GB RAM),
    which of the architectural components above have the WORST
    compute-to-quality ratio specifically on CPU (i.e., which are cheap on
    GPU due to parallel matrix-multiply hardware but disproportionately
    expensive on CPU due to poor SIMD utilization, cache behavior, or
    inherently sequential dependencies)? For example: does Differential
    Attention's doubled softmax computation matter more on CPU than the
    doubled FLOPs alone would suggest, due to cache/memory-bandwidth
    effects? Does MRM-v2's per-token sequential write (which cannot be
    trivially parallelized across timesteps within one sequence, unlike
    attention) become a bottleneck specifically at CPU clock speeds? Give me
    a concrete ranked list of "most CPU-expensive relative to quality gain"
    components if you can find or reason toward evidence for this.

============================================================
FORMAT OF YOUR FINAL ANSWER
============================================================
Structure your report as:
1. A short table: Component | What I claim it does | What the primary
   source actually says | Verdict (faithful / partially faithful / deviates
   — and how) — covering DiffAttn, MTP, WSD, Z-loss, Value Residual, and
   Adaptive RoPE banding specifically, since these are the ones I'm least
   certain I've implemented correctly per their source papers.
2. Direct, cited answers to W1-W7 above (my suspected weaknesses) — for each,
   tell me the state-of-the-art fix/mechanism from the literature and how
   feasible it would be to port into a small (1-50M param), CPU-only, from-
   scratch Rust codebase (flag anything that fundamentally requires GPU-
   scale parallelism or auto-diff frameworks I don't have).
3. Direct, cited answers to Q1-Q5 (my broader open questions).
4. A final prioritized list (ranked by expected quality-gain-per-engineering-
   effort, given my CPU-only/small-scale constraints) of the TOP 5 concrete
   changes I should make next, each with a one-line justification citing
   which source/finding above it's based on.

Be rigorous and skeptical. If a technique I'm using is oversold, is only
validated at scales far beyond mine, or is actively contraindicated for my
regime, tell me clearly rather than being diplomatically vague. I would
rather learn "component X is likely net-negative for you, here's why" than
receive generic praise.
```
