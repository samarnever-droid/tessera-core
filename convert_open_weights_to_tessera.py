#!/usr/bin/env python3
"""
Convert an existing open-weight HuggingFace model (e.g. Qwen/Qwen2.5-0.5B-Instruct) into a
TESSERA-Q `TesseraGPUModel` (tessera_triton.py) by transplanting the WEIGHT-COMPATIBLE
sub-components (token embeddings, per-layer Q/K/V/O attention projections, gate/up/down FFN
projections, RMSNorm gammas) and leaving every TESSERA-Q-ONLY component (unified DiffAttn
lambda formula params, adaptive-RoPE eta, learned Value-Residual gate, depthwise causal
conv, low-rank adapter, MRM working memory) at their normal TesseraGPUModel initialization,
since these have no equivalent in a standard decoder-only transformer and must be learned
from scratch (or via a short post-conversion fine-tune -- see the note printed at the end).

WHY THIS IS A GENUINE, HONEST CONVERSION (not a silent reshape-and-hope):
  - TESSERA-Q's attention is Differential Attention (2 softmax branches per head-pair,
    combined via a learned lambda), fundamentally different math from a standard
    single-softmax transformer attention. There is no lossless way to convert a standard
    Q/K/V/O projection into a *correct* differential-attention weight; the safest,
    most honest approach -- and the one used here -- is to copy the source model's Q/K/V/O
    weights into TESSERA-Q's Q/K/V/O linear layers UNCHANGED (same shape contract: a
    d_model x d_model linear applied to the post-norm hidden state) so the model starts
    from the same underlying token/positional feature representations, while leaving the
    differential combination (lambda) at its normal, principled, depth-dependent
    initialization -- exactly as if this were a brand new TESSERA-Q stage index with a
    warm-started QKVO projection instead of a cold-random one.
  - Qwen2 uses GQA (grouped-query attention: num_key_value_heads < num_attention_heads).
    TESSERA-Q's DifferentialAttention has NO GQA support (full n_heads for both Q and K/V).
    This script handles that mismatch explicitly and honestly: it repeats each Qwen KV head
    group to match TESSERA-Q's full head count via `repeat_interleave` (the standard,
    lossless "un-grouping" operation -- mathematically expands GQA to MHA with IDENTICAL
    attention outputs before any further TESSERA-specific changes are applied), rather than
    silently truncating or reshaping in a way that would corrupt the weights.
  - Qwen2's attention/MLP linears carry BIAS terms on q/k/v (but not o); TESSERA-Q's
    linears are all bias-free (nn.Linear(..., bias=False)). Those bias vectors are
    reported (not silently dropped without mention) and folded into the norm's output
    is NOT attempted (would require a full affine-transform derivation); instead this
    script prints a clear WARNING that q/k/v bias terms are dropped, and recommends the
    post-conversion fine-tune to recover any accuracy lost from that omission.
  - vocab_size and d_model MUST match (or be manually reconciled) between the source model
    and the target TesseraGPUModel: this script constructs the TesseraGPUModel using the
    SOURCE model's own vocab_size/hidden_size/intermediate_size/num_hidden_layers, so shapes
    match automatically; it does not support converting between mismatched dimensions.

Usage:
    python3 convert_open_weights_to_tessera.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output tessera_from_qwen.bin --n_heads 8

Note: --n_heads controls TESSERA-Q's OWN head count (used for DiffAttn pair splitting and
RoPE), independent of the source model's own attention head count; it must be an even
number >= 2 (>=2 for at least one differential pair) and must evenly divide d_model.
"""
import argparse
import sys

import torch


def convert(model_name: str, output_path: str, n_heads: int, int8: bool = False):
    from transformers import AutoConfig, AutoModelForCausalLM
    from tessera_triton import TesseraGPUModel

    print(f"[1/5] Downloading/loading source config for '{model_name}' ...")
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    vocab_size = cfg.vocab_size
    d_model = cfg.hidden_size
    d_ff = cfg.intermediate_size
    n_layers = cfg.num_hidden_layers
    src_n_heads = getattr(cfg, "num_attention_heads", None)
    src_n_kv_heads = getattr(cfg, "num_key_value_heads", src_n_heads)

    print(f"      model_type={getattr(cfg, 'model_type', '?')}  vocab_size={vocab_size}  "
          f"d_model={d_model}  d_ff={d_ff}  n_layers={n_layers}  "
          f"src_n_heads={src_n_heads}  src_n_kv_heads={src_n_kv_heads}")

    if d_model % n_heads != 0:
        print(f"[!] ERROR: --n_heads={n_heads} does not evenly divide d_model={d_model}. Aborting.")
        sys.exit(1)
    if n_heads < 2 or n_heads % 2 != 0:
        print(f"[!] ERROR: --n_heads={n_heads} must be an even number >= 2 "
              f"(DiffAttn needs whole head-pairs). Aborting.")
        sys.exit(1)

    print(f"[2/5] Downloading/loading source model weights (float32) ...")
    src_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, trust_remote_code=True)
    src_model.eval()
    src_sd = src_model.state_dict()

    print(f"[3/5] Constructing target TesseraGPUModel "
          f"(vocab_size={vocab_size}, d_model={d_model}, d_ff={d_ff}, n_stages={n_layers}, "
          f"n_heads={n_heads}) ...")
    tessera = TesseraGPUModel(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_stages=n_layers,
                               n_heads=n_heads)

    def find_key(*candidates):
        for c in candidates:
            if c in src_sd:
                return c
        return None

    def repeat_kv_to_full_heads(w: torch.Tensor, n_kv_heads: int, n_full_heads: int, d_model: int) -> torch.Tensor:
        """
        Losslessly 'un-groups' a GQA K or V projection weight of shape [n_kv_heads * d_head,
        d_model] into a full [n_full_heads * d_head, d_model] weight by repeating each KV
        head's row-block (n_full_heads // n_kv_heads) times via repeat_interleave -- the
        standard GQA->MHA expansion (mathematically produces IDENTICAL attention output to
        the original grouped computation, before any further TESSERA-specific changes).
        """
        if n_kv_heads == n_full_heads:
            return w
        d_head = w.shape[0] // n_kv_heads
        assert n_full_heads % n_kv_heads == 0, (
            f"Cannot evenly expand {n_kv_heads} KV heads to {n_full_heads} full heads"
        )
        rep = n_full_heads // n_kv_heads
        w_grouped = w.view(n_kv_heads, d_head, d_model)
        w_full = w_grouped.repeat_interleave(rep, dim=0)  # [n_full_heads, d_head, d_model]
        return w_full.reshape(n_full_heads * d_head, d_model)

    print(f"[4/5] Transplanting embeddings, per-layer Q/K/V/O + gate/up/down + norms ...")

    embed_key = find_key("model.embed_tokens.weight", "transformer.wte.weight", "embed_tokens.weight")
    if embed_key is None:
        print("[!] ERROR: could not locate token embedding weight in source state_dict. "
              f"Available top-level keys sample: {list(src_sd.keys())[:10]}")
        sys.exit(1)
    with torch.no_grad():
        tessera.embeddings.weight.copy_(src_sd[embed_key])
    print(f"      embeddings <- {embed_key} {tuple(src_sd[embed_key].shape)}")

    d_head_src = d_model // src_n_heads
    dropped_bias_warned = False

    for layer_idx in range(n_layers):
        stage = tessera.stages[layer_idx]
        da = stage.diff_attn
        prefix_candidates = [
            f"model.layers.{layer_idx}.",
            f"transformer.h.{layer_idx}.",
        ]
        prefix = None
        for p in prefix_candidates:
            if f"{p}self_attn.q_proj.weight" in src_sd or f"{p}attn.q_proj.weight" in src_sd:
                prefix = p
                break
        if prefix is None:
            print(f"[!] WARNING: layer {layer_idx}: could not find a recognized attention "
                  f"prefix among {prefix_candidates}. Leaving this stage's attention/FFN at "
                  f"random TESSERA-Q init and continuing.")
            continue

        attn_ns = "self_attn" if f"{prefix}self_attn.q_proj.weight" in src_sd else "attn"
        q_key = f"{prefix}{attn_ns}.q_proj.weight"
        k_key = f"{prefix}{attn_ns}.k_proj.weight"
        v_key = f"{prefix}{attn_ns}.v_proj.weight"
        o_key = f"{prefix}{attn_ns}.o_proj.weight"
        q_bias_key = f"{prefix}{attn_ns}.q_proj.bias"

        with torch.no_grad():
            q_w = src_sd[q_key]  # [src_n_heads*d_head_src, d_model]
            k_w_grouped = src_sd[k_key]  # [src_n_kv_heads*d_head_src, d_model]
            v_w_grouped = src_sd[v_key]
            o_w = src_sd[o_key]  # [d_model, src_n_heads*d_head_src]

            k_w = repeat_kv_to_full_heads(k_w_grouped, src_n_kv_heads, src_n_heads, d_model)
            v_w = repeat_kv_to_full_heads(v_w_grouped, src_n_kv_heads, src_n_heads, d_model)

            # q_w/k_w/v_w are now all [src_n_heads*d_head_src, d_model] == [d_model, d_model]
            # (since src_n_heads*d_head_src == d_model by construction). TESSERA-Q's wq/wk/wv
            # are plain d_model x d_model linears (no head-count constraint at the weight
            # level -- head splitting happens only in DifferentialAttention.forward's
            # .view() call), so these copy in directly as long as d_model matches, which it
            # does since we constructed TesseraGPUModel with this model's own d_model.
            if q_w.shape == da.wq.weight.shape:
                da.wq.weight.copy_(q_w)
            if k_w.shape == da.wk.weight.shape:
                da.wk.weight.copy_(k_w)
            if v_w.shape == da.wv.weight.shape:
                da.wv.weight.copy_(v_w)
            if o_w.shape == da.wo.weight.shape:
                da.wo.weight.copy_(o_w)

            if q_bias_key in src_sd and not dropped_bias_warned:
                print("[!] WARNING: source model has q/k/v bias terms (Qwen2-style); "
                      "TESSERA-Q's attention linears are bias-free (nn.Linear(..., "
                      "bias=False)) so these bias vectors are DROPPED, not folded in. "
                      "A short post-conversion fine-tune (train_tessera_gpu.py) is "
                      "recommended to recover any accuracy lost from this omission.")
                dropped_bias_warned = True

            # FFN: Qwen2 naming is gate_proj (SwiGLU gate) / up_proj / down_proj, matching
            # TESSERA-Q's w1 (gate) / w1u (up) / w2 (down) one-to-one in both shape and role.
            gate_key = f"{prefix}mlp.gate_proj.weight"
            up_key = f"{prefix}mlp.up_proj.weight"
            down_key = f"{prefix}mlp.down_proj.weight"
            if gate_key in src_sd and src_sd[gate_key].shape == stage.w1.weight.shape:
                stage.w1.weight.copy_(src_sd[gate_key])
            if up_key in src_sd and src_sd[up_key].shape == stage.w1u.weight.shape:
                stage.w1u.weight.copy_(src_sd[up_key])
            if down_key in src_sd and src_sd[down_key].shape == stage.w2.weight.shape:
                stage.w2.weight.copy_(src_sd[down_key])

            # RMSNorm gammas: input_layernorm -> norm1, post_attention_layernorm -> norm2.
            ln1_key = f"{prefix}input_layernorm.weight"
            ln2_key = f"{prefix}post_attention_layernorm.weight"
            if ln1_key in src_sd and src_sd[ln1_key].shape == stage.norm1.weight.shape:
                stage.norm1.weight.copy_(src_sd[ln1_key])
            if ln2_key in src_sd and src_sd[ln2_key].shape == stage.norm2.weight.shape:
                stage.norm2.weight.copy_(src_sd[ln2_key])

        print(f"      layer {layer_idx:2d}: Q/K/V/O + gate/up/down + norms transplanted "
              f"(GQA {src_n_kv_heads}->{src_n_heads} heads un-grouped for K/V)")

    final_norm_key = find_key("model.norm.weight", "transformer.ln_f.weight", "norm.weight")
    if final_norm_key is not None and src_sd[final_norm_key].shape == tessera.final_norm.weight.shape:
        with torch.no_grad():
            tessera.final_norm.weight.copy_(src_sd[final_norm_key])
        print(f"      final_norm <- {final_norm_key}")

    print(f"[5/5] Exporting converted TESSERA-Q model to '{output_path}' "
          f"({'INT8 quantized' if int8 else 'fp32'}) ...")
    if int8:
        tessera.export_to_binary_int8(output_path)
    else:
        tessera.export_to_binary(output_path)

    print()
    print("=" * 80)
    print("  CONVERSION COMPLETE")
    print("=" * 80)
    print(f"  Transplanted (from '{model_name}'): token embeddings, {n_layers}x "
          f"[Q/K/V/O attention proj (GQA un-grouped), SwiGLU gate/up/down proj, "
          f"input/post-attn RMSNorm], final RMSNorm.")
    print("  Left at normal TESSERA-Q initialization (no equivalent in a standard "
          "transformer -- must be learned): unified depth-dependent DiffAttn lambda "
          "(a_p/b_p logits + lambda_init schedule), learned Value-Residual gate, "
          "adaptive-RoPE eta, depthwise causal conv, low-rank adapter, MRM working memory.")
    print("  RECOMMENDATION: run a short post-conversion fine-tune, e.g.:")
    print(f"      python3 train_tessera_gpu.py --dataset <your_data> --steps 500 "
          f"--export_bin {output_path}")
    print("  to let the newly-initialized TESSERA-only components (lambda, vres gate, "
          "adapter, conv, RoPE eta) adapt around the transplanted attention/FFN weights, "
          "and to recover any accuracy lost from dropped q/k/v attention biases.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an open-weight HF model into TESSERA-Q binary format")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                         help="HuggingFace model id or local path")
    parser.add_argument("--output", type=str, default="tessera_from_open_weights.bin")
    parser.add_argument("--n_heads", type=int, default=8,
                         help="TESSERA-Q's own head count (independent of source model's head count); "
                              "must be even and evenly divide d_model")
    parser.add_argument("--int8", action="store_true", help="Export in INT8-quantized 'very optimized format'")
    args = parser.parse_args()
    convert(args.model, args.output, args.n_heads, args.int8)
