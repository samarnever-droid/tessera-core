//! Experiment E4: Test-Time Learning via Fast Weights (T5 kill criterion).
//! Compares few-shot adaptation at k=0,1,4,8,16,32 in-context examples.
//! FORGE must use ZERO gradient steps — only rank-1 fast-weight updates.
//! Frozen Transformer is the baseline.
//!
//! Task: character-level next-token prediction on domain-shifted text
//! (source = enwik8, test = Python code snippet → domain shift).

use crate::forge_model::{ForgeConfig, ForgeModel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Few-shot adaptation result for one (model, k) cell.
#[derive(Debug, Clone)]
pub struct FewShotResult {
    pub method: String,
    pub k_examples: usize,
    pub val_bpc: f32,
    pub val_loss: f32,
    pub used_gradient: bool,
}

/// Generate a simple synthetic domain-shift test corpus (Python-like code).
pub fn make_code_corpus(n: usize) -> Vec<u8> {
    let templates = [
        b"def foo(x):\n    return x * 2\n" as &[u8],
        b"for i in range(10):\n    print(i)\n",
        b"class MyClass:\n    def __init__(self):\n        self.val = 0\n",
        b"import numpy as np\nx = np.zeros(100)\n",
        b"if x > 0:\n    result = x\nelse:\n    result = -x\n",
    ];
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    let mut t = 0usize;
    while out.len() < n {
        let tmpl = templates[t % templates.len()];
        out.push(tmpl[i % tmpl.len()]);
        i += 1;
        if i % tmpl.len() == 0 { t += 1; }
    }
    out.truncate(n);
    out
}

/// Evaluate BPC on a text corpus using FORGE model (no gradient updates).
fn eval_bpc_forge(model: &mut ForgeModel, corpus: &[u8], seq_len: usize, n_samples: usize, seed: u64) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let max_start = corpus.len().saturating_sub(seq_len + 1);
    let mut total_loss = 0.0f32;
    let mut total_toks = 0usize;

    for _ in 0..n_samples {
        let s = rng.gen_range(0..=max_start);
        let toks: Vec<usize> = corpus[s..s + seq_len + 1].iter().map(|&b| b as usize).collect();
        let (loss, cnt) = model.forward_sequence_loss(&toks);
        total_loss += loss;
        total_toks += cnt;
    }

    let mean = total_loss / total_toks.max(1) as f32;
    mean / std::f32::consts::LN_2
}

/// Apply fast-weight adaptation from k examples (zero gradient steps).
/// Each example: sequence of bytes → extract a key/value pair from the embedding.
fn apply_fast_weight_adaptation(model: &mut ForgeModel, examples: &[&[u8]], k: usize) {
    // For each block's MRM, accumulate rank-1 outer products from the k examples.
    // Key = mean embedding of the example prefix.
    // Value = mean embedding of the example suffix.
    let d = model.d;
    let k_actual = k.min(examples.len());

    for block in model.blocks.iter_mut() {
        if let Some(ref mut mrm) = block.mrm {
            mrm.reset_fast_weights();
            if k_actual == 0 { continue; }

            let mut mean_key = vec![0.0f32; d];
            let mut mean_val = vec![0.0f32; d];

            for ex in examples.iter().take(k_actual) {
                let half = ex.len() / 2;
                // Key = mean embed of first half, val = mean embed of second half
                let key_toks = &ex[..half.max(1)];
                let val_toks = &ex[half..];

                let mut key_acc = vec![0.0f32; d];
                let mut val_acc = vec![0.0f32; d];

                for &b in key_toks {
                    let emb = &model.embeddings[b as usize * d..(b as usize + 1) * d];
                    axiom_core::tensor::vec_add_scaled(&mut key_acc, emb, 1.0);
                }
                for &b in val_toks {
                    let emb = &model.embeddings[b as usize * d..(b as usize + 1) * d];
                    axiom_core::tensor::vec_add_scaled(&mut val_acc, emb, 1.0);
                }

                let scale_k = 1.0 / key_toks.len().max(1) as f32;
                let scale_v = 1.0 / val_toks.len().max(1) as f32;
                axiom_core::tensor::vec_add_scaled(&mut mean_key, &key_acc, scale_k / k_actual as f32);
                axiom_core::tensor::vec_add_scaled(&mut mean_val, &val_acc, scale_v / k_actual as f32);
            }

            mrm.fast_weight_update(&mean_key, &mean_val);
        }
    }
}

/// Run Experiment E4.
pub fn run_e4(
    base_train_data: &[u8],
    forge_model: &ForgeModel,
    k_values: &[usize],
) -> Vec<FewShotResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E4: TEST-TIME LEARNING — Fast Weights vs Frozen");
    println!("  Task: Char-level prediction on code corpus (domain-shifted from Wikipedia)");
    println!("  FORGE uses ZERO gradient steps (rank-1 fast-weight only).");
    println!("==========================================================================\n");

    let code_corpus = make_code_corpus(32768);
    let seq_len = 64;
    let n_eval_samples = 20;

    // Prepare k=32 example windows from code corpus (the "few-shot examples")
    let examples: Vec<Vec<u8>> = (0..32)
        .map(|i| code_corpus[i * 64..(i * 64 + 64).min(code_corpus.len())].to_vec())
        .collect();
    let example_refs: Vec<&[u8]> = examples.iter().map(|v| v.as_slice()).collect();

    let mut results = Vec::new();

    for &k in k_values {
        println!("  k = {} in-context examples...", k);

        // ── FORGE without fast weights (frozen) ─────────────────────
        {
            let mut m: ForgeModel = forge_model.clone();
            // Disable fast weights
            for block in m.blocks.iter_mut() {
                if let Some(ref mut mrm) = block.mrm {
                    mrm.reset_fast_weights();
                }
            }
            let bpc = eval_bpc_forge(&mut m, &code_corpus, seq_len, n_eval_samples, k as u64);
            results.push(FewShotResult {
                method: "FORGE (frozen, no fast-weights)".to_string(),
                k_examples: k,
                val_bpc: bpc,
                val_loss: bpc * std::f32::consts::LN_2,
                used_gradient: false,
            });
        }

        // ── FORGE with fast weights (zero grad) ─────────────────────
        {
            let mut m: ForgeModel = forge_model.clone();
            apply_fast_weight_adaptation(&mut m, &example_refs[..k], k);
            let bpc = eval_bpc_forge(&mut m, &code_corpus, seq_len, n_eval_samples, k as u64);
            results.push(FewShotResult {
                method: "FORGE+FastWeights (zero grad)".to_string(),
                k_examples: k,
                val_bpc: bpc,
                val_loss: bpc * std::f32::consts::LN_2,
                used_gradient: false,
            });
        }
    }

    // Print table
    println!("\n{:<35} | {:>4} | {:>9} | {:>6}", "Method", "k", "Val BPC", "Grad?");
    println!("{}", "-".repeat(60));
    for r in &results {
        println!("{:<35} | {:>4} | {:>9.4} | {:>6}",
            r.method, r.k_examples, r.val_bpc,
            if r.used_gradient { "Yes" } else { "No" },
        );
    }

    // T5 verdict
    println!("\n--- T5 Kill Criterion: Fast weights produce measurable improvement (vs frozen) ---");
    let k16_frozen = results.iter().find(|r| r.k_examples == 16 && r.method.contains("frozen"));
    let k16_fw = results.iter().find(|r| r.k_examples == 16 && r.method.contains("FastWeights"));
    match (k16_frozen, k16_fw) {
        (Some(f), Some(fw)) => {
            let delta_bpc = f.val_bpc - fw.val_bpc;
            if delta_bpc > 0.05 {
                println!(">>> T5 PASSED: Fast weights improve k=16 BPC by {:.4} (Δ={:+.4})", fw.val_bpc, delta_bpc);
            } else {
                println!(">>> T5 FAILED: Fast weight improvement at k=16 is only {:.4} BPC (< 0.05 threshold)", delta_bpc);
                println!("    Remove fast-weight component or redesign the adaptation mechanism.");
            }
        }
        _ => println!(">>> T5: Missing k=16 data"),
    }
    println!();

    results
}
