//! Experiment E2: T1 Recall — Needle-in-haystack recall test for MRM.
//! Inserts a unique key/value pair at a random position in a random-token sequence.
//! Then queries the value via the same key at the end of the sequence.
//! Compares: Transformer KV cache vs MRM vs MRM+fast-weights.
//! Target: >90% recall @ 4K (design spec T1).

use crate::mrm::MultiResMemory;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// Single recall trial result.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub context_len: usize,
    pub method: String,
    pub recall_correct: bool,
    pub best_cosine_sim: f32,
    pub latency_us: f64,
    pub memory_bytes: usize,
}

/// Aggregate recall statistics across trials.
#[derive(Debug, Clone)]
pub struct RecallStats {
    pub context_len: usize,
    pub method: String,
    pub accuracy: f32,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub mean_cosine_sim: f32,
    pub memory_bytes: usize,
}

/// Run a single needle-in-haystack recall trial.
/// Returns (correct, cosine_sim, latency_us).
fn recall_trial_mrm(
    mrm: &mut MultiResMemory,
    context_len: usize,
    insert_pos: usize,
    key_signal: &[f32],
    val_signal: &[f32],
    rng: &mut StdRng,
    d: usize,
    use_fast_weights: bool,
) -> (bool, f32, f64, usize) {
    // Reset memory
    mrm.fine_keys.fill(0.0f32);
    mrm.fine_vals.fill(0.0f32);
    mrm.coarse_keys.fill(0.0f32);
    mrm.coarse_vals.fill(0.0f32);
    mrm.write_ptr = 0;
    mrm.mean_h.fill(0.0f32);
    mrm.reset_fast_weights();
    mrm.stats = crate::mrm::MrmStats::default();

    let t0 = Instant::now();

    // Stream random tokens through MRM; force-write key/val at insert_pos
    for t in 0..context_len {
        let mut h = (0..d).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect::<Vec<f32>>();
        // Normalize
        let norm = h.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in h.iter_mut() { *x /= norm; }

        if t == insert_pos {
            mrm.force_write(key_signal, val_signal);
        } else {
            let mut out = vec![0.0f32; d];
            mrm.forward(&h, &mut out, false);
        }
    }

    // Optional fast-weight test-time update
    if use_fast_weights {
        mrm.fast_weight_update(key_signal, val_signal);
    }

    // Query
    let (recalled_out, cosine_sim) = mrm.recall(key_signal);

    // Check: did the recalled output resemble val_signal?
    // Map val_signal through W_o to get expected output
    let d_model = d;
    let wo = axiom_core::tensor::MatrixView::new(&mrm.w_o, d_model, d_model);
    let mut expected_out = vec![0.0f32; d_model];
    axiom_core::matvec::matvec(&wo, val_signal, &mut expected_out);
    axiom_core::tensor::vec_add_scaled(&mut expected_out, key_signal, 1.0); // residual

    // Cosine similarity between recalled_out and expected_out
    let dot_re: f32 = recalled_out.iter().zip(expected_out.iter()).map(|(a, b)| a * b).sum();
    let norm_r = recalled_out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let norm_e = expected_out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let cos_sim = dot_re / (norm_r * norm_e);
    let correct = cos_sim > 0.7; // threshold for "recall correct"

    let latency_us = t0.elapsed().as_secs_f64() * 1e6;
    let memory_bytes = (mrm.k_fine + mrm.k_coarse) * d * 4;

    (correct, cos_sim, latency_us, memory_bytes)
}

/// Transformer KV cache recall: exact attention over full context.
/// Stores every token key/value pair in RAM; O(N) memory, O(N) per-token compute.
fn recall_trial_transformer_kv(
    context_len: usize,
    insert_pos: usize,
    key_signal: &[f32],
    val_signal: &[f32],
    rng: &mut StdRng,
    d: usize,
) -> (bool, f32, f64, usize) {
    let t0 = Instant::now();

    // Store all (k, v) pairs (full attention cache)
    let mut kv_cache_k: Vec<Vec<f32>> = Vec::with_capacity(context_len);
    let mut kv_cache_v: Vec<Vec<f32>> = Vec::with_capacity(context_len);

    for t in 0..context_len {
        let mut h = (0..d).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect::<Vec<f32>>();
        let norm = h.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in h.iter_mut() { *x /= norm; }

        if t == insert_pos {
            kv_cache_k.push(key_signal.to_vec());
            kv_cache_v.push(val_signal.to_vec());
        } else {
            kv_cache_k.push(h.clone());
            kv_cache_v.push(h);
        }
    }

    // Exact attention: softmax(q · K^T) · V — query = key_signal
    let mut scores = vec![0.0f32; context_len];
    for (i, k) in kv_cache_k.iter().enumerate() {
        scores[i] = axiom_core::tensor::dot(key_signal, k);
    }
    let mut probs = vec![0.0f32; context_len];
    axiom_core::softmax::softmax(&scores, &mut probs);

    let mut recalled = vec![0.0f32; d];
    for (i, v) in kv_cache_v.iter().enumerate() {
        axiom_core::tensor::vec_add_scaled(&mut recalled, v, probs[i]);
    }

    // Cosine similarity between recalled and val_signal
    let dot_rv: f32 = recalled.iter().zip(val_signal.iter()).map(|(a, b)| a * b).sum();
    let norm_r = recalled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let norm_v = val_signal.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let cos_sim = dot_rv / (norm_r * norm_v);
    let correct = cos_sim > 0.7;

    let latency_us = t0.elapsed().as_secs_f64() * 1e6;
    let memory_bytes = context_len * d * 2 * 4; // 2 (k+v) × d × sizeof(f32)

    (correct, cos_sim, latency_us, memory_bytes)
}

fn compute_stats(results: &[RecallResult]) -> RecallStats {
    if results.is_empty() {
        return RecallStats {
            context_len: 0,
            method: "?".to_string(),
            accuracy: 0.0,
            p50_latency_us: 0.0,
            p95_latency_us: 0.0,
            p99_latency_us: 0.0,
            mean_cosine_sim: 0.0,
            memory_bytes: 0,
        };
    }
    let n = results.len() as f32;
    let accuracy = results.iter().filter(|r| r.recall_correct).count() as f32 / n;
    let mean_cos = results.iter().map(|r| r.best_cosine_sim).sum::<f32>() / n;
    let memory = results[0].memory_bytes;
    let context_len = results[0].context_len;
    let method = results[0].method.clone();

    let mut latencies: Vec<f64> = results.iter().map(|r| r.latency_us).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[(n * 0.50) as usize];
    let p95 = latencies[((n * 0.95) as usize).min(latencies.len() - 1)];
    let p99 = latencies[((n * 0.99) as usize).min(latencies.len() - 1)];

    RecallStats { context_len, method, accuracy, p50_latency_us: p50, p95_latency_us: p95, p99_latency_us: p99, mean_cosine_sim: mean_cos, memory_bytes: memory }
}

/// Run full Experiment E2 across all context lengths and methods.
pub fn run_e2(d: usize, k_fine: usize, k_coarse: usize, n_trials: usize) -> Vec<RecallStats> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E2: T1 RECALL — Needle-in-Haystack");
    println!("  d={} | k_fine={} | k_coarse={} | {} trials/cell", d, k_fine, k_coarse, n_trials);
    println!("  Target: >90% recall @ 4K (T1 kill criterion)");
    println!("==========================================================================\n");

    let context_lengths = [256usize, 1024, 4096, 8192, 16384, 32768];
    let mut all_stats: Vec<RecallStats> = Vec::new();

    for &ctx_len in &context_lengths {
        println!("  Context length: {} tokens...", ctx_len);
        let mut rng = StdRng::seed_from_u64(31337);

        // Generate a fixed unique key/value signal
        let key_signal: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.7 + 0.3) % 1.0) * 2.0 - 1.0).collect();
        let val_signal: Vec<f32> = (0..d).map(|i| ((i as f32 * 0.13 + 0.5) % 1.0) * 2.0 - 1.0).collect();
        let k_norm = key_signal.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        let key_signal: Vec<f32> = key_signal.iter().map(|x| x / k_norm).collect();
        let v_norm = val_signal.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        let val_signal: Vec<f32> = val_signal.iter().map(|x| x / v_norm).collect();

        // Method 1: Transformer KV cache (exact, O(N) memory)
        {
            let mut trials: Vec<RecallResult> = Vec::new();
            for _ in 0..n_trials {
                let insert_pos = rng.gen_range(0..ctx_len);
                let (correct, cos_sim, lat, mem) = recall_trial_transformer_kv(
                    ctx_len, insert_pos, &key_signal, &val_signal, &mut rng, d,
                );
                trials.push(RecallResult {
                    context_len: ctx_len,
                    method: "Transformer KV Cache".to_string(),
                    recall_correct: correct,
                    best_cosine_sim: cos_sim,
                    latency_us: lat,
                    memory_bytes: mem,
                });
            }
            all_stats.push(compute_stats(&trials));
        }

        // Method 2: MRM (no fast weights)
        {
            let mut mrm = MultiResMemory::new(d, k_fine, k_coarse, 42);
            let mut trials: Vec<RecallResult> = Vec::new();
            for _ in 0..n_trials {
                let insert_pos = rng.gen_range(0..ctx_len);
                let (correct, cos_sim, lat, mem) = recall_trial_mrm(
                    &mut mrm, ctx_len, insert_pos, &key_signal, &val_signal, &mut rng, d, false,
                );
                trials.push(RecallResult {
                    context_len: ctx_len,
                    method: "FORGE MRM".to_string(),
                    recall_correct: correct,
                    best_cosine_sim: cos_sim,
                    latency_us: lat,
                    memory_bytes: mem,
                });
            }
            all_stats.push(compute_stats(&trials));
        }

        // Method 3: MRM + fast weights
        {
            let mut mrm = MultiResMemory::new(d, k_fine, k_coarse, 42);
            let mut trials: Vec<RecallResult> = Vec::new();
            for _ in 0..n_trials {
                let insert_pos = rng.gen_range(0..ctx_len);
                let (correct, cos_sim, lat, mem) = recall_trial_mrm(
                    &mut mrm, ctx_len, insert_pos, &key_signal, &val_signal, &mut rng, d, true,
                );
                trials.push(RecallResult {
                    context_len: ctx_len,
                    method: "FORGE MRM+FastWeights".to_string(),
                    recall_correct: correct,
                    best_cosine_sim: cos_sim,
                    latency_us: lat,
                    memory_bytes: mem,
                });
            }
            all_stats.push(compute_stats(&trials));
        }
    }

    // Print table
    println!("\n{:<22} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>12}",
        "Method", "Ctx", "Recall%", "p50(µs)", "p95(µs)", "p99(µs)", "Memory(KB)");
    println!("{}", "-".repeat(90));
    for s in &all_stats {
        println!("{:<22} | {:>6} | {:>7.1}% | {:>8.1} | {:>8.1} | {:>8.1} | {:>12}",
            s.method, s.context_len,
            s.accuracy * 100.0,
            s.p50_latency_us, s.p95_latency_us, s.p99_latency_us,
            s.memory_bytes / 1024,
        );
    }

    // T1 verdict
    println!("\n--- T1 Kill Criterion: >90% recall @ 4K ---");
    let mrm_4k = all_stats.iter().find(|s| s.context_len == 4096 && s.method.contains("MRM") && !s.method.contains("Fast"));
    match mrm_4k {
        Some(s) if s.accuracy >= 0.90 => println!(">>> T1 PASSED: MRM recall at 4K = {:.1}% (≥90%)", s.accuracy * 100.0),
        Some(s) => println!(">>> T1 FAILED: MRM recall at 4K = {:.1}% (<90%). Compressed memory loses needles.", s.accuracy * 100.0),
        None => println!(">>> T1 DATA MISSING"),
    }
    println!();

    all_stats
}
