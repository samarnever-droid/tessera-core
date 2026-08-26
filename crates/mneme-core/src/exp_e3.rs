//! Experiment E3: Residency / Cache Wall.
//! Measures single-token decode throughput, latency, and memory footprint
//! across varying trunk sizes to identify the cache residency boundary.

use crate::mneme_model::{MnemeConfig, MnemeModel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ResidencyBenchResult {
    pub d_model: usize,
    pub resident_l3_kb: usize,
    pub tokens_per_sec: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub cache_residency_tier: String,
}

pub fn run_e3() -> Vec<ResidencyBenchResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E3: RESIDENCY / CACHE WALL INFERENCE BENCHMARK");
    println!("  Measuring single-thread decode throughput across trunk dimensions d");
    println!("==========================================================================\n");

    let d_sizes = [32usize, 64, 96, 128, 192, 256];
    let vocab_size = 256;
    let seq_len = 64;
    let n_trials = 50;
    let mut rng = StdRng::seed_from_u64(42);

    let mut results = Vec::new();

    for &d in &d_sizes {
        let mut cfg = MnemeConfig::nano_default();
        cfg.d_model = d;
        cfg.d_ff = d * 2;
        cfg.d_state = (d / 4).max(16);

        let mut model = MnemeModel::new(vocab_size, seq_len, cfg, 42);
        let (_, _, _, resident_bytes) = model.parameter_metrics();
        let resident_kb = resident_bytes / 1024;

        let tier = if resident_kb <= 256 {
            "L1/L2 Resident"
        } else if resident_kb <= 16384 {
            "L3 Resident"
        } else {
            "Exceeds L3 (DRAM Spill)"
        };

        // Warm up cache
        let dummy_seq: Vec<usize> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
        let mut dummy_grads = crate::mneme_model::MnemeModelGrads::new(
            vocab_size, d, seq_len, &model.unique_blocks, cfg.n_passes
        );
        let _ = model.forward_backward_sequence(&dummy_seq, &dummy_seq, cfg.n_passes, &mut dummy_grads);

        // Benchmark decode latency
        let mut latencies = Vec::with_capacity(n_trials);
        for _ in 0..n_trials {
            let t0 = Instant::now();
            dummy_grads.zero();
            let _ = model.forward_backward_sequence(&dummy_seq, &dummy_seq, cfg.n_passes, &mut dummy_grads);
            let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
            latencies.push(elapsed_us / seq_len as f64);
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = latencies.len();
        let p50 = latencies[n / 2];
        let p95 = latencies[(n * 95 / 100).min(n - 1)];
        let p99 = latencies[(n * 99 / 100).min(n - 1)];
        let mean_us: f64 = latencies.iter().sum::<f64>() / n as f64;
        let tok_s = 1e6 / mean_us.max(1e-3);

        println!(
            "  d={:<3} | Trunk Core: {:>5} KB ({:<18}) | tok/s: {:>7.0} | p50: {:>6.2} µs | p99: {:>6.2} µs",
            d, resident_kb, tier, tok_s, p50, p99
        );

        results.push(ResidencyBenchResult {
            d_model: d,
            resident_l3_kb: resident_kb,
            tokens_per_sec: tok_s,
            p50_latency_us: p50,
            p95_latency_us: p95,
            p99_latency_us: p99,
            cache_residency_tier: tier.to_string(),
        });
    }

    println!("\n=======================================================================================================================");
    println!("                                EXPERIMENT E3: CACHE RESIDENCY RESULTS");
    println!("=======================================================================================================================");
    println!("{:<8} | {:<12} | {:<22} | {:<10} | {:<10} | {:<10}",
        "d_model", "Trunk Core", "Cache Tier", "Tok/s", "p50 (µs)", "p99 (µs)");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<8} | {:<12} | {:<22} | {:<10.0} | {:<10.2} | {:<10.2}",
            r.d_model, format!("{} KB", r.resident_l3_kb), r.cache_residency_tier, r.tokens_per_sec, r.p50_latency_us, r.p99_latency_us
        );
    }
    println!("=======================================================================================================================\n");

    results
}
