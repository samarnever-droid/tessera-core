//! Experiment E1: Ablation — Tiny FORGE vs Dense Transformer.
//! Runs five model arms on identical data, tokenizer, seq len, dtype, step count:
//!   A. Dense Transformer (baseline)
//!   B. FORGE without MRM
//!   C. FORGE with MRM (surprise gate disabled → always write)
//!   D. FORGE with MRM + surprise gate
//!   E. Full FORGE with MRM + surprise gate + fast weights

use crate::forge_model::{ForgeConfig, ForgeModel};
use crate::forge_trainer::{evaluate_forge_bpc, train_forge, TrainResult};
use axiom_baseline::transformer::TransformerModel;
use axiom_baseline::transformer_trainer::{evaluate_transformer_bpc, train_transformer};
use axiom_train::dataset::CharDataset;
use std::time::Instant;

/// Run Experiment E1 and return results for all five arms.
pub fn run_e1(
    dataset_path: &str,
    steps: usize,
) -> Vec<TrainResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E1: TINY FORGE ABLATION vs DENSE TRANSFORMER");
    println!("  Dataset: {} | Steps/arm: {} | Arms: A B C D E", dataset_path, steps);
    println!("==========================================================================\n");

    let raw = CharDataset::from_file(dataset_path).expect("E1: dataset load failed");
    let (train_ds, val_ds) = raw.split(0.9);

    // Shared hyperparameters — identical across all arms
    let vocab_size = 256;
    let d_model    = 128;
    let d_ff       = 256;
    let n_blocks   = 2;
    let k_fine     = 64;
    let k_coarse   = 16;
    let max_seq    = 128;
    let batch_size = 32;
    let seq_len    = 64;
    let base_lr    = 3e-3_f32;
    let max_time   = 240.0_f64;

    let mut results: Vec<TrainResult> = Vec::new();

    // ── ARM A: Dense Transformer ──────────────────────────────────────
    println!("[A] Dense Transformer Baseline...");
    let mut transformer = TransformerModel::new(vocab_size, d_model, n_blocks, d_ff, max_seq, 42);
    let t_history = train_transformer(
        &mut transformer, &train_ds.data, &val_ds.data,
        batch_size, seq_len, max_time as u64, steps, base_lr,
    );
    let (t_loss, t_bpc) = evaluate_transformer_bpc(&transformer, &val_ds.data, 20, seq_len);
    let t_elapsed = t_history.last().map(|h| h.3).unwrap_or(1.0);
    let dense_params = transformer.embeddings.len()
        + transformer.pos_embeddings.len()
        + transformer.blocks.iter().map(|b| b.wq.len() + b.wk.len() + b.wv.len() + b.wo.len() + b.w1.len() + b.w2.len()).sum::<usize>()
        + transformer.head.len();
    // Theoretical forward FLOPs for Transformer at seq_len, d_model
    // ≈ 2 * seq_len * (4d² + 2d·seq_len) per layer × n_blocks
    let t_flops_per_tok = 2.0 * (4.0 * d_model as f64 * d_model as f64 + 2.0 * d_model as f64 * seq_len as f64)
        * n_blocks as f64;
    let t_total_flops = t_flops_per_tok * (steps * batch_size * seq_len) as f64 * 3.0; // fwd+bwd

    results.push(TrainResult {
        model_name: "A: Dense Transformer".to_string(),
        config: ForgeConfig { use_mrm: false, use_surprise_gate: false, use_fast_weights: false },
        total_params: dense_params,
        total_train_flops: t_total_flops,
        final_val_loss: t_loss,
        final_val_bpc: t_bpc,
        tokens_per_sec: (steps * batch_size * seq_len) as f64 / t_elapsed,
        elapsed_sec: t_elapsed,
        history: t_history.into_iter().map(|(s, l, b, e)| (s, l, b, e)).collect(),
    });

    // ── ARMS B-E: FORGE ablation variants ─────────────────────────────
    let forge_configs = [
        ForgeConfig::no_mrm(),
        ForgeConfig::mrm_only(),
        ForgeConfig::mrm_surprise(),
        ForgeConfig::full(),
    ];

    for cfg in forge_configs.iter() {
        println!("\n[{}] Running {}...", cfg.name().chars().next().unwrap(), cfg.name());
        let mut forge = ForgeModel::new(
            vocab_size, d_model, d_ff, n_blocks, k_fine, k_coarse, max_seq, *cfg, 42,
        );
        let result = train_forge(
            &mut forge, &train_ds.data, &val_ds.data,
            batch_size, seq_len, steps, max_time, base_lr,
            cfg.name(),
        );
        results.push(result);
    }

    // ── Print comparative table ────────────────────────────────────────
    println!("\n=======================================================================================================================");
    println!("  EXPERIMENT E1 — RESULTS");
    println!("=======================================================================================================================");
    println!("{:<30} | {:>9} | {:>12} | {:>9} | {:>9} | {:>9} | {:>9}",
        "Arm", "Params", "Train FLOPs", "Val Loss", "Val BPC", "Tok/s", "Time(s)");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<30} | {:>9} | {:>12.2e} | {:>9.4} | {:>9.4} | {:>9.0} | {:>9.1}",
            r.model_name,
            format!("{:.2}M", r.total_params as f32 / 1e6),
            r.total_train_flops,
            r.final_val_loss,
            r.final_val_bpc,
            r.tokens_per_sec,
            r.elapsed_sec,
        );
    }
    println!("=======================================================================================================================\n");

    // ── Mechanism attribution ──────────────────────────────────────────
    if results.len() >= 5 {
        let bpc_a = results[0].final_val_bpc;
        let bpc_b = results[1].final_val_bpc;
        let bpc_c = results[2].final_val_bpc;
        let bpc_d = results[3].final_val_bpc;
        let bpc_e = results[4].final_val_bpc;
        println!("--- E1 Mechanism Attribution ---");
        println!("  MRM contribution (C-B):             {:+.4} BPC", bpc_c - bpc_b);
        println!("  Surprise gate contribution (D-C):   {:+.4} BPC", bpc_d - bpc_c);
        println!("  Fast weights contribution (E-D):    {:+.4} BPC", bpc_e - bpc_d);
        println!("  FORGE Full vs Transformer (E-A):    {:+.4} BPC", bpc_e - bpc_a);
        println!("  FLOP ratio (Transformer/FORGE-Full):{:.2}×", results[0].total_train_flops / results[4].total_train_flops.max(1.0));
        println!();
    }

    results
}
