//! Experiment E4: Recurrent Depth Reuse Falsification.
//! Evaluates whether inference passes R in {1, 2, 4, 8} buy quality without additional DRAM traffic.

use crate::mneme_model::{MnemeConfig, MnemeModel};
use crate::mneme_trainer::{evaluate_mneme_bpc, train_mneme};
use axiom_train::dataset::CharDataset;

#[derive(Debug, Clone)]
pub struct DepthReuseResult {
    pub passes_r: usize,
    pub val_loss: f32,
    pub val_bpc: f32,
    pub flops_multiplier: f32,
    pub dram_bytes_per_token: usize,
}

pub fn run_e4(dataset_path: &str, train_steps: usize) -> Vec<DepthReuseResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E4: RECURRENT DEPTH REUSE FALSIFICATION (R = 1, 2, 4, 8)");
    println!("  Pre-registered test: Does increasing inference passes R improve quality?");
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    // Train with Stochastic R (R ~ {1, 2, 4}) so model is anytime-valid per HARDPOINT §2.1
    let mut cfg = MnemeConfig::nano_default();
    cfg.n_passes = 4; // Max depth 4 during training
    let mut model = MnemeModel::new(vocab_size, seq_len, cfg, 42);
    let (_, _, dram_bytes, _) = model.parameter_metrics();

    println!("Training MNEME model with stochastic depth R in 1..4...");
    let _ = train_mneme(
        &mut model,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        train_steps,
        base_lr,
        true, // stochastic R
        "MNEME Stochastic-R Training",
    );

    let test_r_values = [1usize, 2, 4, 8];
    let mut results = Vec::new();

    println!("\n--- Evaluating BPC vs Inference Depth R ---");
    for &r in &test_r_values {
        let (loss, bpc) = evaluate_mneme_bpc(&mut model, &val_data.data, 20, seq_len, r);
        let flops_mult = r as f32 / 2.0;

        println!(
            "  Inference Passes R={:<2} | Val Loss: {:.4} | Val BPC: {:.4} | FLOPs: {:.1}x | DRAM: {} B/tok (Invariant)",
            r, loss, bpc, flops_mult, dram_bytes
        );

        results.push(DepthReuseResult {
            passes_r: r,
            val_loss: loss,
            val_bpc: bpc,
            flops_multiplier: flops_mult,
            dram_bytes_per_token: dram_bytes,
        });
    }

    println!("\n=======================================================================================================================");
    println!("                                EXPERIMENT E4: DEPTH REUSE RESULTS");
    println!("=======================================================================================================================");
    println!("{:<16} | {:<12} | {:<12} | {:<16} | {:<16}",
        "Inference R", "Val Loss", "Val BPC", "FLOPs Multiplier", "DRAM Bytes/token");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<16} | {:<12.4} | {:<12.4} | {:<16.1} | {:<16}",
            format!("R = {}", r.passes_r), r.val_loss, r.val_bpc, format!("{}x", r.flops_multiplier), format!("{} B", r.dram_bytes_per_token)
        );
    }
    println!("=======================================================================================================================\n");

    let bpc_r1 = results[0].val_bpc;
    let bpc_r2 = results[1].val_bpc;
    let bpc_r4 = results[2].val_bpc;
    let bpc_r8 = results[3].val_bpc;

    println!("--- Kill Condition 3: Recurrent Depth Reuse ---");
    println!("R=1 -> R=2 Quality Delta: {:+.4} BPC", bpc_r1 - bpc_r2);
    println!("R=2 -> R=4 Quality Delta: {:+.4} BPC", bpc_r2 - bpc_r4);
    println!("R=4 -> R=8 Quality Delta: {:+.4} BPC", bpc_r4 - bpc_r8);

    if (bpc_r1 - bpc_r4) > 0.05 {
        println!(">>> Finding: Depth-recurrence R successfully buys quality at ZERO additional DRAM bytes!");
    } else {
        println!(">>> Kill Condition 3 Triggered: Increasing R produces flat or worse BPC ({:+.4} delta).", bpc_r1 - bpc_r4);
    }
    println!();

    results
}
