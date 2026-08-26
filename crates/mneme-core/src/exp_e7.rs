//! Experiment E7: Quantization Cascade under Depth-Recurrence.
//! Evaluates whether shared-weight quantization (FP32, INT8, W6, W4) compounds destructively across R passes.

use crate::mneme_model::{MnemeConfig, MnemeModel};
use crate::mneme_trainer::{evaluate_mneme_bpc, train_mneme};
use axiom_train::dataset::CharDataset;

#[derive(Debug, Clone)]
pub struct QuantizationResult {
    pub precision: String,
    pub bits: usize,
    pub bpc_r1: f32,
    pub bpc_r2: f32,
    pub bpc_r4: f32,
    pub bpc_r8: f32,
}

pub fn run_e7(dataset_path: &str, train_steps: usize) -> Vec<QuantizationResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E7: QUANTIZATION CASCADE UNDER DEPTH-RECURRENCE (R = 1, 2, 4, 8)");
    println!("  Testing FP32, INT8, W6, W4 precision under shared weight depth reuse");
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    // Train baseline FP32 model with stochastic depth
    let mut cfg = MnemeConfig::nano_default();
    cfg.n_passes = 4;
    let mut base_model = MnemeModel::new(vocab_size, seq_len, cfg, 42);

    println!("Training FP32 Base Model with stochastic R in 1..4...");
    let _ = train_mneme(
        &mut base_model,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        train_steps,
        base_lr,
        true,
        "FP32 Model for Quantization",
    );

    let precision_levels = [
        ("FP32 (Unquantized)", 32usize),
        ("INT8 (8-bit Quant)", 8usize),
        ("W6 (6-bit Quant)", 6usize),
        ("W4 (4-bit Quant)", 4usize),
    ];

    let mut results = Vec::new();

    for (name, bits) in precision_levels {
        let mut q_model = base_model.clone();
        q_model.apply_quantization_simulation(bits);

        let (_, bpc_r1) = evaluate_mneme_bpc(&mut q_model, &val_data.data, 15, seq_len, 1);
        let (_, bpc_r2) = evaluate_mneme_bpc(&mut q_model, &val_data.data, 15, seq_len, 2);
        let (_, bpc_r4) = evaluate_mneme_bpc(&mut q_model, &val_data.data, 15, seq_len, 4);
        let (_, bpc_r8) = evaluate_mneme_bpc(&mut q_model, &val_data.data, 15, seq_len, 8);

        println!(
            "  {:<20} | R=1: {:.4} | R=2: {:.4} | R=4: {:.4} | R=8: {:.4}",
            name, bpc_r1, bpc_r2, bpc_r4, bpc_r8
        );

        results.push(QuantizationResult {
            precision: name.to_string(),
            bits,
            bpc_r1,
            bpc_r2,
            bpc_r4,
            bpc_r8,
        });
    }

    println!("\n=======================================================================================================================");
    println!("                                EXPERIMENT E7: QUANTIZATION CASCADE MATRIX");
    println!("=======================================================================================================================");
    println!("{:<22} | {:<12} | {:<12} | {:<12} | {:<12}",
        "Precision Tier", "BPC @ R=1", "BPC @ R=2", "BPC @ R=4", "BPC @ R=8");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<22} | {:<12.4} | {:<12.4} | {:<12.4} | {:<12.4}",
            r.precision, r.bpc_r1, r.bpc_r2, r.bpc_r4, r.bpc_r8
        );
    }
    println!("=======================================================================================================================\n");

    let fp32_r4 = results[0].bpc_r4;
    let w4_r4 = results[3].bpc_r4;
    let w4_gap = w4_r4 - fp32_r4;

    println!("--- Kill Condition 5: W4 Quantization Cascade under Recurrence ---");
    println!("FP32 (R=4) BPC: {:.4}", fp32_r4);
    println!("W4   (R=4) BPC: {:.4}", w4_r4);
    println!("Quantization Degradation Gap: {:+.4} BPC", w4_gap);

    if w4_gap < 0.20 {
        println!(">>> Finding: W4 quantization remains stable under depth recurrence (gap {:+.4} BPC < 0.20).", w4_gap);
    } else {
        println!(">>> Kill Condition 5 Triggered: W4 quantization causes severe degradation under recurrence (gap {:+.4} BPC >= 0.20).", w4_gap);
    }
    println!();

    results
}
