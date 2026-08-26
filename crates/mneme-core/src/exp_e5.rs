//! Experiment E5: Gated Delta-Rule Ablation Suite.
//! Measures the causal quality and stability impact of the gated delta rule & erasure term.

use crate::delta_rule::DeltaRuleMode;
use crate::mneme_model::{MnemeConfig, MnemeModel};
use crate::mneme_trainer::{evaluate_mneme_bpc, train_mneme};
use axiom_train::dataset::CharDataset;

#[derive(Debug, Clone)]
pub struct DeltaAblationResult {
    pub mode_name: String,
    pub mode: DeltaRuleMode,
    pub val_loss: f32,
    pub val_bpc: f32,
    pub tok_s: f64,
}

pub fn run_e5(dataset_path: &str, steps_per_arm: usize) -> Vec<DeltaAblationResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E5: GATED DELTA-RULE ABLATION SUITE");
    println!("  Ablating: No-Delta vs Ungated vs Gated-No-Erasure vs Full-Gated-Erasure");
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    let modes = [
        ("A: No Delta Rule (State-free)", DeltaRuleMode::NoDelta),
        ("B: Ungated Delta Rule (α=1, β=1)", DeltaRuleMode::Ungated),
        ("C: Gated Delta Rule (No Erasure)", DeltaRuleMode::GatedNoErasure),
        ("D: Full Gated Delta + Erasure", DeltaRuleMode::FullGatedErasure),
    ];

    let mut results = Vec::new();

    for (name, mode) in modes {
        println!("\n--- Running Delta Mode: {} ---", name);
        let mut cfg = MnemeConfig::nano_default();
        cfg.delta_mode = mode;
        let mut model = MnemeModel::new(vocab_size, seq_len, cfg, 42);

        let history = train_mneme(
            &mut model,
            &train_data.data,
            &val_data.data,
            batch_size,
            seq_len,
            max_time_secs,
            steps_per_arm,
            base_lr,
            false,
            name,
        );

        let (loss, bpc) = evaluate_mneme_bpc(&mut model, &val_data.data, 20, seq_len, cfg.n_passes);
        let elapsed = history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
        let tok_s = (steps_per_arm * batch_size * seq_len) as f64 / elapsed;

        results.push(DeltaAblationResult {
            mode_name: name.to_string(),
            mode,
            val_loss: loss,
            val_bpc: bpc,
            tok_s,
        });
    }

    println!("\n=======================================================================================================================");
    println!("                                EXPERIMENT E5: DELTA RULE ABLATION RESULTS");
    println!("=======================================================================================================================");
    println!("{:<36} | {:<12} | {:<12} | {:<12}",
        "Delta Rule Variant", "Val Loss", "Val BPC", "Tok/s");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<36} | {:<12.4} | {:<12.4} | {:<12.0}",
            r.mode_name, r.val_loss, r.val_bpc, r.tok_s
        );
    }
    println!("=======================================================================================================================\n");

    let bpc_none = results[0].val_bpc;
    let bpc_full = results[3].val_bpc;
    let delta_gain = bpc_none - bpc_full;

    println!("--- Kill Condition 4: Gated Delta-Rule Contribution ---");
    println!("State-Free BPC:            {:.4}", bpc_none);
    println!("Full Gated Delta-Rule BPC: {:.4}", bpc_full);
    println!("Delta-Rule Quality Gain:   {:+.4} BPC", delta_gain);

    if delta_gain > 0.05 {
        println!(">>> Finding: Gated delta-rule with erasure provides decisive quality advantage ({:+.4} BPC).", delta_gain);
    } else {
        println!(">>> Kill Condition 4 Triggered: Gated delta-rule contributed negligible quality ({:+.4} BPC <= 0.05).", delta_gain);
    }
    println!();

    results
}
