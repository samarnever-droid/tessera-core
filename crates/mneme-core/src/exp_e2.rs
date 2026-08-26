//! Experiment E2: The Knowledge / Computation Wall.
//! Evaluates scaling sparse knowledge capacity under fixed active compute.

use crate::mneme_model::{MnemeConfig, MnemeModel};
use crate::mneme_trainer::{evaluate_mneme_bpc, train_mneme};
use axiom_train::dataset::CharDataset;

#[derive(Debug, Clone)]
pub struct E2RungResult {
    pub rung_name: String,
    pub n_experts: usize,
    pub total_params: usize,
    pub active_params: usize,
    pub dram_bytes_per_token: usize,
    pub resident_l3_bytes: usize,
    pub val_loss: f32,
    pub val_bpc: f32,
    pub elapsed_sec: f64,
    pub tokens_per_sec: f64,
}

pub fn run_e2(dataset_path: &str, steps_per_rung: usize) -> Vec<E2RungResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E2: THE KNOWLEDGE / COMPUTATION WALL");
    println!("  Scaling Sparse Knowledge Tier across R0, R1, R2, R3 with Fixed Active Compute");
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    let rungs = [
        ("R0 (No Knowledge Tier, E=0)", 0usize),
        ("R1 (Small Knowledge Tier, E=16)", 16usize),
        ("R2 (Medium Knowledge Tier, E=64)", 64usize),
        ("R3 (Large Knowledge Tier, E=256)", 256usize),
    ];

    let mut results = Vec::new();

    for (label, n_exp) in rungs {
        println!("\n--- Running Rung: {} ---", label);
        let mut cfg = MnemeConfig::nano_default();
        cfg.n_experts = n_exp;
        let mut model = MnemeModel::new(vocab_size, seq_len, cfg, 42);
        let (tot_p, act_p, dram_b, l3_b) = model.parameter_metrics();

        let history = train_mneme(
            &mut model,
            &train_data.data,
            &val_data.data,
            batch_size,
            seq_len,
            max_time_secs,
            steps_per_rung,
            base_lr,
            false,
            label,
        );

        let (loss, bpc) = evaluate_mneme_bpc(&mut model, &val_data.data, 20, seq_len, cfg.n_passes);
        let elapsed = history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
        let tok_s = (steps_per_rung * batch_size * seq_len) as f64 / elapsed;

        results.push(E2RungResult {
            rung_name: label.to_string(),
            n_experts: n_exp,
            total_params: tot_p,
            active_params: act_p,
            dram_bytes_per_token: dram_b,
            resident_l3_bytes: l3_b,
            val_loss: loss,
            val_bpc: bpc,
            elapsed_sec: elapsed,
            tokens_per_sec: tok_s,
        });
    }

    // Print Comparative Scaling Curves
    println!("\n=======================================================================================================================");
    println!("                                EXPERIMENT E2: KNOWLEDGE WALL SCALING MATRIX");
    println!("=======================================================================================================================");
    println!("{:<35} | {:<10} | {:<10} | {:<12} | {:<9} | {:<9}",
        "Rung", "Total P", "Active P", "DRAM B/tok", "Val Loss", "Val BPC");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<35} | {:<10} | {:<10} | {:<12} | {:<9.4} | {:<9.4}",
            r.rung_name,
            format!("{:.2}M", r.total_params as f32 / 1e6),
            format!("{:.2}M", r.active_params as f32 / 1e6),
            format!("{} B", r.dram_bytes_per_token),
            r.val_loss,
            r.val_bpc,
        );
    }
    println!("=======================================================================================================================\n");

    // Evaluate Kill Condition 1 (Knowledge Wall)
    let bpc_r0 = results[0].val_bpc;
    let bpc_r3 = results[3].val_bpc;
    let delta_bpc = bpc_r0 - bpc_r3;

    println!("--- Knowledge Tier Scaling Evaluation (R0 -> R3) ---");
    println!("R0 (No Experts) BPC:       {:.4}", bpc_r0);
    println!("R3 (E=256 Experts) BPC:    {:.4}", bpc_r3);
    println!("Quality Gain (Δ BPC):      {:+.4} BPC", delta_bpc);

    if delta_bpc > 0.05 {
        println!(">>> Finding: Increasing sparse knowledge capacity produces measurable quality gains ({:+.4} BPC).", delta_bpc);
    } else {
        println!(">>> Kill Condition 1 Triggered: Sparse knowledge capacity produced negligible quality gain ({:+.4} BPC <= 0.05).", delta_bpc);
    }
    println!();

    results
}
