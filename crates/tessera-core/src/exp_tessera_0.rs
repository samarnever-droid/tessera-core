//! EXP-TESSERA-0: Decisive Pre-Build Falsification Benchmark.
//! Evaluates Arms A, B, C, D with same-run comparative gating.

use crate::mrm_v2::MultiResMemoryV2;
use crate::tessera_model::{TesseraConfig, TesseraModel};
use crate::tessera_trainer::{evaluate_tessera_bpc, train_tessera};
use axiom_baseline::transformer::TransformerModel;
use axiom_baseline::transformer_trainer::{evaluate_transformer_bpc, train_transformer};
use axiom_train::dataset::CharDataset;
use sysinfo::System;

#[derive(Debug, Clone)]
pub struct TesseraArmResult {
    pub arm_name: String,
    pub total_params: usize,
    pub active_params: usize,
    pub dram_bytes_per_token: usize,
    pub resident_l3_bytes: usize,
    pub val_loss: f32,
    pub val_bpc: f32,
    pub tokens_per_sec: f64,
    pub elapsed_sec: f64,
    pub peak_rss_mb: f64,
}

pub fn get_current_rss_mb() -> f64 {
    let mut sys = System::new_all();
    sys.refresh_all();
    let pid = sysinfo::get_current_pid().ok();
    if let Some(p) = pid.and_then(|id| sys.process(id)) {
        p.memory() as f64 / (1024.0 * 1024.0)
    } else {
        0.0
    }
}

pub fn run_exp_tessera_0(dataset_path: &str, steps_per_arm: usize) {
    println!("==========================================================================");
    println!("  EXP-TESSERA-0: PRE-BUILD DECISIVE FALSIFICATION BENCHMARK");
    println!("  Dataset: {} | Steps per arm: {}", dataset_path, steps_per_arm);
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    let mut results = Vec::new();

    // ── ARM A: Dense Transformer Baseline (Same-Run Control) ──────────────
    println!("[1/4] Running Arm A: Dense Transformer Baseline (Same-Run Control)...");
    let mut transformer = TransformerModel::new(vocab_size, 128, 2, 512, seq_len, 42);
    let dense_params = transformer.embeddings.len()
        + transformer.pos_embeddings.len()
        + transformer.blocks.iter().map(|b| b.wq.len() + b.wk.len() + b.wv.len() + b.wo.len() + b.w1.len() + b.w2.len()).sum::<usize>()
        + transformer.head.len();

    let t_history = train_transformer(
        &mut transformer,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_arm,
        base_lr,
    );

    let (t_loss, t_bpc) = evaluate_transformer_bpc(&transformer, &val_data.data, 20, seq_len);
    let t_elapsed = t_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let t_tok_s = (steps_per_arm * batch_size * seq_len) as f64 / t_elapsed;

    results.push(TesseraArmResult {
        arm_name: "Arm A: Dense Transformer (Control)".to_string(),
        total_params: dense_params,
        active_params: dense_params,
        dram_bytes_per_token: dense_params * 4,
        resident_l3_bytes: 0,
        val_loss: t_loss,
        val_bpc: t_bpc,
        tokens_per_sec: t_tok_s,
        elapsed_sec: t_elapsed,
        peak_rss_mb: get_current_rss_mb(),
    });

    // ── ARM B: TESSERA Trunk Only (Progressive Folding, No MRM) ───────────
    println!("\n[2/4] Running Arm B: TESSERA-Trunk Only (Progressive Folding, No MRM)...");
    let mut cfg_b = TesseraConfig::nano_default();
    cfg_b.use_mrm_v2 = false;
    let mut model_b = TesseraModel::new(vocab_size, seq_len, cfg_b, 42);
    let (b_tot, b_act, b_dram, b_l3) = model_b.parameter_metrics();

    let b_history = train_tessera(
        &mut model_b,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_arm,
        base_lr,
        "Arm B: TESSERA-Trunk",
    );

    let (b_loss, b_bpc) = evaluate_tessera_bpc(&mut model_b, &val_data.data, 20, seq_len);
    let b_elapsed = b_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let b_tok_s = (steps_per_arm * batch_size * seq_len) as f64 / b_elapsed;

    results.push(TesseraArmResult {
        arm_name: "Arm B: TESSERA Trunk (No MRM)".to_string(),
        total_params: b_tot,
        active_params: b_act,
        dram_bytes_per_token: b_dram,
        resident_l3_bytes: b_l3,
        val_loss: b_loss,
        val_bpc: b_bpc,
        tokens_per_sec: b_tok_s,
        elapsed_sec: b_elapsed,
        peak_rss_mb: get_current_rss_mb(),
    });

    // ── ARM C: TESSERA Full (Progressive Folding + MRM-v2) ─────────────────
    println!("\n[3/4] Running Arm C: TESSERA Full (Progressive Folding + MRM-v2)...");
    let mut cfg_c = TesseraConfig::nano_default();
    cfg_c.use_mrm_v2 = true;
    let mut model_c = TesseraModel::new(vocab_size, seq_len, cfg_c, 42);
    let (c_tot, c_act, c_dram, c_l3) = model_c.parameter_metrics();

    let c_history = train_tessera(
        &mut model_c,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_arm,
        base_lr,
        "Arm C: TESSERA-Full",
    );

    let (c_loss, c_bpc) = evaluate_tessera_bpc(&mut model_c, &val_data.data, 20, seq_len);
    let c_elapsed = c_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let c_tok_s = (steps_per_arm * batch_size * seq_len) as f64 / c_elapsed;

    results.push(TesseraArmResult {
        arm_name: "Arm C: TESSERA Full (Trunk + MRM-v2)".to_string(),
        total_params: c_tot,
        active_params: c_act,
        dram_bytes_per_token: c_dram,
        resident_l3_bytes: c_l3,
        val_loss: c_loss,
        val_bpc: c_bpc,
        tokens_per_sec: c_tok_s,
        elapsed_sec: c_elapsed,
        peak_rss_mb: get_current_rss_mb(),
    });

    // ── ARM D: Needle-in-Haystack 1K Context Recall Benchmark ─────────────
    println!("\n[4/4] Running Arm D: MRM-v2 Needle-in-Haystack 1K Context Recall Probe...");
    let num_needle_trials = 30;
    let mut successful_recalls = 0usize;
    let mut mean_sim = 0.0f32;

    for trial in 0..num_needle_trials {
        let mut mrm = MultiResMemoryV2::new(128, 128, 16, 1000 + trial as u64);
        let cos_sim = mrm.probe_needle_recall(1024, 2000 + trial as u64);
        mean_sim += cos_sim;
        if cos_sim >= 0.70 {
            successful_recalls += 1;
        }
    }
    let needle_recall_rate = (successful_recalls as f32 / num_needle_trials as f32) * 100.0;
    let avg_cos_sim = mean_sim / num_needle_trials as f32;

    println!("Needle-in-Haystack 1K Context Results: {} / {} passed ({:.1}%) | Avg Cosine Sim: {:.4}",
        successful_recalls, num_needle_trials, needle_recall_rate, avg_cos_sim);

    // ── Comparative Performance Table ─────────────────────────────────────
    println!("\n=======================================================================================================================");
    println!("                                   EXP-TESSERA-0 MEASURED BENCHMARK MATRIX");
    println!("=======================================================================================================================");
    println!("{:<36} | {:<10} | {:<10} | {:<12} | {:<9} | {:<9} | {:<9}",
        "Configuration", "Total P", "Active P", "DRAM B/tok", "Val Loss", "Val BPC", "Tok/s");
    println!("-----------------------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<36} | {:<10} | {:<10} | {:<12} | {:<9.4} | {:<9.4} | {:<9.0}",
            r.arm_name,
            format!("{:.2}M", r.total_params as f32 / 1e6),
            format!("{:.2}M", r.active_params as f32 / 1e6),
            format!("{} B", r.dram_bytes_per_token),
            r.val_loss,
            r.val_bpc,
            r.tokens_per_sec,
        );
    }
    println!("=======================================================================================================================\n");

    // ── Pre-Registered Kill Criteria Evaluation ───────────────────────────
    println!("==========================================================================");
    println!("  PRE-REGISTERED KILL CRITERIA EVALUATION");
    println!("==========================================================================\n");

    let bpc_a = results[0].val_bpc; // Control
    let bpc_b = results[1].val_bpc; // Trunk Only
    let bpc_c = results[2].val_bpc; // Full TESSERA
    let dram_a = results[0].dram_bytes_per_token as f32;
    let dram_c = results[2].dram_bytes_per_token as f32;
    let byte_reduction = dram_a / dram_c.max(1.0);

    let relative_gap = bpc_c - bpc_a;
    let mrm_gain = bpc_b - bpc_c;

    println!("1. Quality Parity vs Same-Run Control (Arm C vs Arm A):");
    println!("   Same-Run Transformer Control (A): {:.4} BPC", bpc_a);
    println!("   TESSERA Full (C):                 {:.4} BPC", bpc_c);
    println!("   Relative Gap:                     {:+.4} BPC (Pre-registered threshold <= +0.10 BPC)", relative_gap);
    let pass_k1 = relative_gap <= 0.10;
    println!("   >>> Status: {}", if pass_k1 { "PASS" } else { "FAIL (Quality deficit too large)" });

    println!("\n2. Causal MRM-v2 Contribution (Arm C vs Arm B):");
    println!("   Trunk Only (B):                   {:.4} BPC", bpc_b);
    println!("   Trunk + MRM-v2 (C):               {:.4} BPC", bpc_c);
    println!("   Causal MRM Gain:                  {:+.4} BPC (Pre-registered threshold >= +0.08 BPC)", mrm_gain);
    let pass_k2 = mrm_gain >= 0.08;
    println!("   >>> Status: {}", if pass_k2 { "PASS" } else { "FAIL (MRM-v2 did not provide >= 0.08 BPC gain)" });

    println!("\n3. Needle-in-Haystack 1K Recall (Arm D):");
    println!("   1K Context Recall Rate:           {:.1}% (Pre-registered threshold >= 75.0%)", needle_recall_rate);
    let pass_k3 = needle_recall_rate >= 75.0;
    println!("   >>> Status: {}", if pass_k3 { "PASS" } else { "FAIL (Needle evicted or lost)" });

    println!("\n4. DRAM Bandwidth Reduction:");
    println!("   Transformer vs TESSERA Full:      {:.1}x Byte Reduction ({} B/tok vs {} B/tok)",
        byte_reduction, results[0].dram_bytes_per_token, results[2].dram_bytes_per_token);
    let pass_k5 = byte_reduction >= 50.0;
    println!("   >>> Status: {}", if pass_k5 { "PASS" } else { "FAIL (Insufficient byte win)" });

    println!("\n==========================================================================");
    let verdict = if pass_k1 && pass_k2 && pass_k3 && pass_k5 {
        "GREEN (TESSERA SURVIVES)"
    } else if pass_k3 && pass_k5 {
        "YELLOW (INTERESTING BUT REDESIGN)"
    } else {
        "RED (KILL ARCHITECTURE)"
    };

    println!("  FINAL TESSERA VERDICT: {}", verdict);
    println!("==========================================================================");
}
