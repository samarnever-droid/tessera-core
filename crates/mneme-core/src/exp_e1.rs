//! Experiment E1: Frozen Control / MNEME-Nano Ablation vs Dense Transformer Baseline.

use crate::mneme_model::{MnemeConfig, MnemeModel};
use crate::mneme_trainer::{evaluate_mneme_bpc, train_mneme};
use axiom_baseline::transformer::TransformerModel;
use axiom_baseline::transformer_trainer::{evaluate_transformer_bpc, train_transformer};
use axiom_train::dataset::CharDataset;
use sysinfo::System;

#[derive(Debug, Clone)]
pub struct E1ArmResult {
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

pub fn run_e1(dataset_path: &str, steps_per_arm: usize) -> Vec<E1ArmResult> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E1: MNEME-NANO ABLATION vs DENSE TRANSFORMER CONTROL");
    println!("  Corpus: {} | Controlled Step Budget: {} steps/arm", dataset_path, steps_per_arm);
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    let mut results = Vec::new();

    // ── ARM A: Dense Transformer Baseline Control ─────────────────────────
    println!("[1/4] Running Arm A: Dense Transformer Baseline...");
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

    results.push(E1ArmResult {
        arm_name: "Arm A: Dense Transformer (Control)".to_string(),
        total_params: dense_params,
        active_params: dense_params,
        dram_bytes_per_token: dense_params * 4, // Dense reads ALL weights per token
        resident_l3_bytes: 0,
        val_loss: t_loss,
        val_bpc: t_bpc,
        tokens_per_sec: t_tok_s,
        elapsed_sec: t_elapsed,
        peak_rss_mb: get_current_rss_mb(),
    });

    // ── ARM B: MNEME Trunk Only (No sparse experts, E=0) ──────────────────
    println!("\n[2/4] Running Arm B: MNEME Trunk Only (No experts, E=0)...");
    let mut cfg_b = MnemeConfig::nano_default();
    cfg_b.n_experts = 0;
    let mut model_b = MnemeModel::new(vocab_size, seq_len, cfg_b, 42);
    let (b_tot, b_act, b_dram, b_l3) = model_b.parameter_metrics();

    let b_history = train_mneme(
        &mut model_b,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_arm,
        base_lr,
        false,
        "Arm B: MNEME-Trunk",
    );

    let (b_loss, b_bpc) = evaluate_mneme_bpc(&mut model_b, &val_data.data, 20, seq_len, cfg_b.n_passes);
    let b_elapsed = b_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let b_tok_s = (steps_per_arm * batch_size * seq_len) as f64 / b_elapsed;

    results.push(E1ArmResult {
        arm_name: "Arm B: MNEME Trunk Only (E=0)".to_string(),
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

    // ── ARM C: MNEME Trunk + Sparse Knowledge Tier (E=64, k=4) ────────────
    println!("\n[3/4] Running Arm C: MNEME Trunk + Knowledge Tier (E=64, k=4)...");
    let mut cfg_c = MnemeConfig::nano_default();
    cfg_c.n_experts = 64;
    let mut model_c = MnemeModel::new(vocab_size, seq_len, cfg_c, 42);
    let (c_tot, c_act, c_dram, c_l3) = model_c.parameter_metrics();

    let c_history = train_mneme(
        &mut model_c,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_arm,
        base_lr,
        false,
        "Arm C: MNEME+Experts",
    );

    let (c_loss, c_bpc) = evaluate_mneme_bpc(&mut model_c, &val_data.data, 20, seq_len, cfg_c.n_passes);
    let c_elapsed = c_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let c_tok_s = (steps_per_arm * batch_size * seq_len) as f64 / c_elapsed;

    results.push(E1ArmResult {
        arm_name: "Arm C: MNEME Trunk + Experts (E=64)".to_string(),
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

    // ── ARM D: Full MNEME-Nano (Trunk + Experts + Low-rank Adapters) ──────
    println!("\n[4/4] Running Arm D: Full MNEME-Nano (Trunk + Experts + Adapters)...");
    let mut cfg_d = MnemeConfig::nano_default();
    cfg_d.n_experts = 64;
    cfg_d.adapter_rank = 4;
    let mut model_d = MnemeModel::new(vocab_size, seq_len, cfg_d, 42);
    let (d_tot, d_act, d_dram, d_l3) = model_d.parameter_metrics();

    let d_history = train_mneme(
        &mut model_d,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_arm,
        base_lr,
        true, // Train with stochastic R for anytime validity
        "Arm D: Full MNEME-Nano",
    );

    let (d_loss, d_bpc) = evaluate_mneme_bpc(&mut model_d, &val_data.data, 20, seq_len, cfg_d.n_passes);
    let d_elapsed = d_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let d_tok_s = (steps_per_arm * batch_size * seq_len) as f64 / d_elapsed;

    results.push(E1ArmResult {
        arm_name: "Arm D: Full MNEME-Nano (Stochastic R)".to_string(),
        total_params: d_tot,
        active_params: d_act,
        dram_bytes_per_token: d_dram,
        resident_l3_bytes: d_l3,
        val_loss: d_loss,
        val_bpc: d_bpc,
        tokens_per_sec: d_tok_s,
        elapsed_sec: d_elapsed,
        peak_rss_mb: get_current_rss_mb(),
    });

    // ── Print Comparative Table ───────────────────────────────────────────
    println!("\n=======================================================================================================================");
    println!("                                   EXPERIMENT E1 FINAL MEASURED RESULTS");
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

    let t_bytes = results[0].dram_bytes_per_token as f32;
    let m_bytes = results[3].dram_bytes_per_token as f32;
    let byte_reduction = t_bytes / m_bytes.max(1.0);
    println!(">>> DRAM Inference Bytes Reduction (Transformer / Full MNEME): {:.1}x ({:.1} KB vs {:.1} KB)",
        byte_reduction, t_bytes / 1024.0, m_bytes / 1024.0);

    results
}
