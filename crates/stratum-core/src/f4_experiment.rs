//! STRATUM Pre-Build Falsification Test (Experiment F4).
//! Rigorously evaluates the core capacity thesis:
//! "Does increasing sparse addressable capacity buy lower loss while active compute stays fixed?"

use crate::stratum_model::StratumModel;
use crate::stratum_trainer::{evaluate_stratum_bpc, train_stratum};
use axiom_baseline::transformer::TransformerModel;
use axiom_baseline::transformer_trainer::{evaluate_transformer_bpc, train_transformer};
use axiom_train::dataset::CharDataset;
use sysinfo::System;

/// Single experimental arm result in Experiment F4.
#[derive(Debug, Clone)]
pub struct F4ArmResult {
    pub model_name: String,
    pub total_params: usize,
    pub active_params_per_token: usize,
    pub bytes_read_per_token: usize,
    pub final_val_loss: f32,
    pub final_val_bpc: f32,
    pub training_tokens_per_sec: f64,
    pub elapsed_time_sec: f64,
    pub training_flops: f64,
    pub peak_rss_mb: f64,
    pub slot_utilization_pct: Option<f32>,
    pub routing_entropy: Option<f32>,
    pub median_to_mean_ratio: Option<f32>,
    pub hit_histogram: Option<Vec<(usize, usize, usize)>>,
}

/// Helper to get current RSS memory in MB.
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

/// Run full Experiment F4 suite.
pub fn run_f4_experiment(dataset_path: &str, steps_per_model: usize) -> Vec<F4ArmResult> {
    println!("=========================================================================");
    println!("   STRATUM PRE-BUILD FALSIFICATION TEST (EXPERIMENT F4)");
    println!("   Corpus: {} | Controlled Step Budget: {} steps/model", dataset_path, steps_per_model);
    println!("=========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let d_model = 128;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let total_train_tokens = steps_per_model * batch_size * seq_len;

    let mut results = Vec::new();

    // ---------------------------------------------------------------------
    // ARM 1: Dense Transformer Baseline Control
    // ---------------------------------------------------------------------
    println!("\n[1/4] Running Dense Transformer Baseline Control...");
    let mut transformer = TransformerModel::new(vocab_size, d_model, 2, 512, seq_len, 42);
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
        steps_per_model,
        3e-3,
    );

    let (t_loss, t_bpc) = evaluate_transformer_bpc(&transformer, &val_data.data, 20, seq_len);
    let t_elapsed = t_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let t_tok_s = total_train_tokens as f64 / t_elapsed;
    let t_flops = 6.0 * (dense_params as f64) * (total_train_tokens as f64);

    results.push(F4ArmResult {
        model_name: "Dense Transformer (Control)".to_string(),
        total_params: dense_params,
        active_params_per_token: dense_params,
        bytes_read_per_token: dense_params * 4,
        final_val_loss: t_loss,
        final_val_bpc: t_bpc,
        training_tokens_per_sec: t_tok_s,
        elapsed_time_sec: t_elapsed,
        training_flops: t_flops,
        peak_rss_mb: get_current_rss_mb(),
        slot_utilization_pct: None,
        routing_entropy: None,
        median_to_mean_ratio: None,
        hit_histogram: None,
    });

    // ---------------------------------------------------------------------
    // ARM 2: STRATUM (N = 256 slots, m = 16, k = 16) - Matched Active Compute
    // ---------------------------------------------------------------------
    println!("\n[2/4] Running STRATUM-N256 (Matched Active Params)...");
    let mut stratum_256 = StratumModel::new(vocab_size, d_model, seq_len, 16, 16, 42);
    let (s256_tot, s256_act, s256_bytes) = stratum_256.parameter_metrics();

    let s256_history = train_stratum(
        &mut stratum_256,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_model,
        3e-3,
    );

    let (s256_loss, s256_bpc) = evaluate_stratum_bpc(&mut stratum_256, &val_data.data, 20, seq_len);
    let s256_elapsed = s256_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let s256_tok_s = total_train_tokens as f64 / s256_elapsed;
    let s256_flops = 6.0 * (s256_act as f64) * (total_train_tokens as f64);
    let s256_mean_ent = stratum_256.pkm_layer.stats.routing_entropies.iter().sum::<f32>() / stratum_256.pkm_layer.stats.routing_entropies.len().max(1) as f32;

    results.push(F4ArmResult {
        model_name: "STRATUM (N=256 slots, m=16)".to_string(),
        total_params: s256_tot,
        active_params_per_token: s256_act,
        bytes_read_per_token: s256_bytes,
        final_val_loss: s256_loss,
        final_val_bpc: s256_bpc,
        training_tokens_per_sec: s256_tok_s,
        elapsed_time_sec: s256_elapsed,
        training_flops: s256_flops,
        peak_rss_mb: get_current_rss_mb(),
        slot_utilization_pct: Some(stratum_256.pkm_layer.stats.slot_utilization()),
        routing_entropy: Some(s256_mean_ent),
        median_to_mean_ratio: Some(stratum_256.pkm_layer.stats.median_to_mean_ratio()),
        hit_histogram: Some(stratum_256.pkm_layer.stats.hit_count_histogram()),
    });

    // ---------------------------------------------------------------------
    // ARM 3: STRATUM Capacity Scale-Up (N = 4,096 slots, m = 64, k = 16)
    // ---------------------------------------------------------------------
    println!("\n[3/4] Running STRATUM-N4096 (4x Total Capacity Scale-Up, Fixed Active Compute)...");
    let mut stratum_4096 = StratumModel::new(vocab_size, d_model, seq_len, 64, 16, 42);
    let (s4096_tot, s4096_act, s4096_bytes) = stratum_4096.parameter_metrics();

    let s4096_history = train_stratum(
        &mut stratum_4096,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_model,
        3e-3,
    );

    let (s4096_loss, s4096_bpc) = evaluate_stratum_bpc(&mut stratum_4096, &val_data.data, 20, seq_len);
    let s4096_elapsed = s4096_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let s4096_tok_s = total_train_tokens as f64 / s4096_elapsed;
    let s4096_flops = 6.0 * (s4096_act as f64) * (total_train_tokens as f64);
    let s4096_mean_ent = stratum_4096.pkm_layer.stats.routing_entropies.iter().sum::<f32>() / stratum_4096.pkm_layer.stats.routing_entropies.len().max(1) as f32;

    results.push(F4ArmResult {
        model_name: "STRATUM (N=4096 slots, m=64)".to_string(),
        total_params: s4096_tot,
        active_params_per_token: s4096_act,
        bytes_read_per_token: s4096_bytes,
        final_val_loss: s4096_loss,
        final_val_bpc: s4096_bpc,
        training_tokens_per_sec: s4096_tok_s,
        elapsed_time_sec: s4096_elapsed,
        training_flops: s4096_flops,
        peak_rss_mb: get_current_rss_mb(),
        slot_utilization_pct: Some(stratum_4096.pkm_layer.stats.slot_utilization()),
        routing_entropy: Some(s4096_mean_ent),
        median_to_mean_ratio: Some(stratum_4096.pkm_layer.stats.median_to_mean_ratio()),
        hit_histogram: Some(stratum_4096.pkm_layer.stats.hit_count_histogram()),
    });

    // ---------------------------------------------------------------------
    // ARM 4: STRATUM Extreme Capacity Scale-Up (N = 65,536 slots, m = 256, k = 16)
    // ---------------------------------------------------------------------
    println!("\n[4/4] Running STRATUM-N65536 (40x Total Capacity Scale-Up, Fixed Active Compute)...");
    let mut stratum_65k = StratumModel::new(vocab_size, d_model, seq_len, 256, 16, 42);
    let (s65k_tot, s65k_act, s65k_bytes) = stratum_65k.parameter_metrics();

    let s65k_history = train_stratum(
        &mut stratum_65k,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        steps_per_model,
        3e-3,
    );

    let (s65k_loss, s65k_bpc) = evaluate_stratum_bpc(&mut stratum_65k, &val_data.data, 20, seq_len);
    let s65k_elapsed = s65k_history.last().map(|&(_, _, _, time)| time).unwrap_or(1.0);
    let s65k_tok_s = total_train_tokens as f64 / s65k_elapsed;
    let s65k_flops = 6.0 * (s65k_act as f64) * (total_train_tokens as f64);
    let s65k_mean_ent = stratum_65k.pkm_layer.stats.routing_entropies.iter().sum::<f32>() / stratum_65k.pkm_layer.stats.routing_entropies.len().max(1) as f32;

    results.push(F4ArmResult {
        model_name: "STRATUM (N=65536 slots, m=256)".to_string(),
        total_params: s65k_tot,
        active_params_per_token: s65k_act,
        bytes_read_per_token: s65k_bytes,
        final_val_loss: s65k_loss,
        final_val_bpc: s65k_bpc,
        training_tokens_per_sec: s65k_tok_s,
        elapsed_time_sec: s65k_elapsed,
        training_flops: s65k_flops,
        peak_rss_mb: get_current_rss_mb(),
        slot_utilization_pct: Some(stratum_65k.pkm_layer.stats.slot_utilization()),
        routing_entropy: Some(s65k_mean_ent),
        median_to_mean_ratio: Some(stratum_65k.pkm_layer.stats.median_to_mean_ratio()),
        hit_histogram: Some(stratum_65k.pkm_layer.stats.hit_count_histogram()),
    });

    // ---------------------------------------------------------------------
    // PRINT COMPREHENSIVE F4 COMPARATIVE MATRIX & VERDICT
    // ---------------------------------------------------------------------
    println!("\n==========================================================================================================");
    println!("                                   EXPERIMENT F4 FINAL MEASURED RESULTS");
    println!("==========================================================================================================");
    println!("{:<32} | {:<10} | {:<10} | {:<9} | {:<9} | {:<9} | {:<9}",
        "Model", "Total P", "Active P", "Val Loss", "Val BPC", "Tok/s", "RAM (MB)");
    println!("----------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:<32} | {:<10} | {:<10} | {:<9.4} | {:<9.4} | {:<9.0} | {:<9.2}",
            r.model_name,
            format!("{:.2}M", r.total_params as f32 / 1e6),
            format!("{:.2}M", r.active_params_per_token as f32 / 1e6),
            r.final_val_loss,
            r.final_val_bpc,
            r.training_tokens_per_sec,
            r.peak_rss_mb,
        );
    }
    println!("==========================================================================================================\n");

    // Print Routing Diagnostics
    println!("--- PKM Sparse Routing & Utilization Diagnostics ---");
    for r in &results {
        if let (Some(util), Some(ent), Some(ratio)) = (r.slot_utilization_pct, r.routing_entropy, r.median_to_mean_ratio) {
            println!("{:<32} | Utilization: {:>5.1}% | Mean Entropy: {:>5.2} | Median/Mean Update Ratio: {:.4}",
                r.model_name, util, ent, ratio);
        }
    }

    // Print Histograms
    for r in &results {
        if let Some(ref hist) = r.hit_histogram {
            println!("\n--- Slot Hit-Count Histogram for {} ---", r.model_name);
            println!("{:<20} | {:<10}", "Hit Range (min-max)", "Slot Count");
            for (min_h, max_h, count) in hist {
                println!("{:<20} | {:<10}", format!("{}-{}", min_h, max_h), count);
            }
        }
    }

    // Pre-registered verdict evaluation
    let dense_bpc = results[0].final_val_bpc;
    let dense_p = results[0].total_params as f32;

    println!("\n=========================================================================");
    println!("   PRE-REGISTERED F4 VERDICT EVALUATION");
    println!("=========================================================================");
    println!("Dense Transformer Target BPC: {:.4} BPC (Total Params: {:.2}M)", dense_bpc, dense_p / 1e6);

    let best_stratum = results.iter().skip(1).min_by(|a, b| a.final_val_bpc.partial_cmp(&b.final_val_bpc).unwrap()).unwrap();
    println!("Best STRATUM Model:           {} (Val BPC: {:.4}, Total Params: {:.2}M)",
        best_stratum.model_name, best_stratum.final_val_bpc, best_stratum.total_params as f32 / 1e6);

    // Compute parameter multiplier
    let param_multiplier = (best_stratum.total_params as f32) / dense_p;
    println!("Measured Total Parameter Multiplier: {:.2}x", param_multiplier);

    if best_stratum.final_val_bpc <= dense_bpc {
        if param_multiplier < 1.5 {
            println!(">>> Verdict: [<1.5x] STRATUM STRONGLY WINS ITS CAPACITY THESIS.");
        } else if param_multiplier <= 2.5 {
            println!(">>> Verdict: [1.5x - 2.5x] STRATUM STILL HAS A PLAUSIBLE ECONOMIC ADVANTAGE.");
        } else if param_multiplier <= 8.0 {
            println!(">>> Verdict: [2.5x - 8.0x] MARGINAL; INVESTIGATE MEMORY ECONOMICS.");
        } else {
            println!(">>> Verdict: [>8.0x] TREAT CORE CAPACITY THESIS AS FAILED.");
        }
    } else {
        let bpc_gap = best_stratum.final_val_bpc - dense_bpc;
        println!(">>> Finding: Even at {:.2}M total parameters, STRATUM (+{:.4} BPC gap) DID NOT MATCH the Dense Transformer at {:.2}M params.",
            best_stratum.total_params as f32 / 1e6, bpc_gap, dense_p / 1e6);
        println!(">>> Parameter Multiplier for Equal Loss: > {:.1}x (UNDEFINED / FAILS TO INTERSECT)", (results[3].total_params as f32) / dense_p);
        println!(">>> Verdict: [>8.0x] TREAT CORE CAPACITY THESIS AS FAILED.");
    }
    println!("=========================================================================\n");

    results
}
