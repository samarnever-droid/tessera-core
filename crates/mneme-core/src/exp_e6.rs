//! Experiment E6: Routing Stability and Expert Distribution Analysis.
//! Tracks expert hit distributions, routing entropy, and dead/hot expert counts.

use crate::mneme_model::{MnemeConfig, MnemeModel};
use crate::mneme_trainer::train_mneme;
use axiom_train::dataset::CharDataset;

#[derive(Debug, Clone)]
pub struct RoutingAuditReport {
    pub total_experts: usize,
    pub top_k: usize,
    pub dead_experts_count: usize,
    pub hot_experts_count: usize,
    pub mean_entropy: f32,
    pub active_expert_ratio: f32,
}

pub fn run_e6(dataset_path: &str, train_steps: usize) -> RoutingAuditReport {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E6: ROUTING STABILITY AND EXPERT DISTRIBUTION AUDIT");
    println!("  Auditing E=256 Product-Key Expert Tier over {} training steps", train_steps);
    println!("==========================================================================\n");

    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let vocab_size = 256;
    let seq_len = 64;
    let batch_size = 32;
    let max_time_secs = 240;
    let base_lr = 3e-3;

    let mut cfg = MnemeConfig::nano_default();
    cfg.n_experts = 256;
    cfg.top_k_experts = 4;
    let mut model = MnemeModel::new(vocab_size, seq_len, cfg, 42);

    let _ = train_mneme(
        &mut model,
        &train_data.data,
        &val_data.data,
        batch_size,
        seq_len,
        max_time_secs,
        train_steps,
        base_lr,
        false,
        "E6: Routing Stability Audit",
    );

    // Extract statistics from the first block's expert tier
    let exp_tier = model.unique_blocks[0].expert_tier.as_ref().unwrap();
    let stats = &exp_tier.stats;

    let dead = stats.dead_experts();
    let hot = stats.hot_experts();
    let mean_ent = stats.mean_entropy();
    let active_ratio = (stats.total_experts - dead) as f32 / stats.total_experts as f32;

    println!("\n=======================================================================================================================");
    println!("                                EXPERIMENT E6: ROUTING STABILITY AUDIT");
    println!("=======================================================================================================================");
    println!("Total Experts (E):             {}", stats.total_experts);
    println!("Top-k Selected:                {}", stats.top_k);
    println!("Dead Experts (<1% avg hits):   {} / {} ({:.1}%)", dead, stats.total_experts, (dead as f32 / stats.total_experts as f32) * 100.0);
    println!("Hot Experts (>5x avg hits):    {} / {} ({:.1}%)", hot, stats.total_experts, (hot as f32 / stats.total_experts as f32) * 100.0);
    println!("Active Expert Ratio:           {:.1}%", active_ratio * 100.0);
    println!("Mean Routing Entropy:          {:.4} nats", mean_ent);
    println!("=======================================================================================================================\n");

    RoutingAuditReport {
        total_experts: stats.total_experts,
        top_k: stats.top_k,
        dead_experts_count: dead,
        hot_experts_count: hot,
        mean_entropy: mean_ent,
        active_expert_ratio: active_ratio,
    }
}
