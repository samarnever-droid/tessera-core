//! Price System (Phase 9): Lagrangian Dual Ascent & Hedge Value Predictor.
//!
//! Replaces heuristic eviction tournaments with mathematical dual ascent:
//! admit k <=> v_k > sum_r (lambda_r * c_{k,r})

use std::sync::RwLock;

#[derive(Clone, Debug, PartialEq)]
pub struct PriceVector {
    pub lambda_dram: f64,
    pub lambda_flash: f64,
    pub lambda_origin: f64,
    pub lambda_cpu: f64,
    pub lambda_slack: f64,
}

impl Default for PriceVector {
    fn default() -> Self {
        Self {
            lambda_dram: 1.0,
            lambda_flash: 0.5,
            lambda_origin: 2.0,
            lambda_cpu: 0.1,
            lambda_slack: 0.05,
        }
    }
}

pub struct DualAscentEngine {
    pub prices: RwLock<PriceVector>,
    learning_rate: f64,
}

impl DualAscentEngine {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            prices: RwLock::new(PriceVector::default()),
            learning_rate,
        }
    }

    /// Update shadow prices based on resource pressure / slack.
    /// lambda_r <- [ lambda_r + eta * (usage_r / Cap_r - 1) ]+
    pub fn step_update(&self, dram_ratio: f64, flash_ratio: f64, origin_ratio: f64, cpu_ratio: f64) {
        let mut p = self.prices.write().unwrap();
        let eta = self.learning_rate;

        p.lambda_dram = (p.lambda_dram + eta * (dram_ratio - 1.0)).max(0.01).min(100.0);
        p.lambda_flash = (p.lambda_flash + eta * (flash_ratio - 1.0)).max(0.01).min(100.0);
        p.lambda_origin = (p.lambda_origin + eta * (origin_ratio - 1.0)).max(0.01).min(100.0);
        p.lambda_cpu = (p.lambda_cpu + eta * (cpu_ratio - 1.0)).max(0.01).min(100.0);
    }

    /// Single inequality admission decision: admit <=> predicted_value > total_shadow_cost
    pub fn admit(&self, predicted_value: f64, dram_bytes: usize, origin_cost: f64) -> bool {
        let p = self.prices.read().unwrap();
        let shadow_cost = (dram_bytes as f64 / 1024.0) * p.lambda_dram + origin_cost * p.lambda_origin;
        predicted_value >= shadow_cost
    }
}
