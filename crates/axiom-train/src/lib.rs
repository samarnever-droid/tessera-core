//! `axiom-train`: Local loss functions, parallel layer optimizer, and training loop.

pub mod dataset;
pub mod freezing_trainer;
pub mod optimizer;
pub mod stacked_trainer;
pub mod trainer;

pub use dataset::*;
pub use freezing_trainer::*;
pub use optimizer::*;
pub use stacked_trainer::*;
pub use trainer::*;

/// Loss weight hyperparameters for local layer objective (§4.2):
/// L_l = lambda_1 * CE(p^l, y)
///     + lambda_2 * ||decode^l(h_l) - h_{l-1}||^2
///     + lambda_3 * load_balance(g)
///     + lambda_4 * ||h_l - h_{l-1}||^2 * mask
#[derive(Debug, Clone)]
pub struct LocalLossConfig {
    pub lambda_pred: f32,
    pub lambda_recon: f32,
    pub lambda_balance: f32,
    pub lambda_residual: f32,
}

impl Default for LocalLossConfig {
    fn default() -> Self {
        Self {
            lambda_pred: 1.0,
            lambda_recon: 0.05,
            lambda_balance: 0.0,
            lambda_residual: 0.01,
        }
    }
}
