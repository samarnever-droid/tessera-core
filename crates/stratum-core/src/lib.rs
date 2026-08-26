//! `stratum-core`: STRATUM CPU-primary architecture, Product-Key Memory, Delta-Rule recurrence, and F4 test harness.

pub mod delta_rule;
pub mod f4_experiment;
pub mod pkm;
pub mod stratum_model;
pub mod stratum_trainer;

pub use delta_rule::*;
pub use f4_experiment::*;
pub use pkm::*;
pub use stratum_model::*;
pub use stratum_trainer::*;
