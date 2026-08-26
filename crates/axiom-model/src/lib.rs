//! `axiom-model`: LayerState, sparse expert MLP, decoupled forward passes,
//! and model definitions for AXIOM.

pub mod expert;
pub mod layer;
pub mod model;
pub mod stacked_model;

pub use expert::*;
pub use layer::*;
pub use model::*;
pub use stacked_model::*;

use axiom_core::*;

/// Architectural configuration for an AXIOM model.
#[derive(Debug, Clone)]
pub struct AxiomConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub active_experts: usize,
    pub buffer_capacity: usize,
    pub d_ffn: usize,
    pub hebbian_decay: f32,
    pub hebbian_lr: f32,
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            d_model: 128,
            num_layers: 4,
            num_experts: 8,
            active_experts: 2,
            buffer_capacity: 512,
            d_ffn: 512,
            hebbian_decay: 0.999,
            hebbian_lr: 1e-4,
        }
    }
}

/// Persistent state for a single AXIOM layer (§3.2).
#[derive(Debug, Clone)]
pub struct LayerState {
    pub memory: HebbianMemory,
    pub recurrent_state: Vec<f32>,
    pub buffer: CircularTokenBuffer,
}

impl LayerState {
    pub fn new(config: &AxiomConfig) -> Self {
        Self {
            memory: HebbianMemory::new(config.d_model, config.hebbian_decay, config.hebbian_lr),
            recurrent_state: vec![0.0f32; config.d_model],
            buffer: CircularTokenBuffer::new(config.buffer_capacity),
        }
    }

    pub fn reset(&mut self) {
        self.memory.clear();
        self.recurrent_state.fill(0.0f32);
        self.buffer.clear();
    }
}
