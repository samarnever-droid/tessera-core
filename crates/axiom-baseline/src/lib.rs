//! `axiom-baseline`: Reference standard recurrent and Transformer baselines for matched comparisons.

pub mod gru;
pub mod gru_trainer;
pub mod transformer;
pub mod transformer_trainer;

pub use gru::*;
pub use gru_trainer::*;
pub use transformer::*;
pub use transformer_trainer::*;

/// Configuration for standard reference Transformer baseline.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ffn: usize,
    pub max_seq_len: usize,
}

impl TransformerConfig {
    /// Matched 1M parameter toy tier (for Phase 1 & 2 character-level testing).
    pub fn toy_1m(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 128,
            n_layers: 4,
            n_heads: 4,
            d_ffn: 512,
            max_seq_len: 256,
        }
    }

    /// Matched 10M parameter small tier (for WikiText-2 testing).
    pub fn small_10m(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            d_ffn: 1536,
            max_seq_len: 512,
        }
    }

    /// Matched 100M parameter base tier (for Scale testing).
    pub fn base_100m(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
            d_ffn: 3072,
            max_seq_len: 2048,
        }
    }

    /// Calculate total trainable parameter count.
    pub fn param_count(&self) -> usize {
        let embed_params = self.vocab_size * self.d_model + self.max_seq_len * self.d_model;
        let per_layer_attn = 4 * self.d_model * self.d_model; // Q, K, V, Out
        let per_layer_ffn = 2 * self.d_model * self.d_ffn;   // Up, Down
        let per_layer_ln = 4 * self.d_model;                 // 2 LayerNorms (gamma, beta)
        let per_layer = per_layer_attn + per_layer_ffn + per_layer_ln;
        let head_params = self.d_model * self.vocab_size;

        embed_params + (self.n_layers * per_layer) + head_params
    }
}
