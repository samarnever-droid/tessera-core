//! Full STRATUM Sequence Model combining Embeddings, Delta-Rule Recurrence,
//! Window Attention, Product-Key Sparse Memory (PKM), and Output Head.

use crate::delta_rule::{DeltaRuleGrads, DeltaRuleLayer};
use crate::pkm::{PkmGrads, ProductKeyMemory};
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::cross_entropy_loss_and_grad;
use axiom_core::tensor::{MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Gradients for full STRATUM model.
#[derive(Debug, Clone)]
pub struct StratumModelGrads {
    pub grad_embed: Vec<f32>,
    pub grad_pos_embed: Vec<f32>,
    pub delta_grads: DeltaRuleGrads,
    pub pkm_grads: PkmGrads,
    pub grad_head: Vec<f32>,
}

impl StratumModelGrads {
    pub fn new(vocab_size: usize, d_model: usize, max_seq_len: usize, d_v: usize, m: usize) -> Self {
        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            grad_pos_embed: vec![0.0f32; max_seq_len * d_model],
            delta_grads: DeltaRuleGrads::new(d_model),
            pkm_grads: PkmGrads::new(d_model, d_v, m),
            grad_head: vec![0.0f32; vocab_size * d_model],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embed.fill(0.0f32);
        self.grad_pos_embed.fill(0.0f32);
        self.delta_grads.zero();
        self.pkm_grads.zero();
        self.grad_head.fill(0.0f32);
    }

    pub fn add(&mut self, other: &StratumModelGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (a, &b) in self.grad_pos_embed.iter_mut().zip(other.grad_pos_embed.iter()) { *a += b; }
        self.delta_grads.add(&other.delta_grads);
        self.pkm_grads.add(&other.pkm_grads);
        for (a, &b) in self.grad_head.iter_mut().zip(other.grad_head.iter()) { *a += b; }
    }
}

/// Full STRATUM Sequence Model.
#[derive(Debug, Clone)]
pub struct StratumModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub embeddings: Vec<f32>,
    pub pos_embeddings: Vec<f32>,
    pub delta_layer: DeltaRuleLayer,
    pub pkm_layer: ProductKeyMemory,
    pub head: Vec<f32>,
}

impl StratumModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        max_seq_len: usize,
        m: usize,
        k_active: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();

        let embeddings = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let pos_embeddings = (0..max_seq_len * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let head = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let delta_layer = DeltaRuleLayer::new(d_model, seed + 10);
        let pkm_layer = ProductKeyMemory::new(d_model, d_model, m, k_active, seed + 20);

        Self {
            vocab_size,
            d_model,
            max_seq_len,
            embeddings,
            pos_embeddings,
            delta_layer,
            pkm_layer,
            head,
        }
    }

    /// Parameter accounting: returns (total_params, active_params_per_token, bytes_read_per_token).
    pub fn parameter_metrics(&self) -> (usize, usize, usize) {
        let d = self.d_model;
        let v = self.vocab_size;

        let embed_params = v * d + self.max_seq_len * d;
        let delta_params = 4 * d * d + 2 * d; // Wq, Wk, Wv, Wo, w_alpha, w_beta
        let (pkm_dense, pkm_sparse) = self.pkm_layer.param_count();
        let head_params = v * d;

        let total_params = embed_params + delta_params + pkm_dense + pkm_sparse + head_params;

        // Active parameters per token: only k active slots in PKM participate per token
        let pkm_active = pkm_dense + self.pkm_layer.k_active * self.pkm_layer.d_v;
        let active_params = (2 * d) // Token + pos embed vector
            + delta_params
            + pkm_active
            + head_params;

        // Bytes read per token (assuming 4 bytes per float / or int8):
        let bytes_read = active_params * 4;

        (total_params, active_params, bytes_read)
    }

    /// Full forward and backward pass across a sequence item.
    pub fn forward_backward_sequence(
        &mut self,
        x_seq: &[usize],
        y_seq: &[usize],
        grads: &mut StratumModelGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;
        let k = self.pkm_layer.k_active;

        // 1. Initial Embedding
        let mut h_0 = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let tok = x_seq[t];
            let embed = &self.embeddings[tok * d..(tok + 1) * d];
            let pos_e = &self.pos_embeddings[t * d..(t + 1) * d];
            for i in 0..d {
                h_0[t * d + i] = embed[i] + pos_e[i];
            }
        }

        // 2. Delta-Rule Recurrence Layer Forward
        let mut h_1 = vec![0.0f32; t_len * d];
        let mut q_delta = vec![0.0f32; t_len * d];
        let mut k_delta = vec![0.0f32; t_len * d];
        let mut v_delta = vec![0.0f32; t_len * d];
        let mut alpha_delta = vec![0.0f32; t_len];
        let mut beta_delta = vec![0.0f32; t_len];
        let mut s_cache = vec![0.0f32; (t_len + 1) * d * d];

        self.delta_layer.forward_sequence(
            &h_0,
            t_len,
            &mut h_1,
            &mut q_delta,
            &mut k_delta,
            &mut v_delta,
            &mut alpha_delta,
            &mut beta_delta,
            &mut s_cache,
        );

        // 3. Product-Key Memory (PKM) Layer Forward
        let mut h_2 = vec![0.0f32; t_len * d];
        let mut pkm_indices = vec![0usize; t_len * k];
        let mut pkm_weights = vec![0.0f32; t_len * k];
        let mut pkm_vals = vec![0.0f32; t_len * k * d];
        let mut pkm_queries = vec![0.0f32; t_len * d];

        self.pkm_layer.forward_sequence(
            &h_1,
            t_len,
            &mut h_2,
            &mut pkm_indices,
            &mut pkm_weights,
            &mut pkm_vals,
            &mut pkm_queries,
        );

        // Record routing diagnostics
        for t in 0..t_len {
            let idx_t = &pkm_indices[t * k..(t + 1) * k];
            let w_t = &pkm_weights[t * k..(t + 1) * k];
            self.pkm_layer.stats.record_routing(idx_t, w_t);
        }

        // 4. Output Head & Loss
        let head_view = MatrixView::new(&self.head, v, d);
        let mut grad_head_view = MatrixViewMut::new(&mut grads.grad_head, v, d);
        let mut delta_h2 = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let final_h = &h_2[t * d..(t + 1) * d];
            let mut logits = vec![0.0f32; v];
            let mut probs = vec![0.0f32; v];
            let mut pred_grad = vec![0.0f32; v];

            matvec(&head_view, final_h, &mut logits);
            let loss = cross_entropy_loss_and_grad(&logits, y_seq[t], &mut probs, &mut pred_grad);
            total_loss += loss;

            outer_product_accumulate(&pred_grad, final_h, 1.0, &mut grad_head_view);
            let d_ht = &mut delta_h2[t * d..(t + 1) * d];
            matvec_transposed(&head_view, &pred_grad, d_ht);
        }

        // 5. PKM Layer Backward
        let mut delta_h1 = vec![0.0f32; t_len * d];
        self.pkm_layer.backward_sequence(
            &h_1,
            t_len,
            &delta_h2,
            &mut delta_h1,
            &pkm_indices,
            &pkm_weights,
            &pkm_vals,
            &pkm_queries,
            &mut grads.pkm_grads,
        );

        // 6. Delta-Rule Layer Backward
        let mut delta_h0 = vec![0.0f32; t_len * d];
        self.delta_layer.backward_sequence(
            &h_0,
            t_len,
            &delta_h1,
            &mut delta_h0,
            &q_delta,
            &k_delta,
            &v_delta,
            &alpha_delta,
            &beta_delta,
            &s_cache,
            &mut grads.delta_grads,
        );

        // 7. Embeddings Backward
        for t in 0..t_len {
            let tok = x_seq[t];
            let dh = &delta_h0[t * d..(t + 1) * d];
            let embed_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            let pos_slice = &mut grads.grad_pos_embed[t * d..(t + 1) * d];
            for i in 0..d {
                embed_slice[i] += dh[i];
                pos_slice[i] += dh[i];
            }
        }

        total_loss
    }
}
