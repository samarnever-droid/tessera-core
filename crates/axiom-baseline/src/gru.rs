//! Single-layer Gated Recurrent Unit (GRU) baseline with standard full Backprop-Through-Time (BPTT).

use axiom_core::activations::sigmoid;
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::cross_entropy_loss_and_grad;
use axiom_core::tensor::{MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Gradients for Single-Layer GRU Model.
#[derive(Debug, Clone)]
pub struct GruGrads {
    pub grad_embeddings: Vec<f32>,
    pub grad_w_gate: Vec<f32>, // (2d x 2d)
    pub grad_w_cand: Vec<f32>, // (d x 2d)
    pub grad_w_head: Vec<f32>, // (V x d)
}

impl GruGrads {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let d = d_model;
        Self {
            grad_embeddings: vec![0.0f32; vocab_size * d],
            grad_w_gate: vec![0.0f32; 2 * d * 2 * d],
            grad_w_cand: vec![0.0f32; d * 2 * d],
            grad_w_head: vec![0.0f32; vocab_size * d],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embeddings.fill(0.0f32);
        self.grad_w_gate.fill(0.0f32);
        self.grad_w_cand.fill(0.0f32);
        self.grad_w_head.fill(0.0f32);
    }

    pub fn add(&mut self, other: &GruGrads) {
        for (a, &b) in self.grad_embeddings.iter_mut().zip(other.grad_embeddings.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_w_gate.iter_mut().zip(other.grad_w_gate.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_w_cand.iter_mut().zip(other.grad_w_cand.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_w_head.iter_mut().zip(other.grad_w_head.iter()) {
            *a += b;
        }
    }
}

/// Standalone Single-Layer GRU Language Model.
#[derive(Debug, Clone)]
pub struct GruModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embeddings: Vec<f32>, // (vocab_size x d_model)
    pub w_gate: Vec<f32>,     // (2*d_model x 2*d_model) -> produces reset gate r_t and update gate z_t
    pub w_cand: Vec<f32>,     // (d_model x 2*d_model)   -> produces candidate state h_tilde
    pub w_head: Vec<f32>,     // (vocab_size x d_model)  -> produces prediction logits
}

impl GruModel {
    pub fn new(vocab_size: usize, d_model: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = d_model;
        let scale_embed = (1.0f32 / d as f32).sqrt();
        let scale_gate = (1.0f32 / (2 * d) as f32).sqrt();
        let scale_head = (1.0f32 / d as f32).sqrt();

        let embeddings: Vec<f32> = (0..vocab_size * d)
            .map(|_| rng.gen_range(-scale_embed..scale_embed))
            .collect();
        let w_gate: Vec<f32> = (0..2 * d * 2 * d)
            .map(|_| rng.gen_range(-scale_gate..scale_gate))
            .collect();
        let w_cand: Vec<f32> = (0..d * 2 * d)
            .map(|_| rng.gen_range(-scale_gate..scale_gate))
            .collect();
        let w_head: Vec<f32> = (0..vocab_size * d)
            .map(|_| rng.gen_range(-scale_head..scale_head))
            .collect();

        Self {
            vocab_size,
            d_model,
            embeddings,
            w_gate,
            w_cand,
            w_head,
        }
    }

    /// Forward pass over a sequence of tokens [0..T], caching activations for BPTT.
    /// Returns total cross-entropy loss over the sequence.
    pub fn forward_sequence(
        &self,
        x_seq: &[usize],
        y_seq: &[usize],
        // Pre-allocated cache buffers:
        h_cache: &mut [f32],     // ((T + 1) * d)
        rz_cache: &mut [f32],    // (T * 2d)
        cand_cache: &mut [f32],  // (T * d)
        cand_in_cache: &mut [f32], // (T * 2d)
        logits_cache: &mut [f32], // (T * V)
        probs_cache: &mut [f32], // (T * V)
        pred_grad_cache: &mut [f32], // (T * V)
    ) -> f32 {
        let d = self.d_model;
        let v = self.vocab_size;
        let t_len = x_seq.len();

        let gate_view = MatrixView::new(&self.w_gate, 2 * d, 2 * d);
        let cand_view = MatrixView::new(&self.w_cand, d, 2 * d);
        let head_view = MatrixView::new(&self.w_head, v, d);

        // h_0 is initialized to zeros
        h_cache[..d].fill(0.0f32);

        let mut gate_in = vec![0.0f32; 2 * d];
        let mut raw_gates = vec![0.0f32; 2 * d];
        let mut raw_cand = vec![0.0f32; d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let token_x = x_seq[t];
            let token_y = y_seq[t];

            let x_embed = &self.embeddings[token_x * d..(token_x + 1) * d];
            let h_prev = &h_cache[t * d..(t + 1) * d];

            // 1. [x_t; h_{t-1}]
            gate_in[..d].copy_from_slice(x_embed);
            gate_in[d..2 * d].copy_from_slice(h_prev);

            // 2. [r_t; z_t] = sigmoid(W_gate * [x_t; h_{t-1}])
            matvec(&gate_view, &gate_in, &mut raw_gates);
            let rz_t = &mut rz_cache[t * 2 * d..(t + 1) * 2 * d];
            sigmoid(&raw_gates, rz_t);

            let r_t = &rz_t[..d];
            let z_t = &rz_t[d..2 * d];

            // 3. Candidate input: [x_t; r_t ⊙ h_{t-1}]
            let cand_in_t = &mut cand_in_cache[t * 2 * d..(t + 1) * 2 * d];
            cand_in_t[..d].copy_from_slice(x_embed);
            for i in 0..d {
                cand_in_t[d + i] = r_t[i] * h_prev[i];
            }

            // 4. h_tilde = tanh(W_cand * cand_in_t)
            matvec(&cand_view, cand_in_t, &mut raw_cand);
            let cand_t = &mut cand_cache[t * d..(t + 1) * d];
            for i in 0..d {
                cand_t[i] = raw_cand[i].tanh();
            }

            // 5. h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
            for i in 0..d {
                h_cache[(t + 1) * d + i] = (1.0 - z_t[i]) * h_cache[t * d + i] + z_t[i] * cand_t[i];
            }

            // 6. Logits & Cross-Entropy loss
            let logits_t = &mut logits_cache[t * v..(t + 1) * v];
            let probs_t = &mut probs_cache[t * v..(t + 1) * v];
            let pred_grad_t = &mut pred_grad_cache[t * v..(t + 1) * v];
            let h_curr = &h_cache[(t + 1) * d..(t + 2) * d];

            matvec(&head_view, h_curr, logits_t);
            let step_loss = cross_entropy_loss_and_grad(logits_t, token_y, probs_t, pred_grad_t);
            total_loss += step_loss;
        }

        total_loss
    }

    /// Full Backprop-Through-Time (BPTT) backward pass across the sequence.
    pub fn backward_sequence(
        &self,
        x_seq: &[usize],
        h_cache: &[f32],
        rz_cache: &[f32],
        cand_cache: &[f32],
        cand_in_cache: &[f32],
        pred_grad_cache: &[f32],
        grads: &mut GruGrads,
    ) {
        let d = self.d_model;
        let v = self.vocab_size;
        let t_len = x_seq.len();

        let head_view = MatrixView::new(&self.w_head, v, d);
        let gate_view = MatrixView::new(&self.w_gate, 2 * d, 2 * d);
        let cand_view = MatrixView::new(&self.w_cand, d, 2 * d);

        let mut grad_head_view = MatrixViewMut::new(&mut grads.grad_w_head, v, d);
        let mut grad_gate_view = MatrixViewMut::new(&mut grads.grad_w_gate, 2 * d, 2 * d);
        let mut grad_cand_view = MatrixViewMut::new(&mut grads.grad_w_cand, d, 2 * d);

        let mut delta_h = vec![0.0f32; d];
        let mut delta_gate_in = vec![0.0f32; 2 * d];
        let mut delta_cand_in = vec![0.0f32; 2 * d];
        let mut delta_raw_cand = vec![0.0f32; d];
        let mut delta_raw_gate = vec![0.0f32; 2 * d];
        let mut gate_in_t = vec![0.0f32; 2 * d];

        // Backprop through time backwards from t = T-1 down to 0
        for t in (0..t_len).rev() {
            let token_x = x_seq[t];
            let x_embed = &self.embeddings[token_x * d..(token_x + 1) * d];
            let h_prev = &h_cache[t * d..(t + 1) * d];
            let h_curr = &h_cache[(t + 1) * d..(t + 2) * d];
            let rz_t = &rz_cache[t * 2 * d..(t + 1) * 2 * d];
            let r_t = &rz_t[..d];
            let z_t = &rz_t[d..2 * d];
            let cand_t = &cand_cache[t * d..(t + 1) * d];
            let cand_in_t = &cand_in_cache[t * 2 * d..(t + 1) * 2 * d];
            let pred_grad_t = &pred_grad_cache[t * v..(t + 1) * v];

            // 1. Accumulate prediction head gradients & backprop into delta_h
            outer_product_accumulate(pred_grad_t, h_curr, 1.0, &mut grad_head_view);
            matvec_transposed(&head_view, pred_grad_t, &mut delta_gate_in[..d]); // reuse buffer for W_head^T * grad
            for i in 0..d {
                delta_h[i] += delta_gate_in[i];
            }

            // 2. Backprop into candidate state and update gate
            for i in 0..d {
                // dL / d(h_tilde) = delta_h * z_t
                let d_cand = delta_h[i] * z_t[i];
                // dL / d(raw_cand) = d_cand * (1 - cand_t^2)
                delta_raw_cand[i] = d_cand * (1.0 - cand_t[i] * cand_t[i]);

                // dL / d(z_t) = delta_h * (cand_t - h_prev)
                let d_z = delta_h[i] * (cand_t[i] - h_prev[i]);
                // dL / d(raw_z) = d_z * z_t * (1 - z_t)
                delta_raw_gate[d + i] = d_z * z_t[i] * (1.0 - z_t[i]);
            }

            // 3. Accumulate W_cand gradients and backprop into cand_in
            outer_product_accumulate(&delta_raw_cand, cand_in_t, 1.0, &mut grad_cand_view);
            matvec_transposed(&cand_view, &delta_raw_cand, &mut delta_cand_in);

            // delta_cand_in: [delta_x_cand; delta_rh]
            // delta_rh_i = delta_cand_in[d + i] = dL / d(r_t * h_prev)
            for i in 0..d {
                let d_rh = delta_cand_in[d + i];
                // dL / d(r_t) = d_rh * h_prev
                let d_r = d_rh * h_prev[i];
                delta_raw_gate[i] = d_r * r_t[i] * (1.0 - r_t[i]);
            }

            // 4. Accumulate W_gate gradients and backprop into gate_in
            gate_in_t[..d].copy_from_slice(x_embed);
            gate_in_t[d..2 * d].copy_from_slice(h_prev);
            outer_product_accumulate(&delta_raw_gate, &gate_in_t, 1.0, &mut grad_gate_view);
            matvec_transposed(&gate_view, &delta_raw_gate, &mut delta_gate_in);

            // 5. Backprop into embedding table:
            // delta_x = delta_gate_in[..d] + delta_cand_in[..d]
            let embed_grad_slice = &mut grads.grad_embeddings[token_x * d..(token_x + 1) * d];
            for i in 0..d {
                embed_grad_slice[i] += delta_gate_in[i] + delta_cand_in[i];
            }

            // 6. Update delta_h for previous step t-1 (BPTT recurrent step):
            // delta_h_{t-1} = delta_h_t * (1 - z_t) + delta_gate_in[d..2d] + delta_cand_in[d..2d] * r_t
            for i in 0..d {
                delta_h[i] = delta_h[i] * (1.0 - z_t[i]) + delta_gate_in[d + i] + delta_cand_in[d + i] * r_t[i];
            }
        }
    }
}
