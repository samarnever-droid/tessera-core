//! AXIOM Layer implementation with explicit separation of `forward_infer` and `forward_train`,
//! input residual skip connections (§9.1 mitigation), and Truncated BPTT backward pass.

use crate::expert::{ExpertGrads, ExpertMLP};
use crate::{AxiomConfig, LayerState};
use axiom_core::activations::{mse_loss_and_grad, sigmoid, sigmoid_grad};
use axiom_core::matvec::{matvec, matvec_transposed, matvec_transposed_accumulate, outer_product_accumulate};
use axiom_core::softmax::{cross_entropy_loss_and_grad, entropy};
use axiom_core::tensor::{dot, vec_add_scaled, vec_copy, MatrixView, MatrixViewMut};
use axiom_core::topk::{top2, topk_softmax};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Pre-allocated scratch buffers to guarantee zero dynamic allocation in forward and backward passes.
#[derive(Debug, Clone)]
pub struct LayerScratch {
    pub m_recall: Vec<f32>,
    pub s_concat: Vec<f32>,
    pub raw_recurrent: Vec<f32>,
    pub gate_scores: Vec<f32>,
    pub expert_raw_ffn1: Vec<f32>,
    pub expert_act_ffn1: Vec<f32>,
    pub expert_out1: Vec<f32>,
    pub expert_raw_ffn2: Vec<f32>,
    pub expert_act_ffn2: Vec<f32>,
    pub expert_out2: Vec<f32>,
    pub pred_logits: Vec<f32>,
    pub pred_probs: Vec<f32>,
    pub pred_grad: Vec<f32>,
    pub recon_hidden: Vec<f32>,
    pub recon_grad: Vec<f32>,
    // Backward scratch
    pub delta_h1: Vec<f32>,
    pub delta_s: Vec<f32>,
    pub delta_us: Vec<f32>,
    pub scratch_delta_a: Vec<f32>,
    pub scratch_delta_z: Vec<f32>,
    pub scratch_exp_grad: Vec<f32>,
    pub scratch_transposed: Vec<f32>,
}

impl LayerScratch {
    pub fn new(config: &AxiomConfig) -> Self {
        Self {
            m_recall: vec![0.0f32; config.d_model],
            s_concat: vec![0.0f32; 3 * config.d_model],
            raw_recurrent: vec![0.0f32; config.d_model],
            gate_scores: vec![0.0f32; config.num_experts],
            expert_raw_ffn1: vec![0.0f32; config.d_ffn],
            expert_act_ffn1: vec![0.0f32; config.d_ffn],
            expert_out1: vec![0.0f32; config.d_model],
            expert_raw_ffn2: vec![0.0f32; config.d_ffn],
            expert_act_ffn2: vec![0.0f32; config.d_ffn],
            expert_out2: vec![0.0f32; config.d_model],
            pred_logits: vec![0.0f32; config.vocab_size],
            pred_probs: vec![0.0f32; config.vocab_size],
            pred_grad: vec![0.0f32; config.vocab_size],
            recon_hidden: vec![0.0f32; config.d_model],
            recon_grad: vec![0.0f32; config.d_model],
            delta_h1: vec![0.0f32; config.d_model],
            delta_s: vec![0.0f32; config.d_model],
            delta_us: vec![0.0f32; config.d_model],
            scratch_delta_a: vec![0.0f32; config.d_ffn],
            scratch_delta_z: vec![0.0f32; config.d_ffn],
            scratch_exp_grad: vec![0.0f32; config.d_model],
            scratch_transposed: vec![0.0f32; 3 * config.d_model],
        }
    }
}

/// Pre-allocated cache for sequence activations to support fast within-layer BPTT.
#[derive(Debug, Clone)]
pub struct SequenceCache {
    pub max_seq_len: usize,
    pub s_history: Vec<f32>,           // ((T + 1) * d)
    pub m_recall_history: Vec<f32>,    // (T * d)
    pub h_in_history: Vec<f32>,        // (T * d)
    pub h_out_history: Vec<f32>,       // (T * d)
    pub pred_grad_history: Vec<f32>,   // (T * V)
    pub recon_grad_history: Vec<f32>,  // (T * d)
    pub top_indices: Vec<(usize, usize)>, // T pairs
    pub top_weights: Vec<[f32; 2]>,    // T pairs
    pub exp_raw1: Vec<f32>,            // (T * d_ffn)
    pub exp_act1: Vec<f32>,            // (T * d_ffn)
    pub exp_out1: Vec<f32>,            // (T * d)
    pub exp_raw2: Vec<f32>,            // (T * d_ffn)
    pub exp_act2: Vec<f32>,            // (T * d_ffn)
    pub exp_out2: Vec<f32>,            // (T * d)
}

impl SequenceCache {
    pub fn new(config: &AxiomConfig, max_seq_len: usize) -> Self {
        let t = max_seq_len;
        let d = config.d_model;
        let v = config.vocab_size;
        let ffn = config.d_ffn;

        Self {
            max_seq_len: t,
            s_history: vec![0.0f32; (t + 1) * d],
            m_recall_history: vec![0.0f32; t * d],
            h_in_history: vec![0.0f32; t * d],
            h_out_history: vec![0.0f32; t * d],
            pred_grad_history: vec![0.0f32; t * v],
            recon_grad_history: vec![0.0f32; t * d],
            top_indices: vec![(0, 1); t],
            top_weights: vec![[0.0f32; 2]; t],
            exp_raw1: vec![0.0f32; t * ffn],
            exp_act1: vec![0.0f32; t * ffn],
            exp_out1: vec![0.0f32; t * d],
            exp_raw2: vec![0.0f32; t * ffn],
            exp_act2: vec![0.0f32; t * ffn],
            exp_out2: vec![0.0f32; t * d],
        }
    }
}

/// Gradients for all parameters in a single AXIOM layer.
#[derive(Debug, Clone)]
pub struct LayerGrads {
    pub grad_w_s: Vec<f32>,
    pub grad_w_gate: Vec<f32>,
    pub expert_grads: Vec<ExpertGrads>,
    pub grad_w_pred: Vec<f32>,
    pub grad_w_decode: Vec<f32>,
}

impl LayerGrads {
    pub fn new(config: &AxiomConfig) -> Self {
        let d = config.d_model;
        let mut expert_grads = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            expert_grads.push(ExpertGrads::new(d, config.d_ffn));
        }

        Self {
            grad_w_s: vec![0.0f32; d * (3 * d)],
            grad_w_gate: vec![0.0f32; config.num_experts * d],
            expert_grads,
            grad_w_pred: vec![0.0f32; config.vocab_size * d],
            grad_w_decode: vec![0.0f32; d * d],
        }
    }

    pub fn zero(&mut self) {
        self.grad_w_s.fill(0.0f32);
        self.grad_w_gate.fill(0.0f32);
        for eg in self.expert_grads.iter_mut() {
            eg.zero();
        }
        self.grad_w_pred.fill(0.0f32);
        self.grad_w_decode.fill(0.0f32);
    }

    pub fn add(&mut self, other: &LayerGrads) {
        for (a, &b) in self.grad_w_s.iter_mut().zip(other.grad_w_s.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_w_gate.iter_mut().zip(other.grad_w_gate.iter()) {
            *a += b;
        }
        for (a_eg, b_eg) in self.expert_grads.iter_mut().zip(other.expert_grads.iter()) {
            for (a, &b) in a_eg.grad_w_up.iter_mut().zip(b_eg.grad_w_up.iter()) {
                *a += b;
            }
            for (a, &b) in a_eg.grad_w_down.iter_mut().zip(b_eg.grad_w_down.iter()) {
                *a += b;
            }
        }
        for (a, &b) in self.grad_w_pred.iter_mut().zip(other.grad_w_pred.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_w_decode.iter_mut().zip(other.grad_w_decode.iter()) {
            *a += b;
        }
    }
}

/// A single AXIOM Layer (§3.2, §3.3).
#[derive(Debug, Clone)]
pub struct AxiomLayer {
    pub config: AxiomConfig,
    pub w_s: Vec<f32>,       // (d_model x 3*d_model)
    pub w_gate: Vec<f32>,    // (num_experts x d_model)
    pub experts: Vec<ExpertMLP>, // num_experts
    pub w_pred: Vec<f32>,    // (vocab_size x d_model) - for local next-token prediction
    pub w_decode: Vec<f32>,  // (d_model x d_model) - for input reconstruction
}

impl AxiomLayer {
    pub fn new(config: AxiomConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = config.d_model;
        let scale_ws = (2.0f32 / (3 * d) as f32).sqrt();
        let scale_gate = (2.0f32 / d as f32).sqrt();
        let scale_pred = (2.0f32 / d as f32).sqrt();
        let scale_decode = (2.0f32 / d as f32).sqrt();

        let w_s: Vec<f32> = (0..d * (3 * d))
            .map(|_| rng.gen_range(-scale_ws..scale_ws))
            .collect();
        let w_gate: Vec<f32> = (0..config.num_experts * d)
            .map(|_| rng.gen_range(-scale_gate..scale_gate))
            .collect();

        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            experts.push(ExpertMLP::new(d, config.d_ffn, seed + 100 + i as u64));
        }

        let w_pred: Vec<f32> = (0..config.vocab_size * d)
            .map(|_| rng.gen_range(-scale_pred..scale_pred))
            .collect();
        let w_decode: Vec<f32> = (0..d * d)
            .map(|_| rng.gen_range(-scale_decode..scale_decode))
            .collect();

        Self {
            config,
            w_s,
            w_gate,
            experts,
            w_pred,
            w_decode,
        }
    }

    /// Fast Inference Forward Pass (§11.3):
    /// Computes associative recall, recurrent update, sparse expert routing/compute,
    /// residual skip connection (h_out = h_in + s_t + experts), and Hebbian memory write.
    #[inline]
    pub fn forward_infer(
        &self,
        h_in: &[f32],
        state: &mut LayerState,
        scratch: &mut LayerScratch,
        h_out: &mut [f32],
    ) {
        let d = self.config.d_model;
        debug_assert_eq!(h_in.len(), d);
        debug_assert_eq!(h_out.len(), d);

        // 1. Associative Memory Recall: m_t = M_{t-1} * h_in
        state.memory.recall(h_in, &mut scratch.m_recall);

        // 2. Recurrent Update: s_t = sigmoid(W_s * [h_in; m_t; s_{t-1}])
        scratch.s_concat[..d].copy_from_slice(h_in);
        scratch.s_concat[d..2 * d].copy_from_slice(&scratch.m_recall);
        scratch.s_concat[2 * d..3 * d].copy_from_slice(&state.recurrent_state);

        let ws_view = MatrixView::new(&self.w_s, d, 3 * d);
        matvec(&ws_view, &scratch.s_concat, &mut scratch.raw_recurrent);
        sigmoid(&scratch.raw_recurrent, &mut state.recurrent_state);

        // 3. Expert Routing: top-2 from softmax(W_gate * s_t)
        let gate_view = MatrixView::new(&self.w_gate, self.config.num_experts, d);
        matvec(&gate_view, &state.recurrent_state, &mut scratch.gate_scores);
        let top_experts = top2(&scratch.gate_scores);
        let mut expert_weights = [0.0f32; 2];
        topk_softmax(&top_experts, &mut expert_weights);

        // 4. Sparse Expert Computation with residual skip: h_out = h_in + s_t + g1*Exp1 + g2*Exp2
        let idx1 = top_experts[0].1;
        let idx2 = top_experts[1].1;
        self.experts[idx1].forward(
            &state.recurrent_state,
            &mut scratch.expert_raw_ffn1,
            &mut scratch.expert_out1,
        );
        self.experts[idx2].forward(
            &state.recurrent_state,
            &mut scratch.expert_raw_ffn2,
            &mut scratch.expert_out2,
        );

        vec_copy(h_out, h_in);
        vec_add_scaled(h_out, &state.recurrent_state, 1.0);
        vec_add_scaled(h_out, &scratch.expert_out1, expert_weights[0]);
        vec_add_scaled(h_out, &scratch.expert_out2, expert_weights[1]);

        // 5. Fast Memory Write (Hebbian, online): M_t = lambda*M_{t-1} + eta*h_out*h_out^T
        state.memory.update(h_out);
    }

    /// Full Training Forward Pass with intermediate activation caching for backward autodiff.
    #[inline]
    pub fn forward_train(
        &self,
        h_in: &[f32],
        target_token: usize,
        state: &mut LayerState,
        scratch: &mut LayerScratch,
        h_out: &mut [f32],
    ) -> (f32, f32) {
        let d = self.config.d_model;

        // 1. Associative Memory Recall: m_t = M_{t-1} * h_in
        state.memory.recall(h_in, &mut scratch.m_recall);

        // 2. Recurrent Update: s_t = sigmoid(W_s * [h_in; m_t; s_{t-1}])
        scratch.s_concat[..d].copy_from_slice(h_in);
        scratch.s_concat[d..2 * d].copy_from_slice(&scratch.m_recall);
        scratch.s_concat[2 * d..3 * d].copy_from_slice(&state.recurrent_state);

        let ws_view = MatrixView::new(&self.w_s, d, 3 * d);
        matvec(&ws_view, &scratch.s_concat, &mut scratch.raw_recurrent);
        sigmoid(&scratch.raw_recurrent, &mut state.recurrent_state);

        // 3. Expert Routing: top-2 from softmax(W_gate * s_t)
        let gate_view = MatrixView::new(&self.w_gate, self.config.num_experts, d);
        matvec(&gate_view, &state.recurrent_state, &mut scratch.gate_scores);
        let top_experts = top2(&scratch.gate_scores);
        let mut expert_weights = [0.0f32; 2];
        topk_softmax(&top_experts, &mut expert_weights);

        // 4. Sparse Expert Computation with caching
        let idx1 = top_experts[0].1;
        let idx2 = top_experts[1].1;
        self.experts[idx1].forward_with_cache(
            &state.recurrent_state,
            &mut scratch.expert_raw_ffn1,
            &mut scratch.expert_act_ffn1,
            &mut scratch.expert_out1,
        );
        self.experts[idx2].forward_with_cache(
            &state.recurrent_state,
            &mut scratch.expert_raw_ffn2,
            &mut scratch.expert_act_ffn2,
            &mut scratch.expert_out2,
        );

        vec_copy(h_out, h_in);
        vec_add_scaled(h_out, &state.recurrent_state, 1.0);
        vec_add_scaled(h_out, &scratch.expert_out1, expert_weights[0]);
        vec_add_scaled(h_out, &scratch.expert_out2, expert_weights[1]);

        // 5. Fast Memory Write (Hebbian, online, no gradient through M)
        state.memory.update(h_out);

        // 6. Local Prediction Head: p^l = softmax(W_pred * h_out)
        let pred_view = MatrixView::new(&self.w_pred, self.config.vocab_size, d);
        matvec(&pred_view, h_out, &mut scratch.pred_logits);
        let loss_pred = cross_entropy_loss_and_grad(
            &scratch.pred_logits,
            target_token,
            &mut scratch.pred_probs,
            &mut scratch.pred_grad,
        );

        // Buffer push if surprisal (entropy) > threshold
        if entropy(&scratch.pred_probs) > 2.0 {
            state.buffer.push(target_token as u32);
        }

        // 7. Reconstruction Decoder: h_hat = W_decode * h_out
        let decode_view = MatrixView::new(&self.w_decode, d, d);
        matvec(&decode_view, h_out, &mut scratch.recon_hidden);
        let loss_recon = mse_loss_and_grad(&scratch.recon_hidden, h_in, &mut scratch.recon_grad);

        (loss_pred, loss_recon)
    }

    /// Fast analytical backward pass for local loss (§4.2) with residual skip connection.
    #[inline]
    pub fn backward_layer(
        &self,
        h_in: &[f32],
        state_prev: &[f32], // s_{t-1}
        state_curr: &[f32], // s_t
        h_out: &[f32],
        lambda_pred: f32,
        lambda_recon: f32,
        lambda_residual: f32,
        scratch: &mut LayerScratch,
        grads: &mut LayerGrads,
        grad_h_in: &mut [f32],
    ) {
        let d = self.config.d_model;
        let pred_view = MatrixView::new(&self.w_pred, self.config.vocab_size, d);
        let decode_view = MatrixView::new(&self.w_decode, d, d);
        let ws_view = MatrixView::new(&self.w_s, d, 3 * d);

        // 1. grad_w_pred += lambda_pred * (pred_grad ⊗ h_out)
        let mut grad_pred_view = MatrixViewMut::new(&mut grads.grad_w_pred, self.config.vocab_size, d);
        outer_product_accumulate(&scratch.pred_grad, h_out, lambda_pred, &mut grad_pred_view);

        // 2. grad_w_decode += lambda_recon * (recon_grad ⊗ h_out)
        let mut grad_decode_view = MatrixViewMut::new(&mut grads.grad_w_decode, d, d);
        outer_product_accumulate(&scratch.recon_grad, h_out, lambda_recon, &mut grad_decode_view);

        // 3. Vectorized delta_h1 calculation:
        scratch.delta_h1.fill(0.0f32);
        if lambda_pred > 0.0f32 {
            matvec_transposed_accumulate(&pred_view, &scratch.pred_grad, &mut scratch.delta_h1);
            if lambda_pred != 1.0f32 {
                axiom_core::tensor::vec_scale(&mut scratch.delta_h1, lambda_pred);
            }
        }
        if lambda_recon > 0.0f32 {
            if lambda_recon == 1.0f32 {
                matvec_transposed_accumulate(&decode_view, &scratch.recon_grad, &mut scratch.delta_h1);
            } else {
                for (r, &rg) in scratch.recon_grad.iter().enumerate() {
                    let scale = lambda_recon * rg;
                    if scale != 0.0f32 {
                        let row = decode_view.row(r);
                        vec_add_scaled(&mut scratch.delta_h1, row, scale);
                    }
                }
            }
        }
        if lambda_residual > 0.0f32 {
            let inv_d = 2.0f32 * lambda_residual / (d as f32);
            for (dh, (&ho, &hi)) in scratch.delta_h1.iter_mut().zip(h_out.iter().zip(h_in.iter())) {
                *dh += inv_d * (ho - hi);
            }
        }

        // 4. Expert backpropagation
        let top_experts = top2(&scratch.gate_scores);
        let mut expert_weights = [0.0f32; 2];
        topk_softmax(&top_experts, &mut expert_weights);

        let idx1 = top_experts[0].1;
        let idx2 = top_experts[1].1;
        let g1 = expert_weights[0];
        let g2 = expert_weights[1];

        // delta_s initialized to delta_h1 (from h_out = h_in + s_t + ...)
        vec_copy(&mut scratch.delta_s, &scratch.delta_h1);

        // Expert 1 backward: delta_exp1 = g1 * delta_h1
        vec_copy(&mut scratch.scratch_exp_grad, &scratch.delta_h1);
        axiom_core::tensor::vec_scale(&mut scratch.scratch_exp_grad, g1);
        self.experts[idx1].backward(
            state_curr,
            &scratch.expert_raw_ffn1,
            &scratch.expert_act_ffn1,
            &scratch.scratch_exp_grad,
            &mut scratch.scratch_delta_a,
            &mut scratch.scratch_delta_z,
            &mut grads.expert_grads[idx1],
            &mut scratch.delta_s,
        );

        // Expert 2 backward: delta_exp2 = g2 * delta_h1
        vec_copy(&mut scratch.scratch_exp_grad, &scratch.delta_h1);
        axiom_core::tensor::vec_scale(&mut scratch.scratch_exp_grad, g2);
        self.experts[idx2].backward(
            state_curr,
            &scratch.expert_raw_ffn2,
            &scratch.expert_act_ffn2,
            &scratch.scratch_exp_grad,
            &mut scratch.scratch_delta_a,
            &mut scratch.scratch_delta_z,
            &mut grads.expert_grads[idx2],
            &mut scratch.delta_s,
        );

        // Router gating gradients through softmax
        let dl_dg1 = dot(&scratch.delta_h1, &scratch.expert_out1);
        let dl_dg2 = dot(&scratch.delta_h1, &scratch.expert_out2);
        let delta_gate1 = g1 * (1.0 - g1) * dl_dg1 - g1 * g2 * dl_dg2;
        let delta_gate2 = g2 * (1.0 - g2) * dl_dg2 - g1 * g2 * dl_dg1;

        let mut grad_gate_view = MatrixViewMut::new(&mut grads.grad_w_gate, self.config.num_experts, d);
        let row_gate1 = grad_gate_view.row_mut(idx1);
        vec_add_scaled(row_gate1, state_curr, delta_gate1);
        let row_gate2 = grad_gate_view.row_mut(idx2);
        vec_add_scaled(row_gate2, state_curr, delta_gate2);

        let gate_view = MatrixView::new(&self.w_gate, self.config.num_experts, d);
        let w_gate1 = gate_view.row(idx1);
        let w_gate2 = gate_view.row(idx2);
        vec_add_scaled(&mut scratch.delta_s, w_gate1, delta_gate1);
        vec_add_scaled(&mut scratch.delta_s, w_gate2, delta_gate2);

        // 5. Recurrent cell backward: delta_us = delta_s * sigmoid'(raw_recurrent)
        sigmoid_grad(state_curr, &scratch.delta_s, &mut scratch.delta_us);

        // grad_w_s += delta_us ⊗ s_concat
        scratch.s_concat[..d].copy_from_slice(h_in);
        scratch.s_concat[d..2 * d].copy_from_slice(&scratch.m_recall);
        scratch.s_concat[2 * d..3 * d].copy_from_slice(state_prev);

        let mut grad_ws_view = MatrixViewMut::new(&mut grads.grad_w_s, d, 3 * d);
        outer_product_accumulate(&scratch.delta_us, &scratch.s_concat, 1.0, &mut grad_ws_view);

        // 6. Gradient into input embedding h_in:
        // From direct residual skip connection: delta_h1
        vec_copy(grad_h_in, &scratch.delta_h1);

        // Plus from recurrent input projection: W_s[:, 0..d]^T * delta_us
        matvec_transposed(&ws_view, &scratch.delta_us, &mut scratch.scratch_transposed);
        vec_add_scaled(grad_h_in, &scratch.scratch_transposed[..d], 1.0);

        // Plus from reconstruction target: -lambda_recon * recon_grad
        if lambda_recon > 0.0f32 {
            vec_add_scaled(grad_h_in, &scratch.recon_grad, -lambda_recon);
        }
        if lambda_residual > 0.0f32 {
            let inv_d = 2.0f32 * lambda_residual / (d as f32);
            for (gh, (&ho, &hi)) in grad_h_in.iter_mut().zip(h_out.iter().zip(h_in.iter())) {
                *gh -= inv_d * (ho - hi);
            }
        }
    }
}
