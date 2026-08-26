//! AXIOM Single-Layer Model container with token and positional embeddings,
//! associative memory, and Truncated BPTT within-layer autodiff.

use crate::layer::{AxiomLayer, LayerGrads, LayerScratch, SequenceCache};
use crate::{AxiomConfig, LayerState};
use axiom_core::activations::{mse_loss_and_grad, sigmoid};
use axiom_core::matvec::{matvec, matvec_transposed, matvec_transposed_accumulate, outer_product_accumulate};
use axiom_core::softmax::cross_entropy_loss_and_grad;
use axiom_core::tensor::{dot, vec_add_scaled, vec_copy, MatrixView, MatrixViewMut};
use axiom_core::topk::{top2, topk_softmax};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Full model gradients for a single-layer AXIOM model.
#[derive(Debug, Clone)]
pub struct ModelGrads {
    pub grad_embeddings: Vec<f32>,
    pub grad_pos_embeddings: Vec<f32>,
    pub layer_grads: LayerGrads,
}

impl ModelGrads {
    pub fn new(config: &AxiomConfig, max_seq_len: usize) -> Self {
        Self {
            grad_embeddings: vec![0.0f32; config.vocab_size * config.d_model],
            grad_pos_embeddings: vec![0.0f32; max_seq_len * config.d_model],
            layer_grads: LayerGrads::new(config),
        }
    }

    pub fn zero(&mut self) {
        self.grad_embeddings.fill(0.0f32);
        self.grad_pos_embeddings.fill(0.0f32);
        self.layer_grads.zero();
    }

    pub fn add(&mut self, other: &ModelGrads) {
        for (a, &b) in self.grad_embeddings.iter_mut().zip(other.grad_embeddings.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_pos_embeddings.iter_mut().zip(other.grad_pos_embeddings.iter()) {
            *a += b;
        }
        self.layer_grads.add(&other.layer_grads);
    }
}

/// Standalone Single-Layer AXIOM Sequence Model.
#[derive(Debug, Clone)]
pub struct AxiomSingleLayerModel {
    pub config: AxiomConfig,
    pub max_seq_len: usize,
    pub embeddings: Vec<f32>,     // (vocab_size x d_model)
    pub pos_embeddings: Vec<f32>, // (max_seq_len x d_model) - §3.3 Step 1
    pub layer: AxiomLayer,
}

impl AxiomSingleLayerModel {
    pub fn new(config: AxiomConfig, max_seq_len: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_embed = (1.0f32 / config.d_model as f32).sqrt();

        let embeddings: Vec<f32> = (0..config.vocab_size * config.d_model)
            .map(|_| rng.gen_range(-scale_embed..scale_embed))
            .collect();
        let pos_embeddings: Vec<f32> = (0..max_seq_len * config.d_model)
            .map(|_| rng.gen_range(-scale_embed..scale_embed))
            .collect();
        let layer = AxiomLayer::new(config.clone(), seed + 1);

        Self {
            config,
            max_seq_len,
            embeddings,
            pos_embeddings,
            layer,
        }
    }

    /// Embed a token with positional embedding: h_0 = Embed(x_t) + PosEmbed(t) (§3.3 Step 1)
    #[inline]
    pub fn embed_token_pos(&self, token: usize, pos: usize, out_h: &mut [f32]) {
        let d = self.config.d_model;
        debug_assert!(token < self.config.vocab_size);
        let token_start = token * d;
        out_h.copy_from_slice(&self.embeddings[token_start..token_start + d]);

        if pos < self.max_seq_len {
            let pos_start = pos * d;
            let pos_slice = &self.pos_embeddings[pos_start..pos_start + d];
            vec_add_scaled(out_h, pos_slice, 1.0);
        }
    }

    /// Single-token forward training step.
    #[inline]
    pub fn forward_train_step(
        &self,
        token_x: usize,
        pos: usize,
        target_y: usize,
        state: &mut LayerState,
        scratch: &mut LayerScratch,
        h_in: &mut [f32],
        h_out: &mut [f32],
    ) -> (f32, f32) {
        self.embed_token_pos(token_x, pos, h_in);
        self.layer.forward_train(h_in, target_y, state, scratch, h_out)
    }

    /// Single-token analytical backward pass.
    #[inline]
    pub fn backward_train_step(
        &self,
        token_x: usize,
        pos: usize,
        h_in: &[f32],
        state_prev: &[f32],
        state_curr: &[f32],
        h_out: &[f32],
        lambda_pred: f32,
        lambda_recon: f32,
        lambda_residual: f32,
        scratch: &mut LayerScratch,
        grads: &mut ModelGrads,
        grad_h_in: &mut [f32],
    ) {
        let d = self.config.d_model;
        self.layer.backward_layer(
            h_in,
            state_prev,
            state_curr,
            h_out,
            lambda_pred,
            lambda_recon,
            lambda_residual,
            scratch,
            &mut grads.layer_grads,
            grad_h_in,
        );

        let token_start = token_x * d;
        let embed_grad_slice = &mut grads.grad_embeddings[token_start..token_start + d];
        for (eg, &gh) in embed_grad_slice.iter_mut().zip(grad_h_in.iter()) {
            *eg += gh;
        }

        if pos < self.max_seq_len {
            let pos_start = pos * d;
            let pos_grad_slice = &mut grads.grad_pos_embeddings[pos_start..pos_start + d];
            for (pg, &gh) in pos_grad_slice.iter_mut().zip(grad_h_in.iter()) {
                *pg += gh;
            }
        }
    }

    /// Forward pass across a full sequence, caching all activations into `SequenceCache` for BPTT.
    pub fn forward_sequence_cache(
        &self,
        x_seq: &[usize],
        y_seq: &[usize],
        state: &mut LayerState,
        scratch: &mut LayerScratch,
        cache: &mut SequenceCache,
    ) -> (f32, f32) {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let ffn = self.config.d_ffn;
        let seq_len = x_seq.len();

        let ws_view = MatrixView::new(&self.layer.w_s, d, 3 * d);
        let gate_view = MatrixView::new(&self.layer.w_gate, self.config.num_experts, d);
        let pred_view = MatrixView::new(&self.layer.w_pred, v, d);
        let decode_view = MatrixView::new(&self.layer.w_decode, d, d);

        // Save initial state s_0
        cache.s_history[..d].copy_from_slice(&state.recurrent_state);

        let mut total_pred_loss = 0.0f32;
        let mut total_recon_loss = 0.0f32;

        let mut h_in = vec![0.0f32; d];
        let mut h_out = vec![0.0f32; d];

        for t in 0..seq_len {
            let token_x = x_seq[t];
            let token_y = y_seq[t];

            // 1. Embed input
            self.embed_token_pos(token_x, t, &mut h_in);
            let h_in_t = &mut cache.h_in_history[t * d..(t + 1) * d];
            h_in_t.copy_from_slice(&h_in);

            // 2. Associative Memory Recall
            let m_t = &mut cache.m_recall_history[t * d..(t + 1) * d];
            state.memory.recall(&h_in, m_t);

            // 3. Recurrent Update
            scratch.s_concat[..d].copy_from_slice(&h_in);
            scratch.s_concat[d..2 * d].copy_from_slice(m_t);
            scratch.s_concat[2 * d..3 * d].copy_from_slice(&state.recurrent_state);

            matvec(&ws_view, &scratch.s_concat, &mut scratch.raw_recurrent);
            sigmoid(&scratch.raw_recurrent, &mut state.recurrent_state);

            let s_t = &mut cache.s_history[(t + 1) * d..(t + 2) * d];
            s_t.copy_from_slice(&state.recurrent_state);

            // 4. Expert Routing
            matvec(&gate_view, &state.recurrent_state, &mut scratch.gate_scores);
            let top_experts = top2(&scratch.gate_scores);
            let mut expert_weights = [0.0f32; 2];
            topk_softmax(&top_experts, &mut expert_weights);

            let idx1 = top_experts[0].1;
            let idx2 = top_experts[1].1;
            cache.top_indices[t] = (idx1, idx2);
            cache.top_weights[t] = expert_weights;

            // 5. Sparse Expert Computation
            let raw1 = &mut cache.exp_raw1[t * ffn..(t + 1) * ffn];
            let act1 = &mut cache.exp_act1[t * ffn..(t + 1) * ffn];
            let out1 = &mut cache.exp_out1[t * d..(t + 1) * d];

            let raw2 = &mut cache.exp_raw2[t * ffn..(t + 1) * ffn];
            let act2 = &mut cache.exp_act2[t * ffn..(t + 1) * ffn];
            let out2 = &mut cache.exp_out2[t * d..(t + 1) * d];

            self.layer.experts[idx1].forward_with_cache(&state.recurrent_state, raw1, act1, out1);
            self.layer.experts[idx2].forward_with_cache(&state.recurrent_state, raw2, act2, out2);

            // Residual skip: h_out = h_in + s_t + g1*Exp1 + g2*Exp2
            vec_copy(&mut h_out, &h_in);
            vec_add_scaled(&mut h_out, &state.recurrent_state, 1.0);
            vec_add_scaled(&mut h_out, out1, expert_weights[0]);
            vec_add_scaled(&mut h_out, out2, expert_weights[1]);

            let h_out_t = &mut cache.h_out_history[t * d..(t + 1) * d];
            h_out_t.copy_from_slice(&h_out);

            // 6. Memory Write
            state.memory.update(&h_out);

            // 7. Prediction Head
            matvec(&pred_view, &h_out, &mut scratch.pred_logits);
            let pred_grad_t = &mut cache.pred_grad_history[t * v..(t + 1) * v];
            let loss_pred = cross_entropy_loss_and_grad(
                &scratch.pred_logits,
                token_y,
                &mut scratch.pred_probs,
                pred_grad_t,
            );
            total_pred_loss += loss_pred;

            // 8. Reconstruction Decoder
            matvec(&decode_view, &h_out, &mut scratch.recon_hidden);
            let recon_grad_t = &mut cache.recon_grad_history[t * d..(t + 1) * d];
            let loss_recon = mse_loss_and_grad(&scratch.recon_hidden, &h_in, recon_grad_t);
            total_recon_loss += loss_recon;
        }

        (total_pred_loss, total_recon_loss)
    }

    /// Truncated Backprop-Through-Time (BPTT) backward pass across the sequence.
    /// Gradients strictly stay inside Layer l (zero-cross-layer invariant).
    pub fn backward_sequence_bptt(
        &self,
        x_seq: &[usize],
        cache: &SequenceCache,
        lambda_pred: f32,
        lambda_recon: f32,
        lambda_residual: f32,
        scratch: &mut LayerScratch,
        grads: &mut ModelGrads,
    ) {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let ffn = self.config.d_ffn;
        let seq_len = x_seq.len();

        let ws_view = MatrixView::new(&self.layer.w_s, d, 3 * d);
        let gate_view = MatrixView::new(&self.layer.w_gate, self.config.num_experts, d);
        let pred_view = MatrixView::new(&self.layer.w_pred, v, d);
        let decode_view = MatrixView::new(&self.layer.w_decode, d, d);

        let mut grad_ws_view = MatrixViewMut::new(&mut grads.layer_grads.grad_w_s, d, 3 * d);
        let mut grad_gate_view = MatrixViewMut::new(&mut grads.layer_grads.grad_w_gate, self.config.num_experts, d);
        let mut grad_pred_view = MatrixViewMut::new(&mut grads.layer_grads.grad_w_pred, v, d);
        let mut grad_decode_view = MatrixViewMut::new(&mut grads.layer_grads.grad_w_decode, d, d);

        let mut delta_s_temporal = vec![0.0f32; d];
        let mut delta_hin = vec![0.0f32; d];

        // Backprop backwards through sequence from t = T-1 down to 0
        for t in (0..seq_len).rev() {
            let token_x = x_seq[t];
            let h_in_t = &cache.h_in_history[t * d..(t + 1) * d];
            let h_out_t = &cache.h_out_history[t * d..(t + 1) * d];
            let s_prev = &cache.s_history[t * d..(t + 1) * d];
            let s_curr = &cache.s_history[(t + 1) * d..(t + 2) * d];
            let m_t = &cache.m_recall_history[t * d..(t + 1) * d];
            let pred_grad_t = &cache.pred_grad_history[t * v..(t + 1) * v];
            let recon_grad_t = &cache.recon_grad_history[t * d..(t + 1) * d];

            let (idx1, idx2) = cache.top_indices[t];
            let [g1, g2] = cache.top_weights[t];

            let raw1 = &cache.exp_raw1[t * ffn..(t + 1) * ffn];
            let act1 = &cache.exp_act1[t * ffn..(t + 1) * ffn];
            let out1 = &cache.exp_out1[t * d..(t + 1) * d];

            let raw2 = &cache.exp_raw2[t * ffn..(t + 1) * ffn];
            let act2 = &cache.exp_act2[t * ffn..(t + 1) * ffn];
            let out2 = &cache.exp_out2[t * d..(t + 1) * d];

            // 1. Accumulate prediction & decode parameter gradients
            if lambda_pred > 0.0f32 {
                outer_product_accumulate(pred_grad_t, h_out_t, lambda_pred, &mut grad_pred_view);
            }
            if lambda_recon > 0.0f32 {
                outer_product_accumulate(recon_grad_t, h_out_t, lambda_recon, &mut grad_decode_view);
            }

            // 2. Compute delta_h_out
            scratch.delta_h1.fill(0.0f32);
            if lambda_pred > 0.0f32 {
                matvec_transposed_accumulate(&pred_view, pred_grad_t, &mut scratch.delta_h1);
                if lambda_pred != 1.0f32 {
                    axiom_core::tensor::vec_scale(&mut scratch.delta_h1, lambda_pred);
                }
            }
            if lambda_recon > 0.0f32 {
                for (r, &rg) in recon_grad_t.iter().enumerate() {
                    let scale = lambda_recon * rg;
                    if scale != 0.0f32 {
                        let row = decode_view.row(r);
                        vec_add_scaled(&mut scratch.delta_h1, row, scale);
                    }
                }
            }
            if lambda_residual > 0.0f32 {
                let inv_d = 2.0f32 * lambda_residual / (d as f32);
                for (dh, (&ho, &hi)) in scratch.delta_h1.iter_mut().zip(h_out_t.iter().zip(h_in_t.iter())) {
                    *dh += inv_d * (ho - hi);
                }
            }

            // 3. Backprop through active experts to get delta_s_local
            vec_copy(&mut scratch.delta_s, &scratch.delta_h1);

            // Expert 1
            vec_copy(&mut scratch.scratch_exp_grad, &scratch.delta_h1);
            axiom_core::tensor::vec_scale(&mut scratch.scratch_exp_grad, g1);
            self.layer.experts[idx1].backward(
                s_curr,
                raw1,
                act1,
                &scratch.scratch_exp_grad,
                &mut scratch.scratch_delta_a,
                &mut scratch.scratch_delta_z,
                &mut grads.layer_grads.expert_grads[idx1],
                &mut scratch.delta_s,
            );

            // Expert 2
            vec_copy(&mut scratch.scratch_exp_grad, &scratch.delta_h1);
            axiom_core::tensor::vec_scale(&mut scratch.scratch_exp_grad, g2);
            self.layer.experts[idx2].backward(
                s_curr,
                raw2,
                act2,
                &scratch.scratch_exp_grad,
                &mut scratch.scratch_delta_a,
                &mut scratch.scratch_delta_z,
                &mut grads.layer_grads.expert_grads[idx2],
                &mut scratch.delta_s,
            );

            // Router gating gradients
            let dl_dg1 = dot(&scratch.delta_h1, out1);
            let dl_dg2 = dot(&scratch.delta_h1, out2);
            let delta_gate1 = g1 * (1.0 - g1) * dl_dg1 - g1 * g2 * dl_dg2;
            let delta_gate2 = g2 * (1.0 - g2) * dl_dg2 - g1 * g2 * dl_dg1;

            let row_gate1 = grad_gate_view.row_mut(idx1);
            vec_add_scaled(row_gate1, s_curr, delta_gate1);
            let row_gate2 = grad_gate_view.row_mut(idx2);
            vec_add_scaled(row_gate2, s_curr, delta_gate2);

            let w_gate1 = gate_view.row(idx1);
            let w_gate2 = gate_view.row(idx2);
            vec_add_scaled(&mut scratch.delta_s, w_gate1, delta_gate1);
            vec_add_scaled(&mut scratch.delta_s, w_gate2, delta_gate2);

            // 4. Incorporate Temporal Gradient from future steps (BPTT recurrence):
            for i in 0..d {
                scratch.delta_s[i] += delta_s_temporal[i];
            }

            // 5. Backprop through sigmoid: delta_us = delta_s ⊙ s_curr ⊙ (1 - s_curr)
            for i in 0..d {
                scratch.delta_us[i] = scratch.delta_s[i] * s_curr[i] * (1.0 - s_curr[i]);
            }

            // 6. Accumulate W_s parameter gradient: delta_us ⊗ [h_in; m_t; s_{t-1}]
            scratch.s_concat[..d].copy_from_slice(h_in_t);
            scratch.s_concat[d..2 * d].copy_from_slice(m_t);
            scratch.s_concat[2 * d..3 * d].copy_from_slice(s_prev);
            outer_product_accumulate(&scratch.delta_us, &scratch.s_concat, 1.0, &mut grad_ws_view);

            // 7. Backprop into token and positional embeddings:
            // delta_hin = delta_h_out + W_s[:, 0..d]^T * delta_us - lambda_recon*recon_grad
            vec_copy(&mut delta_hin, &scratch.delta_h1);
            matvec_transposed(&ws_view, &scratch.delta_us, &mut scratch.scratch_transposed);
            vec_add_scaled(&mut delta_hin, &scratch.scratch_transposed[..d], 1.0);

            if lambda_recon > 0.0f32 {
                vec_add_scaled(&mut delta_hin, recon_grad_t, -lambda_recon);
            }
            if lambda_residual > 0.0f32 {
                let inv_d = 2.0f32 * lambda_residual / (d as f32);
                for (gh, (&ho, &hi)) in delta_hin.iter_mut().zip(h_out_t.iter().zip(h_in_t.iter())) {
                    *gh -= inv_d * (ho - hi);
                }
            }

            let token_start = token_x * d;
            let embed_grad_slice = &mut grads.grad_embeddings[token_start..token_start + d];
            for (eg, &gh) in embed_grad_slice.iter_mut().zip(delta_hin.iter()) {
                *eg += gh;
            }

            if t < self.max_seq_len {
                let pos_start = t * d;
                let pos_grad_slice = &mut grads.grad_pos_embeddings[pos_start..pos_start + d];
                for (pg, &gh) in pos_grad_slice.iter_mut().zip(delta_hin.iter()) {
                    *pg += gh;
                }
            }

            // 8. Compute temporal gradient for previous step s_{t-1}:
            // delta_s_{t-1}^temporal = W_s[:, 2d..3d]^T * delta_us
            vec_copy(&mut delta_s_temporal, &scratch.scratch_transposed[2 * d..3 * d]);
        }
    }

    /// Fast inference forward step.
    #[inline]
    pub fn forward_infer_step(
        &self,
        token_x: usize,
        pos: usize,
        state: &mut LayerState,
        scratch: &mut LayerScratch,
        h_in: &mut [f32],
        h_out: &mut [f32],
        out_logits: &mut [f32],
    ) {
        self.embed_token_pos(token_x, pos, h_in);
        self.layer.forward_infer(h_in, state, scratch, h_out);

        let pred_view = MatrixView::new(
            &self.layer.w_pred,
            self.config.vocab_size,
            self.config.d_model,
        );
        matvec(&pred_view, h_out, out_logits);
    }
}
