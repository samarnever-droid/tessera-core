//! Multi-layer stacked AXIOM model with decoupled per-layer parallel training,
//! detached representation passing, and O(1) memory inference.

use crate::layer::{AxiomLayer, LayerGrads, LayerScratch, SequenceCache};
use crate::{AxiomConfig, LayerState};
use axiom_core::activations::{mse_loss_and_grad, sigmoid};
use axiom_core::matvec::{matvec, matvec_transposed, matvec_transposed_accumulate, outer_product_accumulate};
use axiom_core::softmax::cross_entropy_loss_and_grad;
use axiom_core::tensor::{dot, vec_add_scaled, vec_copy, MatrixView, MatrixViewMut};
use axiom_core::topk::{top2, topk_softmax};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Multi-Layer Model Gradients containing decoupled per-layer gradient buffers.
#[derive(Debug, Clone)]
pub struct StackedModelGrads {
    pub grad_embeddings: Vec<f32>,
    pub grad_pos_embeddings: Vec<f32>,
    pub layer_grads: Vec<LayerGrads>,
}

impl StackedModelGrads {
    pub fn new(config: &AxiomConfig, max_seq_len: usize) -> Self {
        let mut layer_grads = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layer_grads.push(LayerGrads::new(config));
        }

        Self {
            grad_embeddings: vec![0.0f32; config.vocab_size * config.d_model],
            grad_pos_embeddings: vec![0.0f32; max_seq_len * config.d_model],
            layer_grads,
        }
    }

    pub fn zero(&mut self) {
        self.grad_embeddings.fill(0.0f32);
        self.grad_pos_embeddings.fill(0.0f32);
        for lg in &mut self.layer_grads {
            lg.zero();
        }
    }

    pub fn add(&mut self, other: &StackedModelGrads) {
        for (a, &b) in self.grad_embeddings.iter_mut().zip(other.grad_embeddings.iter()) {
            *a += b;
        }
        for (a, &b) in self.grad_pos_embeddings.iter_mut().zip(other.grad_pos_embeddings.iter()) {
            *a += b;
        }
        for (a_lg, b_lg) in self.layer_grads.iter_mut().zip(other.layer_grads.iter()) {
            a_lg.add(b_lg);
        }
    }
}

/// Full Stacked Multi-Layer AXIOM Sequence Model (§3.1, §3.3).
#[derive(Debug, Clone)]
pub struct AxiomModel {
    pub config: AxiomConfig,
    pub max_seq_len: usize,
    pub embeddings: Vec<f32>,     // (vocab_size x d_model)
    pub pos_embeddings: Vec<f32>, // (max_seq_len x d_model)
    pub layers: Vec<AxiomLayer>,  // num_layers independent AXIOM layers
}

impl AxiomModel {
    pub fn new(config: AxiomConfig, max_seq_len: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_embed = (1.0f32 / config.d_model as f32).sqrt();

        let embeddings: Vec<f32> = (0..config.vocab_size * config.d_model)
            .map(|_| rng.gen_range(-scale_embed..scale_embed))
            .collect();
        let pos_embeddings: Vec<f32> = (0..max_seq_len * config.d_model)
            .map(|_| rng.gen_range(-scale_embed..scale_embed))
            .collect();

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            layers.push(AxiomLayer::new(config.clone(), seed + 10 + (l as u64) * 100));
        }

        Self {
            config,
            max_seq_len,
            embeddings,
            pos_embeddings,
            layers,
        }
    }

    /// Embed token with positional encoding: h_0 = Embed(x_t) + PosEmbed(t)
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

    /// Fast O(1) Memory Multi-Layer Sequential Inference Forward Step (§5.1, §11.3).
    /// Executes sequentially through all L layers with zero dynamic allocation.
    #[inline]
    pub fn forward_infer_step(
        &self,
        token_x: usize,
        pos: usize,
        states: &mut [LayerState],
        scratches: &mut [LayerScratch],
        h_buffers: &mut [Vec<f32>], // L + 1 buffers of length d
        out_logits: &mut [f32],     // V logits from final layer
    ) {
        let num_layers = self.config.num_layers;
        debug_assert_eq!(states.len(), num_layers);
        debug_assert_eq!(scratches.len(), num_layers);
        debug_assert_eq!(h_buffers.len(), num_layers + 1);

        // 1. Initial embedding h_0
        self.embed_token_pos(token_x, pos, &mut h_buffers[0]);

        // 2. Sequential forward through layers: h_0 -> h_1 -> ... -> h_L
        for l in 0..num_layers {
            let (prev_buf, next_buf) = h_buffers.split_at_mut(l + 1);
            let h_in = &prev_buf[l];
            let h_out = &mut next_buf[0];

            self.layers[l].forward_infer(h_in, &mut states[l], &mut scratches[l], h_out);
        }

        // 3. Final layer prediction logits
        let last_layer = &self.layers[num_layers - 1];
        let pred_view = MatrixView::new(&last_layer.w_pred, self.config.vocab_size, self.config.d_model);
        matvec(&pred_view, &h_buffers[num_layers], out_logits);
    }

    /// Multi-Layer Forward Pass across sequence.
    /// Evaluates sequential forward pass across layers, caching detached representation
    /// sequence buffers H_l for each layer.
    /// Returns per-layer prediction loss: Vec<f32> of length L.
    pub fn forward_sequence_stacked(
        &self,
        x_seq: &[usize],
        y_seq: &[usize],
        states: &mut [LayerState],
        scratches: &mut [LayerScratch],
        caches: &mut [SequenceCache],
        layer_h_seqs: &mut [Vec<f32>], // L + 1 buffers of length (T * d)
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let ffn = self.config.d_ffn;
        let seq_len = x_seq.len();
        let num_layers = self.config.num_layers;

        let mut per_layer_losses = vec![0.0f32; num_layers];

        // 1. Embed full sequence into layer_h_seqs[0] (H_0)
        let mut h_token = vec![0.0f32; d];
        for t in 0..seq_len {
            self.embed_token_pos(x_seq[t], t, &mut h_token);
            layer_h_seqs[0][t * d..(t + 1) * d].copy_from_slice(&h_token);
        }

        // 2. Sequential forward pass across layers (L stages)
        for l in 0..num_layers {
            let layer = &self.layers[l];
            let ws_view = MatrixView::new(&layer.w_s, d, 3 * d);
            let gate_view = MatrixView::new(&layer.w_gate, self.config.num_experts, d);
            let pred_view = MatrixView::new(&layer.w_pred, v, d);
            let decode_view = MatrixView::new(&layer.w_decode, d, d);

            let state = &mut states[l];
            let scratch = &mut scratches[l];
            let cache = &mut caches[l];

            cache.s_history[..d].copy_from_slice(&state.recurrent_state);

            let (prev_h_seqs, next_h_seqs) = layer_h_seqs.split_at_mut(l + 1);
            let h_in_seq = &prev_h_seqs[l];
            let h_out_seq = &mut next_h_seqs[0];

            let mut h_out_token = vec![0.0f32; d];
            let mut layer_loss = 0.0f32;

            for t in 0..seq_len {
                let token_y = y_seq[t];
                let h_in_t = &h_in_seq[t * d..(t + 1) * d];

                // Cache h_in
                cache.h_in_history[t * d..(t + 1) * d].copy_from_slice(h_in_t);

                // Memory recall
                let m_t = &mut cache.m_recall_history[t * d..(t + 1) * d];
                state.memory.recall(h_in_t, m_t);

                // Recurrent update
                scratch.s_concat[..d].copy_from_slice(h_in_t);
                scratch.s_concat[d..2 * d].copy_from_slice(m_t);
                scratch.s_concat[2 * d..3 * d].copy_from_slice(&state.recurrent_state);

                matvec(&ws_view, &scratch.s_concat, &mut scratch.raw_recurrent);
                sigmoid(&scratch.raw_recurrent, &mut state.recurrent_state);

                let s_t = &mut cache.s_history[(t + 1) * d..(t + 2) * d];
                s_t.copy_from_slice(&state.recurrent_state);

                // Router
                matvec(&gate_view, &state.recurrent_state, &mut scratch.gate_scores);
                let top_experts = top2(&scratch.gate_scores);
                let mut expert_weights = [0.0f32; 2];
                topk_softmax(&top_experts, &mut expert_weights);

                let idx1 = top_experts[0].1;
                let idx2 = top_experts[1].1;
                cache.top_indices[t] = (idx1, idx2);
                cache.top_weights[t] = expert_weights;

                // Experts
                let raw1 = &mut cache.exp_raw1[t * ffn..(t + 1) * ffn];
                let act1 = &mut cache.exp_act1[t * ffn..(t + 1) * ffn];
                let out1 = &mut cache.exp_out1[t * d..(t + 1) * d];

                let raw2 = &mut cache.exp_raw2[t * ffn..(t + 1) * ffn];
                let act2 = &mut cache.exp_act2[t * ffn..(t + 1) * ffn];
                let out2 = &mut cache.exp_out2[t * d..(t + 1) * d];

                layer.experts[idx1].forward_with_cache(&state.recurrent_state, raw1, act1, out1);
                layer.experts[idx2].forward_with_cache(&state.recurrent_state, raw2, act2, out2);

                // Residual skip: h_{l+1} = h_l + s_t + g1*Exp1 + g2*Exp2
                h_out_token.copy_from_slice(h_in_t);
                vec_add_scaled(&mut h_out_token, &state.recurrent_state, 1.0);
                vec_add_scaled(&mut h_out_token, out1, expert_weights[0]);
                vec_add_scaled(&mut h_out_token, out2, expert_weights[1]);

                h_out_seq[t * d..(t + 1) * d].copy_from_slice(&h_out_token);
                cache.h_out_history[t * d..(t + 1) * d].copy_from_slice(&h_out_token);

                // Memory write
                state.memory.update(&h_out_token);

                // Local prediction head
                matvec(&pred_view, &h_out_token, &mut scratch.pred_logits);
                let pred_grad_t = &mut cache.pred_grad_history[t * v..(t + 1) * v];
                let loss_pred = cross_entropy_loss_and_grad(
                    &scratch.pred_logits,
                    token_y,
                    &mut scratch.pred_probs,
                    pred_grad_t,
                );
                layer_loss += loss_pred;

                // Reconstruction decoder
                matvec(&decode_view, &h_out_token, &mut scratch.recon_hidden);
                let recon_grad_t = &mut cache.recon_grad_history[t * d..(t + 1) * d];
                let _ = mse_loss_and_grad(&scratch.recon_hidden, h_in_t, recon_grad_t);
            }

            per_layer_losses[l] = layer_loss;
        }

        per_layer_losses
    }

    /// Decoupled Parallel Layer Backward Pass (§4.1, §4.3).
    /// Executes backward passes across all L layers in parallel across CPU threads with zero cross-layer gradients.
    pub fn backward_decoupled_parallel(
        &self,
        x_seq: &[usize],
        caches: &[SequenceCache],
        _scratches: &mut [LayerScratch],
        lambda_pred: f32,
        lambda_recon: f32,
        lambda_residual: f32,
        grads: &mut StackedModelGrads,
    ) {
        let d = self.config.d_model;
        let num_layers = self.config.num_layers;

        // Parallel backward pass across all L layers using Rayon
        let layer_refs: Vec<&AxiomLayer> = self.layers.iter().collect();
        let cache_refs: Vec<&SequenceCache> = caches.iter().collect();

        let layer_grads_results: Vec<(LayerGrads, Vec<f32>)> = (0..num_layers)
            .into_par_iter()
            .map(|l| {
                let layer = layer_refs[l];
                let cache = cache_refs[l];
                let mut scratch = LayerScratch::new(&layer.config);
                let mut layer_grad = LayerGrads::new(&layer.config);
                let mut input_grads = vec![0.0f32; x_seq.len() * d];

                let ws_view = MatrixView::new(&layer.w_s, d, 3 * d);
                let gate_view = MatrixView::new(&layer.w_gate, layer.config.num_experts, d);
                let pred_view = MatrixView::new(&layer.w_pred, layer.config.vocab_size, d);
                let decode_view = MatrixView::new(&layer.w_decode, d, d);

                let mut grad_ws_view = MatrixViewMut::new(&mut layer_grad.grad_w_s, d, 3 * d);
                let mut grad_gate_view = MatrixViewMut::new(&mut layer_grad.grad_w_gate, layer.config.num_experts, d);
                let mut grad_pred_view = MatrixViewMut::new(&mut layer_grad.grad_w_pred, layer.config.vocab_size, d);
                let mut grad_decode_view = MatrixViewMut::new(&mut layer_grad.grad_w_decode, d, d);

                let mut delta_s_temporal = vec![0.0f32; d];
                let mut delta_hin = vec![0.0f32; d];
                let ffn = layer.config.d_ffn;
                let v = layer.config.vocab_size;

                for t in (0..x_seq.len()).rev() {
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

                    // 1. Parameter gradients
                    if lambda_pred > 0.0f32 {
                        outer_product_accumulate(pred_grad_t, h_out_t, lambda_pred, &mut grad_pred_view);
                    }
                    if lambda_recon > 0.0f32 {
                        outer_product_accumulate(recon_grad_t, h_out_t, lambda_recon, &mut grad_decode_view);
                    }

                    // 2. delta_h_out
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

                    // 3. Experts backward
                    vec_copy(&mut scratch.delta_s, &scratch.delta_h1);

                    vec_copy(&mut scratch.scratch_exp_grad, &scratch.delta_h1);
                    axiom_core::tensor::vec_scale(&mut scratch.scratch_exp_grad, g1);
                    layer.experts[idx1].backward(
                        s_curr,
                        raw1,
                        act1,
                        &scratch.scratch_exp_grad,
                        &mut scratch.scratch_delta_a,
                        &mut scratch.scratch_delta_z,
                        &mut layer_grad.expert_grads[idx1],
                        &mut scratch.delta_s,
                    );

                    vec_copy(&mut scratch.scratch_exp_grad, &scratch.delta_h1);
                    axiom_core::tensor::vec_scale(&mut scratch.scratch_exp_grad, g2);
                    layer.experts[idx2].backward(
                        s_curr,
                        raw2,
                        act2,
                        &scratch.scratch_exp_grad,
                        &mut scratch.scratch_delta_a,
                        &mut scratch.scratch_delta_z,
                        &mut layer_grad.expert_grads[idx2],
                        &mut scratch.delta_s,
                    );

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

                    // 4. Temporal gradient incorporation (BPTT within layer)
                    for i in 0..d {
                        scratch.delta_s[i] += delta_s_temporal[i];
                    }

                    // 5. Sigmoid gradient
                    for i in 0..d {
                        scratch.delta_us[i] = scratch.delta_s[i] * s_curr[i] * (1.0 - s_curr[i]);
                    }

                    // 6. Accumulate W_s gradient
                    scratch.s_concat[..d].copy_from_slice(h_in_t);
                    scratch.s_concat[d..2 * d].copy_from_slice(m_t);
                    scratch.s_concat[2 * d..3 * d].copy_from_slice(s_prev);
                    outer_product_accumulate(&scratch.delta_us, &scratch.s_concat, 1.0, &mut grad_ws_view);

                    // 7. Layer input gradient (for Layer 0 embedding updates)
                    vec_copy(&mut delta_hin, &scratch.delta_h1);
                    matvec_transposed(&ws_view, &scratch.delta_us, &mut scratch.scratch_transposed);
                    vec_add_scaled(&mut delta_hin, &scratch.scratch_transposed[..d], 1.0);

                    if lambda_recon > 0.0f32 {
                        vec_add_scaled(&mut delta_hin, recon_grad_t, -lambda_recon);
                    }

                    input_grads[t * d..(t + 1) * d].copy_from_slice(&delta_hin);

                    // 8. Temporal step to s_{t-1}
                    vec_copy(&mut delta_s_temporal, &scratch.scratch_transposed[2 * d..3 * d]);
                }

                (layer_grad, input_grads)
            })
            .collect();

        // Assign decoupled layer gradients
        for (l, (lg, input_g)) in layer_grads_results.into_iter().enumerate() {
            grads.layer_grads[l].add(&lg);

            // Layer 0 connects to embeddings E and P
            if l == 0 {
                for t in 0..x_seq.len() {
                    let token_x = x_seq[t];
                    let d_hin = &input_g[t * d..(t + 1) * d];

                    let token_start = token_x * d;
                    let embed_slice = &mut grads.grad_embeddings[token_start..token_start + d];
                    for (eg, &gh) in embed_slice.iter_mut().zip(d_hin.iter()) {
                        *eg += gh;
                    }

                    if t < self.max_seq_len {
                        let pos_start = t * d;
                        let pos_slice = &mut grads.grad_pos_embeddings[pos_start..pos_start + d];
                        for (pg, &gh) in pos_slice.iter_mut().zip(d_hin.iter()) {
                            *pg += gh;
                        }
                    }
                }
            }
        }
    }
}
