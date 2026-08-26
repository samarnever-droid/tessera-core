//! STRATUM Multi-Threaded Batch Trainer with AdamW for Dense Parameters and Sparse SGD for Memory Slots.

use crate::stratum_model::{StratumModel, StratumModelGrads};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

/// Parameter moments for AdamW optimizer.
#[derive(Debug, Clone)]
pub struct ParamMoments {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParamMoments {
    pub fn new(len: usize) -> Self {
        Self {
            m: vec![0.0f32; len],
            v: vec![0.0f32; len],
        }
    }
}

/// AdamW optimizer for STRATUM dense parameters.
pub struct StratumAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub step: usize,
    pub m_embed: ParamMoments,
    pub m_pos_embed: ParamMoments,
    pub m_delta_wq: ParamMoments,
    pub m_delta_wk: ParamMoments,
    pub m_delta_wv: ParamMoments,
    pub m_delta_wo: ParamMoments,
    pub m_delta_alpha: ParamMoments,
    pub m_delta_beta: ParamMoments,
    pub m_pkm_wq: ParamMoments,
    pub m_pkm_wout: ParamMoments,
    pub m_pkm_k1: ParamMoments,
    pub m_pkm_k2: ParamMoments,
    pub m_head: ParamMoments,
}

impl StratumAdamW {
    pub fn new(model: &StratumModel, lr: f32) -> Self {
        let d = model.d_model;
        let d_v = model.pkm_layer.d_v;
        let m = model.pkm_layer.m;
        let d_half = model.pkm_layer.d_half;

        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            step: 0,
            m_embed: ParamMoments::new(model.embeddings.len()),
            m_pos_embed: ParamMoments::new(model.pos_embeddings.len()),
            m_delta_wq: ParamMoments::new(d * d),
            m_delta_wk: ParamMoments::new(d * d),
            m_delta_wv: ParamMoments::new(d * d),
            m_delta_wo: ParamMoments::new(d * d),
            m_delta_alpha: ParamMoments::new(d),
            m_delta_beta: ParamMoments::new(d),
            m_pkm_wq: ParamMoments::new(d * d),
            m_pkm_wout: ParamMoments::new(d * d_v),
            m_pkm_k1: ParamMoments::new(m * d_half),
            m_pkm_k2: ParamMoments::new(m * d_half),
            m_head: ParamMoments::new(model.head.len()),
        }
    }

    pub fn compute_grad_norm(&self, grads: &StratumModelGrads) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &grads.grad_embed { sum_sq += g * g; }
        for &g in &grads.grad_pos_embed { sum_sq += g * g; }
        for &g in &grads.delta_grads.grad_wq { sum_sq += g * g; }
        for &g in &grads.delta_grads.grad_wk { sum_sq += g * g; }
        for &g in &grads.delta_grads.grad_wv { sum_sq += g * g; }
        for &g in &grads.delta_grads.grad_wo { sum_sq += g * g; }
        for &g in &grads.pkm_grads.grad_wq { sum_sq += g * g; }
        for &g in &grads.pkm_grads.grad_w_out { sum_sq += g * g; }
        for &g in &grads.pkm_grads.grad_keys1 { sum_sq += g * g; }
        for &g in &grads.pkm_grads.grad_keys2 { sum_sq += g * g; }
        for &g in &grads.grad_head { sum_sq += g * g; }
        sum_sq.sqrt()
    }

    pub fn step(&mut self, model: &mut StratumModel, grads: &mut StratumModelGrads, current_lr: f32) {
        self.step += 1;
        let t = self.step as f32;

        let grad_norm = self.compute_grad_norm(grads);
        let clip_scale = if grad_norm > self.max_grad_norm && grad_norm > 1e-8 {
            self.max_grad_norm / grad_norm
        } else {
            1.0f32
        };

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        let bc1 = 1.0f32 - beta1.powf(t);
        let bc2 = 1.0f32 - beta2.powf(t);
        let inv_bc1 = 1.0f32 / bc1;
        let inv_bc2 = 1.0f32 / bc2;

        let update_p = |p: &mut [f32], g: &[f32], mom: &mut ParamMoments| {
            for (((param, &grad), m_val), v_val) in p.iter_mut().zip(g.iter()).zip(mom.m.iter_mut()).zip(mom.v.iter_mut()) {
                let scaled_g = grad * clip_scale;
                *m_val = beta1 * *m_val + (1.0 - beta1) * scaled_g;
                *v_val = beta2 * *v_val + (1.0 - beta2) * scaled_g * scaled_g;

                let m_hat = *m_val * inv_bc1;
                let v_hat = *v_val * inv_bc2;

                let step_val = m_hat / (v_hat.sqrt() + eps) + wd * *param;
                *param -= current_lr * step_val;
            }
        };

        // Dense AdamW updates
        update_p(&mut model.embeddings, &grads.grad_embed, &mut self.m_embed);
        update_p(&mut model.pos_embeddings, &grads.grad_pos_embed, &mut self.m_pos_embed);

        update_p(&mut model.delta_layer.wq, &grads.delta_grads.grad_wq, &mut self.m_delta_wq);
        update_p(&mut model.delta_layer.wk, &grads.delta_grads.grad_wk, &mut self.m_delta_wk);
        update_p(&mut model.delta_layer.wv, &grads.delta_grads.grad_wv, &mut self.m_delta_wv);
        update_p(&mut model.delta_layer.wo, &grads.delta_grads.grad_wo, &mut self.m_delta_wo);
        update_p(&mut model.delta_layer.w_alpha, &grads.delta_grads.grad_w_alpha, &mut self.m_delta_alpha);
        update_p(&mut model.delta_layer.w_beta, &grads.delta_grads.grad_w_beta, &mut self.m_delta_beta);

        update_p(&mut model.pkm_layer.wq, &grads.pkm_grads.grad_wq, &mut self.m_pkm_wq);
        update_p(&mut model.pkm_layer.w_out, &grads.pkm_grads.grad_w_out, &mut self.m_pkm_wout);
        update_p(&mut model.pkm_layer.keys1, &grads.pkm_grads.grad_keys1, &mut self.m_pkm_k1);
        update_p(&mut model.pkm_layer.keys2, &grads.pkm_grads.grad_keys2, &mut self.m_pkm_k2);

        update_p(&mut model.head, &grads.grad_head, &mut self.m_head);

        // Sparse SGD update for memory slots (§6.4)
        model.pkm_layer.apply_sparse_value_updates(&grads.pkm_grads.sparse_value_grads, current_lr * 2.0);
    }
}

/// Evaluate STRATUM model validation BPC.
pub fn evaluate_stratum_bpc(
    model: &mut StratumModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let mut dummy_grads = StratumModelGrads::new(
        model.vocab_size,
        model.d_model,
        model.max_seq_len,
        model.pkm_layer.d_v,
        model.pkm_layer.m,
    );

    for _ in 0..num_batches {
        let start = rng.gen_range(0..=max_start);
        let seq = &val_data[start..start + seq_len + 1];
        let x_seq: Vec<usize> = seq[..seq_len].iter().map(|&b| b as usize).collect();
        let y_seq: Vec<usize> = seq[1..seq_len + 1].iter().map(|&b| b as usize).collect();

        dummy_grads.zero();
        let loss = model.forward_backward_sequence(&x_seq, &y_seq, &mut dummy_grads);
        total_loss += loss;
        total_tokens += seq_len;
    }

    let mean_loss = if total_tokens > 0 { total_loss / total_tokens as f32 } else { 0.0f32 };
    let bpc = mean_loss / std::f32::consts::LN_2;
    (mean_loss, bpc)
}

/// Train STRATUM model on CPU using multithreading with Rayon.
pub fn train_stratum(
    model: &mut StratumModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = StratumAdamW::new(model, base_lr);
    let start_time = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();
    let max_start = train_data.len().saturating_sub(seq_len + 1);

    let (p_tot, p_act, _) = model.parameter_metrics();
    println!(
        "Training STRATUM (N={} slots) | Total P: {:.2}M, Active P: {:.2}M | Batch: {}, Seq: {}, Max Time: {}s",
        model.pkm_layer.total_slots, p_tot as f32 / 1e6, p_act as f32 / 1e6, batch_size, seq_len, max_time_secs
    );

    for step in 1..=max_steps {
        let elapsed_sec = start_time.elapsed().as_secs_f64();
        if elapsed_sec >= max_time_secs as f64 {
            println!("\nReached max time limit of {}s at step {}", max_time_secs, step);
            break;
        }

        // Sample batch
        let mut batch_x = Vec::with_capacity(batch_size);
        let mut batch_y = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let start = master_rng.gen_range(0..=max_start);
            let seq = &train_data[start..start + seq_len + 1];
            let x: Vec<usize> = seq[..seq_len].iter().map(|&b| b as usize).collect();
            let y: Vec<usize> = seq[1..seq_len + 1].iter().map(|&b| b as usize).collect();
            batch_x.push(x);
            batch_y.push(y);
        }

        // Parallel forward + backward across batch
        let model_ref = model.clone();
        let thread_results: Vec<(f32, StratumModelGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut local_model = model_ref.clone();
                let mut grads = StratumModelGrads::new(
                    local_model.vocab_size,
                    local_model.d_model,
                    local_model.max_seq_len,
                    local_model.pkm_layer.d_v,
                    local_model.pkm_layer.m,
                );
                let loss = local_model.forward_backward_sequence(&x_seq, &y_seq, &mut grads);
                (loss, grads)
            })
            .collect();

        // Aggregate gradients
        let mut total_grads = StratumModelGrads::new(
            model.vocab_size,
            model.d_model,
            model.max_seq_len,
            model.pkm_layer.d_v,
            model.pkm_layer.m,
        );
        let mut total_loss = 0.0f32;
        let scale = 1.0f32 / (batch_size * seq_len) as f32;

        for (loss, g) in thread_results {
            total_loss += loss;
            total_grads.add(&g);
        }

        axiom_core::tensor::vec_scale(&mut total_grads.grad_embed, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_pos_embed, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.delta_grads.grad_wq, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.delta_grads.grad_wk, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.delta_grads.grad_wv, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.delta_grads.grad_wo, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.delta_grads.grad_w_alpha, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.delta_grads.grad_w_beta, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.pkm_grads.grad_wq, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.pkm_grads.grad_w_out, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.pkm_grads.grad_keys1, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.pkm_grads.grad_keys2, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_head, scale);
        for (_, ref mut val_g) in &mut total_grads.pkm_grads.sparse_value_grads {
            axiom_core::tensor::vec_scale(val_g, scale);
        }

        // LR schedule
        let warmup = 50;
        let current_lr = if step < warmup {
            let alpha = step as f32 / warmup as f32;
            1e-4 + alpha * (base_lr - 1e-4)
        } else {
            let prog = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
            1e-4 + 0.5 * (1.0 + (PI * prog.min(1.0)).cos()) * (base_lr - 1e-4)
        };

        optimizer.step(model, &mut total_grads, current_lr);

        if step % 25 == 0 || step == 1 {
            let mean_train_loss = total_loss * scale;
            let (val_loss, val_bpc) = evaluate_stratum_bpc(model, val_data, 10, seq_len);
            let elapsed = start_time.elapsed().as_secs_f64();
            let tok_s = (step * batch_size * seq_len) as f64 / elapsed;

            println!(
                "Step {:>4} ({:>5.1}s) | Train Loss: {:.4} | Val Loss: {:.4} | Val BPC: {:.4} | LR: {:.2e} | Tok/s: {:.0}",
                step, elapsed, mean_train_loss, val_loss, val_bpc, current_lr, tok_s
            );

            history.push((step, val_loss, val_bpc, elapsed));
        }
    }

    history
}
