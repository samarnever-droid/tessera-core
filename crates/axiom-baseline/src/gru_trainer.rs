//! Multi-threaded CPU batch trainer for GRU baseline model.

use crate::gru::{GruGrads, GruModel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

/// AdamW optimizer for GRU Baseline.
pub struct GruAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub step: usize,
    pub m_embed: Vec<f32>,
    pub v_embed: Vec<f32>,
    pub m_gate: Vec<f32>,
    pub v_gate: Vec<f32>,
    pub m_cand: Vec<f32>,
    pub v_cand: Vec<f32>,
    pub m_head: Vec<f32>,
    pub v_head: Vec<f32>,
}

impl GruAdamW {
    pub fn new(model: &GruModel, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            step: 0,
            m_embed: vec![0.0f32; model.embeddings.len()],
            v_embed: vec![0.0f32; model.embeddings.len()],
            m_gate: vec![0.0f32; model.w_gate.len()],
            v_gate: vec![0.0f32; model.w_gate.len()],
            m_cand: vec![0.0f32; model.w_cand.len()],
            v_cand: vec![0.0f32; model.w_cand.len()],
            m_head: vec![0.0f32; model.w_head.len()],
            v_head: vec![0.0f32; model.w_head.len()],
        }
    }

    pub fn compute_grad_norm(&self, grads: &GruGrads) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &grads.grad_embeddings {
            sum_sq += g * g;
        }
        for &g in &grads.grad_w_gate {
            sum_sq += g * g;
        }
        for &g in &grads.grad_w_cand {
            sum_sq += g * g;
        }
        for &g in &grads.grad_w_head {
            sum_sq += g * g;
        }
        sum_sq.sqrt()
    }

    pub fn step(&mut self, model: &mut GruModel, grads: &mut GruGrads, current_lr: f32) {
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

        let update_p = |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
            for (((param, &grad), m_val), v_val) in p.iter_mut().zip(g.iter()).zip(m.iter_mut()).zip(v.iter_mut()) {
                let scaled_g = grad * clip_scale;
                *m_val = beta1 * *m_val + (1.0 - beta1) * scaled_g;
                *v_val = beta2 * *v_val + (1.0 - beta2) * scaled_g * scaled_g;

                let m_hat = *m_val * inv_bc1;
                let v_hat = *v_val * inv_bc2;

                let step_val = m_hat / (v_hat.sqrt() + eps) + wd * *param;
                *param -= current_lr * step_val;
            }
        };

        update_p(&mut model.embeddings, &grads.grad_embeddings, &mut self.m_embed, &mut self.v_embed);
        update_p(&mut model.w_gate, &grads.grad_w_gate, &mut self.m_gate, &mut self.v_gate);
        update_p(&mut model.w_cand, &grads.grad_w_cand, &mut self.m_cand, &mut self.v_cand);
        update_p(&mut model.w_head, &grads.grad_w_head, &mut self.m_head, &mut self.v_head);
    }
}

/// Evaluate GRU validation loss and BPC.
pub fn evaluate_gru_bpc(
    model: &GruModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let d = model.d_model;
    let v = model.vocab_size;

    let mut h_cache = vec![0.0f32; (seq_len + 1) * d];
    let mut rz_cache = vec![0.0f32; seq_len * 2 * d];
    let mut cand_cache = vec![0.0f32; seq_len * d];
    let mut cand_in_cache = vec![0.0f32; seq_len * 2 * d];
    let mut logits_cache = vec![0.0f32; seq_len * v];
    let mut probs_cache = vec![0.0f32; seq_len * v];
    let mut pred_grad_cache = vec![0.0f32; seq_len * v];

    for _ in 0..num_batches {
        let start = rng.gen_range(0..=max_start);
        let seq = &val_data[start..start + seq_len + 1];
        let x_seq: Vec<usize> = seq[..seq_len].iter().map(|&b| b as usize).collect();
        let y_seq: Vec<usize> = seq[1..seq_len + 1].iter().map(|&b| b as usize).collect();

        let loss = model.forward_sequence(
            &x_seq,
            &y_seq,
            &mut h_cache,
            &mut rz_cache,
            &mut cand_cache,
            &mut cand_in_cache,
            &mut logits_cache,
            &mut probs_cache,
            &mut pred_grad_cache,
        );
        total_loss += loss;
        total_tokens += seq_len;
    }

    let mean_loss = if total_tokens > 0 {
        total_loss / total_tokens as f32
    } else {
        0.0f32
    };
    let bpc = mean_loss / std::f32::consts::LN_2;
    (mean_loss, bpc)
}

/// Train single-layer GRU with BPTT on CPU.
pub fn train_gru(
    model: &mut GruModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = GruAdamW::new(model, base_lr);
    let start_time = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();

    let max_start = train_data.len().saturating_sub(seq_len + 1);

    println!(
        "Starting GRU BPTT Control Training | Batch Size: {}, Seq Len: {}, Max Time: {}s",
        batch_size, seq_len, max_time_secs
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

        // Parallel forward + backward over batch
        let model_ref = &*model;
        let thread_results: Vec<(f32, GruGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let d = model_ref.d_model;
                let v = model_ref.vocab_size;
                let mut h_cache = vec![0.0f32; (seq_len + 1) * d];
                let mut rz_cache = vec![0.0f32; seq_len * 2 * d];
                let mut cand_cache = vec![0.0f32; seq_len * d];
                let mut cand_in_cache = vec![0.0f32; seq_len * 2 * d];
                let mut logits_cache = vec![0.0f32; seq_len * v];
                let mut probs_cache = vec![0.0f32; seq_len * v];
                let mut pred_grad_cache = vec![0.0f32; seq_len * v];

                let loss = model_ref.forward_sequence(
                    &x_seq,
                    &y_seq,
                    &mut h_cache,
                    &mut rz_cache,
                    &mut cand_cache,
                    &mut cand_in_cache,
                    &mut logits_cache,
                    &mut probs_cache,
                    &mut pred_grad_cache,
                );

                let mut grads = GruGrads::new(v, d);
                model_ref.backward_sequence(
                    &x_seq,
                    &h_cache,
                    &rz_cache,
                    &cand_cache,
                    &cand_in_cache,
                    &pred_grad_cache,
                    &mut grads,
                );

                (loss, grads)
            })
            .collect();

        // Aggregate
        let mut total_grads = GruGrads::new(model.vocab_size, model.d_model);
        let mut total_loss = 0.0f32;
        let scale = 1.0f32 / (batch_size * seq_len) as f32;

        for (loss, g) in thread_results {
            total_loss += loss;
            total_grads.add(&g);
        }

        axiom_core::tensor::vec_scale(&mut total_grads.grad_embeddings, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_w_gate, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_w_cand, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_w_head, scale);

        // LR cosine schedule
        let warmup = 100;
        let current_lr = if step < warmup {
            let alpha = step as f32 / warmup as f32;
            1e-4 + alpha * (base_lr - 1e-4)
        } else {
            let prog = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
            1e-4 + 0.5 * (1.0 + (PI * prog.min(1.0)).cos()) * (base_lr - 1e-4)
        };

        optimizer.step(model, &mut total_grads, current_lr);

        if step % 50 == 0 || step == 1 {
            let mean_train_loss = total_loss * scale;
            let (val_loss, val_bpc) = evaluate_gru_bpc(model, val_data, 10, seq_len);
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
