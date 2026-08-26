//! Multi-threaded CPU batch trainer for standard 4-layer Transformer baseline.

use crate::transformer::{TransformerGrads, TransformerModel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

/// AdamW optimizer for Transformer baseline.
pub struct TransformerAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub step: usize,
    pub m_embed: Vec<f32>,
    pub v_embed: Vec<f32>,
    pub m_pos_embed: Vec<f32>,
    pub v_pos_embed: Vec<f32>,
    pub m_blocks_wq: Vec<Vec<f32>>,
    pub v_blocks_wq: Vec<Vec<f32>>,
    pub m_blocks_wk: Vec<Vec<f32>>,
    pub v_blocks_wk: Vec<Vec<f32>>,
    pub m_blocks_wv: Vec<Vec<f32>>,
    pub v_blocks_wv: Vec<Vec<f32>>,
    pub m_blocks_wo: Vec<Vec<f32>>,
    pub v_blocks_wo: Vec<Vec<f32>>,
    pub m_blocks_w1: Vec<Vec<f32>>,
    pub v_blocks_w1: Vec<Vec<f32>>,
    pub m_blocks_w2: Vec<Vec<f32>>,
    pub v_blocks_w2: Vec<Vec<f32>>,
    pub m_head: Vec<f32>,
    pub v_head: Vec<f32>,
}

impl TransformerAdamW {
    pub fn new(model: &TransformerModel, lr: f32) -> Self {
        let n = model.n_layers;
        let mut m_wq = Vec::with_capacity(n); let mut v_wq = Vec::with_capacity(n);
        let mut m_wk = Vec::with_capacity(n); let mut v_wk = Vec::with_capacity(n);
        let mut m_wv = Vec::with_capacity(n); let mut v_wv = Vec::with_capacity(n);
        let mut m_wo = Vec::with_capacity(n); let mut v_wo = Vec::with_capacity(n);
        let mut m_w1 = Vec::with_capacity(n); let mut v_w1 = Vec::with_capacity(n);
        let mut m_w2 = Vec::with_capacity(n); let mut v_w2 = Vec::with_capacity(n);

        for b in &model.blocks {
            m_wq.push(vec![0.0f32; b.wq.len()]); v_wq.push(vec![0.0f32; b.wq.len()]);
            m_wk.push(vec![0.0f32; b.wk.len()]); v_wk.push(vec![0.0f32; b.wk.len()]);
            m_wv.push(vec![0.0f32; b.wv.len()]); v_wv.push(vec![0.0f32; b.wv.len()]);
            m_wo.push(vec![0.0f32; b.wo.len()]); v_wo.push(vec![0.0f32; b.wo.len()]);
            m_w1.push(vec![0.0f32; b.w1.len()]); v_w1.push(vec![0.0f32; b.w1.len()]);
            m_w2.push(vec![0.0f32; b.w2.len()]); v_w2.push(vec![0.0f32; b.w2.len()]);
        }

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
            m_pos_embed: vec![0.0f32; model.pos_embeddings.len()],
            v_pos_embed: vec![0.0f32; model.pos_embeddings.len()],
            m_blocks_wq: m_wq, v_blocks_wq: v_wq,
            m_blocks_wk: m_wk, v_blocks_wk: v_wk,
            m_blocks_wv: m_wv, v_blocks_wv: v_wv,
            m_blocks_wo: m_wo, v_blocks_wo: v_wo,
            m_blocks_w1: m_w1, v_blocks_w1: v_w1,
            m_blocks_w2: m_w2, v_blocks_w2: v_w2,
            m_head: vec![0.0f32; model.head.len()],
            v_head: vec![0.0f32; model.head.len()],
        }
    }

    pub fn compute_grad_norm(&self, grads: &TransformerGrads) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &grads.grad_embed { sum_sq += g * g; }
        for &g in &grads.grad_pos_embed { sum_sq += g * g; }
        for bg in &grads.block_grads {
            for &g in &bg.grad_wq { sum_sq += g * g; }
            for &g in &bg.grad_wk { sum_sq += g * g; }
            for &g in &bg.grad_wv { sum_sq += g * g; }
            for &g in &bg.grad_wo { sum_sq += g * g; }
            for &g in &bg.grad_w1 { sum_sq += g * g; }
            for &g in &bg.grad_w2 { sum_sq += g * g; }
        }
        for &g in &grads.grad_head { sum_sq += g * g; }
        sum_sq.sqrt()
    }

    pub fn step(&mut self, model: &mut TransformerModel, grads: &mut TransformerGrads, current_lr: f32) {
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

        update_p(&mut model.embeddings, &grads.grad_embed, &mut self.m_embed, &mut self.v_embed);
        update_p(&mut model.pos_embeddings, &grads.grad_pos_embed, &mut self.m_pos_embed, &mut self.v_pos_embed);

        for l in 0..model.n_layers {
            let b = &mut model.blocks[l];
            let bg = &grads.block_grads[l];

            update_p(&mut b.wq, &bg.grad_wq, &mut self.m_blocks_wq[l], &mut self.v_blocks_wq[l]);
            update_p(&mut b.wk, &bg.grad_wk, &mut self.m_blocks_wk[l], &mut self.v_blocks_wk[l]);
            update_p(&mut b.wv, &bg.grad_wv, &mut self.m_blocks_wv[l], &mut self.v_blocks_wv[l]);
            update_p(&mut b.wo, &bg.grad_wo, &mut self.m_blocks_wo[l], &mut self.v_blocks_wo[l]);
            update_p(&mut b.w1, &bg.grad_w1, &mut self.m_blocks_w1[l], &mut self.v_blocks_w1[l]);
            update_p(&mut b.w2, &bg.grad_w2, &mut self.m_blocks_w2[l], &mut self.v_blocks_w2[l]);
        }

        update_p(&mut model.head, &grads.grad_head, &mut self.m_head, &mut self.v_head);
    }
}

/// Evaluate Transformer validation BPC.
pub fn evaluate_transformer_bpc(
    model: &TransformerModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let mut dummy_grads = TransformerGrads::new(model.vocab_size, model.d_model, model.max_seq_len, model.n_layers, model.d_ffn);

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

/// Train standard 4-layer Transformer with serial backprop on CPU.
pub fn train_transformer(
    model: &mut TransformerModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = TransformerAdamW::new(model, base_lr);
    let start_time = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();
    let max_start = train_data.len().saturating_sub(seq_len + 1);

    println!(
        "Starting Transformer Baseline Training (L={}) | Batch: {}, Seq: {}, Max Time: {}s",
        model.n_layers, batch_size, seq_len, max_time_secs
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

        // Parallel forward + backward across batch items
        let model_ref = &*model;
        let thread_results: Vec<(f32, TransformerGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut grads = TransformerGrads::new(
                    model_ref.vocab_size,
                    model_ref.d_model,
                    model_ref.max_seq_len,
                    model_ref.n_layers,
                    model_ref.d_ffn,
                );
                let loss = model_ref.forward_backward_sequence(&x_seq, &y_seq, &mut grads);
                (loss, grads)
            })
            .collect();

        // Aggregate
        let mut total_grads = TransformerGrads::new(
            model.vocab_size,
            model.d_model,
            model.max_seq_len,
            model.n_layers,
            model.d_ffn,
        );
        let mut total_loss = 0.0f32;
        let scale = 1.0f32 / (batch_size * seq_len) as f32;

        for (loss, g) in thread_results {
            total_loss += loss;
            total_grads.add(&g);
        }

        axiom_core::tensor::vec_scale(&mut total_grads.grad_embed, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_pos_embed, scale);
        for bg in &mut total_grads.block_grads {
            axiom_core::tensor::vec_scale(&mut bg.grad_wq, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_wk, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_wv, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_wo, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_w1, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_w2, scale);
        }
        axiom_core::tensor::vec_scale(&mut total_grads.grad_head, scale);

        // LR schedule
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
            let (val_loss, val_bpc) = evaluate_transformer_bpc(model, val_data, 10, seq_len);
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
