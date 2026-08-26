//! Multithreaded Batch Trainer for MNEME with AdamW and Sparse SGD.

use crate::mneme_model::{MnemeModel, MnemeModelGrads};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

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

pub struct MnemeAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub m_embed: ParamMoments,
    pub m_pos_embed: ParamMoments,
    pub m_head: ParamMoments,
}

impl MnemeAdamW {
    pub fn new(model: &MnemeModel, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            m_embed: ParamMoments::new(model.embeddings.len()),
            m_pos_embed: ParamMoments::new(model.pos_embeddings.len()),
            m_head: ParamMoments::new(model.head.len()),
        }
    }

    pub fn step(&mut self, model: &mut MnemeModel, grads: &mut MnemeModelGrads, current_lr: f32) {
        self.step += 1;
        let t = self.step as f32;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        let bc1 = 1.0f32 - beta1.powf(t);
        let bc2 = 1.0f32 - beta2.powf(t);
        let inv_bc1 = 1.0f32 / bc1;
        let inv_bc2 = 1.0f32 / bc2;

        let mut update_p = |p: &mut [f32], g: &[f32], mom: &mut ParamMoments| {
            for (((param, &grad), m_val), v_val) in p.iter_mut().zip(g.iter()).zip(mom.m.iter_mut()).zip(mom.v.iter_mut()) {
                *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
                *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;

                let m_hat = *m_val * inv_bc1;
                let v_hat = *v_val * inv_bc2;

                let step_val = m_hat / (v_hat.sqrt() + eps) + wd * *param;
                *param -= current_lr * step_val;
            }
        };

        update_p(&mut model.embeddings, &grads.grad_embed, &mut self.m_embed);
        update_p(&mut model.pos_embeddings, &grads.grad_pos_embed, &mut self.m_pos_embed);
        update_p(&mut model.head, &grads.grad_head, &mut self.m_head);

        // Simple SGD updates for block matrices
        for (block, b_grads) in model.unique_blocks.iter_mut().zip(grads.block_grads.iter_mut()) {
            for (p, &g) in block.delta_layer.wq.iter_mut().zip(b_grads.delta_grads.grad_wq.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.delta_layer.wk.iter_mut().zip(b_grads.delta_grads.grad_wk.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.delta_layer.wv.iter_mut().zip(b_grads.delta_grads.grad_wv.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.delta_layer.wo.iter_mut().zip(b_grads.delta_grads.grad_wo.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.delta_layer.w_alpha.iter_mut().zip(b_grads.delta_grads.grad_w_alpha.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.delta_layer.w_beta.iter_mut().zip(b_grads.delta_grads.grad_w_beta.iter()) { *p -= current_lr * g; }

            for (p, &g) in block.w1.iter_mut().zip(b_grads.grad_w1.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.w1u.iter_mut().zip(b_grads.grad_w1u.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.w2.iter_mut().zip(b_grads.grad_w2.iter()) { *p -= current_lr * g; }

            if let (Some(ref mut exp), Some(ref eg)) = (&mut block.expert_tier, &b_grads.expert_grads) {
                for (p, &g) in exp.wq.iter_mut().zip(eg.grad_wq.iter()) { *p -= current_lr * g; }
                for (p, &g) in exp.c1.iter_mut().zip(eg.grad_c1.iter()) { *p -= current_lr * g; }
                for (p, &g) in exp.c2.iter_mut().zip(eg.grad_c2.iter()) { *p -= current_lr * g; }
                for (p, &g) in exp.wo.iter_mut().zip(eg.grad_wo.iter()) { *p -= current_lr * g; }
                exp.apply_sparse_expert_updates(&eg.sparse_expert_grads, current_lr * 2.0);
            }
        }
    }
}

/// Evaluate MNEME validation loss and BPC.
pub fn evaluate_mneme_bpc(
    model: &mut MnemeModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
    passes_r: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let mut dummy_grads = MnemeModelGrads::new(
        model.vocab_size,
        model.d_model,
        model.max_seq_len,
        &model.unique_blocks,
        passes_r,
    );

    for _ in 0..num_batches {
        let start = rng.gen_range(0..=max_start);
        let seq = &val_data[start..start + seq_len + 1];
        let x_seq: Vec<usize> = seq[..seq_len].iter().map(|&b| b as usize).collect();
        let y_seq: Vec<usize> = seq[1..seq_len + 1].iter().map(|&b| b as usize).collect();

        dummy_grads.zero();
        let loss = model.forward_backward_sequence(&x_seq, &y_seq, passes_r, &mut dummy_grads);
        total_loss += loss;
        total_tokens += seq_len;
    }

    let mean_loss = if total_tokens > 0 { total_loss / total_tokens as f32 } else { 0.0f32 };
    let bpc = mean_loss / std::f32::consts::LN_2;
    (mean_loss, bpc)
}

/// Train MNEME model multithreaded across CPU cores.
pub fn train_mneme(
    model: &mut MnemeModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
    stochastic_r: bool,
    label: &str,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = MnemeAdamW::new(model, base_lr);
    let start_time = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();
    let max_start = train_data.len().saturating_sub(seq_len + 1);
    let fixed_r = model.config.n_passes;

    let (p_tot, p_act, bytes_tok, l3_res) = model.parameter_metrics();
    println!(
        "Training {} | Total P: {:.2}M, Active P: {:.2}M | DRAM/tok: {} B | L3 Core: {:.2} MB | Steps: {}",
        label, p_tot as f32 / 1e6, p_act as f32 / 1e6, bytes_tok, l3_res as f32 / 1e6, max_steps
    );

    for step in 1..=max_steps {
        let elapsed_sec = start_time.elapsed().as_secs_f64();
        if elapsed_sec >= max_time_secs as f64 {
            println!("\nReached max time limit of {}s at step {}", max_time_secs, step);
            break;
        }

        // Stochastic R selection (HARDPOINT §2.1)
        let passes_r = if stochastic_r {
            master_rng.gen_range(1..=fixed_r)
        } else {
            fixed_r
        };

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

        let model_ref = model.clone();
        let thread_results: Vec<(f32, MnemeModelGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut local_model = model_ref.clone();
                let mut grads = MnemeModelGrads::new(
                    local_model.vocab_size,
                    local_model.d_model,
                    local_model.max_seq_len,
                    &local_model.unique_blocks,
                    passes_r,
                );
                let loss = local_model.forward_backward_sequence(&x_seq, &y_seq, passes_r, &mut grads);
                (loss, grads)
            })
            .collect();

        let mut total_grads = MnemeModelGrads::new(
            model.vocab_size,
            model.d_model,
            model.max_seq_len,
            &model.unique_blocks,
            passes_r,
        );
        let mut total_loss = 0.0f32;
        let scale = 1.0f32 / (batch_size * seq_len) as f32;

        for (loss, g) in thread_results {
            total_loss += loss;
            total_grads.add(&g);
        }

        axiom_core::tensor::vec_scale(&mut total_grads.grad_embed, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_pos_embed, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_head, scale);
        for bg in &mut total_grads.block_grads {
            axiom_core::tensor::vec_scale(&mut bg.delta_grads.grad_wq, scale);
            axiom_core::tensor::vec_scale(&mut bg.delta_grads.grad_wk, scale);
            axiom_core::tensor::vec_scale(&mut bg.delta_grads.grad_wv, scale);
            axiom_core::tensor::vec_scale(&mut bg.delta_grads.grad_wo, scale);
            axiom_core::tensor::vec_scale(&mut bg.delta_grads.grad_w_alpha, scale);
            axiom_core::tensor::vec_scale(&mut bg.delta_grads.grad_w_beta, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_w1, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_w1u, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_w2, scale);
            if let Some(ref mut eg) = bg.expert_grads {
                axiom_core::tensor::vec_scale(&mut eg.grad_wq, scale);
                axiom_core::tensor::vec_scale(&mut eg.grad_c1, scale);
                axiom_core::tensor::vec_scale(&mut eg.grad_c2, scale);
                axiom_core::tensor::vec_scale(&mut eg.grad_wo, scale);
                for (_, ref mut vg) in &mut eg.sparse_expert_grads {
                    axiom_core::tensor::vec_scale(vg, scale);
                }
            }
        }

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
            let (val_loss, val_bpc) = evaluate_mneme_bpc(model, val_data, 10, seq_len, fixed_r);
            let elapsed = start_time.elapsed().as_secs_f64();
            let tok_s = (step * batch_size * seq_len) as f64 / elapsed;

            println!(
                "[{}] Step {:>4} ({:>5.1}s) | Train Loss: {:.4} | Val Loss: {:.4} | Val BPC: {:.4} | LR: {:.2e} | Tok/s: {:.0}",
                label, step, elapsed, mean_train_loss, val_loss, val_bpc, current_lr, tok_s
            );

            history.push((step, val_loss, val_bpc, elapsed));
        }
    }

    history
}
