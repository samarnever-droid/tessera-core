//! Multithreaded Batch Trainer for TESSERA with Dual-Head Multi-Token Prediction (MTP), Adaptive RoPE, Warmup-Stable-Decay (WSD), and AVX-Accelerated AdamW Optimization.
//! Implements 2024–2026 DeepSeek-V3 MTP dynamics & WSD scheduling for peak validation convergence.

use crate::tessera_model::{TesseraModel, TesseraModelGrads};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct StageMoments {
    pub m_n1: Vec<f32>, pub v_n1: Vec<f32>,
    pub m_conv: Vec<f32>, pub v_conv: Vec<f32>,
    pub m_gate_attn: Vec<f32>, pub v_gate_attn: Vec<f32>,
    pub m_lambda: Vec<f32>, pub v_lambda: Vec<f32>,
    pub m_vres_gate: Vec<f32>, pub v_vres_gate: Vec<f32>,
    pub m_eta: Vec<f32>, pub v_eta: Vec<f32>,
    pub m_wq: Vec<f32>, pub v_wq: Vec<f32>,
    pub m_wk: Vec<f32>, pub v_wk: Vec<f32>,
    pub m_wv: Vec<f32>, pub v_wv: Vec<f32>,
    pub m_wo: Vec<f32>, pub v_wo: Vec<f32>,
    pub m_n2: Vec<f32>, pub v_n2: Vec<f32>,
    pub m_w1: Vec<f32>, pub v_w1: Vec<f32>,
    pub m_w1u: Vec<f32>, pub v_w1u: Vec<f32>,
    pub m_w2: Vec<f32>, pub v_w2: Vec<f32>,
    pub m_ad_u: Vec<f32>, pub v_ad_u: Vec<f32>,
    pub m_ad_v: Vec<f32>, pub v_ad_v: Vec<f32>,
    pub m_mrm_wq: Option<Vec<f32>>, pub v_mrm_wq: Option<Vec<f32>>,
    pub m_mrm_wk: Option<Vec<f32>>, pub v_mrm_wk: Option<Vec<f32>>,
    pub m_mrm_wv: Option<Vec<f32>>, pub v_mrm_wv: Option<Vec<f32>>,
    pub m_mrm_wo: Option<Vec<f32>>, pub v_mrm_wo: Option<Vec<f32>>,
    pub m_mrm_wgate: Option<Vec<f32>>, pub v_mrm_wgate: Option<Vec<f32>>,
}

impl StageMoments {
    pub fn new(d: usize, d_ff: usize, r: usize, use_mrm: bool, eta_len: usize, lambda_len: usize) -> Self {
        Self {
            m_n1: vec![0.0f32; d], v_n1: vec![0.0f32; d],
            m_conv: vec![0.0f32; 4 * d], v_conv: vec![0.0f32; 4 * d],
            m_gate_attn: vec![0.0f32; d * d], v_gate_attn: vec![0.0f32; d * d],
            m_lambda: vec![0.0f32; lambda_len], v_lambda: vec![0.0f32; lambda_len],
            m_vres_gate: vec![0.0f32; 1], v_vres_gate: vec![0.0f32; 1],
            m_eta: vec![0.0f32; eta_len], v_eta: vec![0.0f32; eta_len],
            m_wq: vec![0.0f32; d * d], v_wq: vec![0.0f32; d * d],
            m_wk: vec![0.0f32; d * d], v_wk: vec![0.0f32; d * d],
            m_wv: vec![0.0f32; d * d], v_wv: vec![0.0f32; d * d],
            m_wo: vec![0.0f32; d * d], v_wo: vec![0.0f32; d * d],
            m_n2: vec![0.0f32; d], v_n2: vec![0.0f32; d],
            m_w1: vec![0.0f32; d_ff * d], v_w1: vec![0.0f32; d_ff * d],
            m_w1u: vec![0.0f32; d_ff * d], v_w1u: vec![0.0f32; d_ff * d],
            m_w2: vec![0.0f32; d * d_ff], v_w2: vec![0.0f32; d * d_ff],
            m_ad_u: vec![0.0f32; d * r], v_ad_u: vec![0.0f32; d * r],
            m_ad_v: vec![0.0f32; r * d], v_ad_v: vec![0.0f32; r * d],
            m_mrm_wq: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            v_mrm_wq: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            m_mrm_wk: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            v_mrm_wk: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            m_mrm_wv: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            v_mrm_wv: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            m_mrm_wo: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            v_mrm_wo: if use_mrm { Some(vec![0.0f32; d * d]) } else { None },
            m_mrm_wgate: if use_mrm { Some(vec![0.0f32; d]) } else { None },
            v_mrm_wgate: if use_mrm { Some(vec![0.0f32; d]) } else { None },
        }
    }
}

pub struct TesseraAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub step: usize,
    pub m_embed: Vec<f32>, pub v_embed: Vec<f32>,
    pub m_final_norm: Vec<f32>, pub v_final_norm: Vec<f32>,
    pub m_mtp_proj: Vec<f32>, pub v_mtp_proj: Vec<f32>,
    pub m_mtp_head: Vec<f32>, pub v_mtp_head: Vec<f32>,
    pub stage_moments: Vec<StageMoments>,
}

impl TesseraAdamW {
    pub fn new(model: &TesseraModel, lr: f32) -> Self {
        let stage_moments = model.stages.iter().map(|s| {
            StageMoments::new(s.d_model, s.d_ff, s.adapter_rank, s.mrm.is_some(), s.eta_rope.len(), s.lambda_diff.len())
        }).collect();

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
            m_final_norm: vec![0.0f32; model.final_norm_gamma.len()],
            v_final_norm: vec![0.0f32; model.final_norm_gamma.len()],
            m_mtp_proj: vec![0.0f32; model.w_mtp_proj.len()],
            v_mtp_proj: vec![0.0f32; model.w_mtp_proj.len()],
            m_mtp_head: vec![0.0f32; model.w_mtp_head.len()],
            v_mtp_head: vec![0.0f32; model.w_mtp_head.len()],
            stage_moments,
        }
    }

    pub fn compute_grad_norm(&self, grads: &TesseraModelGrads) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &grads.grad_embed { sum_sq += g * g; }
        for &g in &grads.grad_final_norm_gamma { sum_sq += g * g; }
        for &g in &grads.grad_w_mtp_proj { sum_sq += g * g; }
        for &g in &grads.grad_w_mtp_head { sum_sq += g * g; }
        for sg in &grads.stage_grads {
            for &g in &sg.grad_norm1_gamma { sum_sq += g * g; }
            for &g in &sg.grad_w_conv { sum_sq += g * g; }
            for &g in &sg.grad_w_gate_attn { sum_sq += g * g; }
            for &g in &sg.grad_lambda_diff { sum_sq += g * g; }
            for &g in &sg.grad_vres_gate { sum_sq += g * g; }
            for &g in &sg.grad_eta_rope { sum_sq += g * g; }
            for &g in &sg.grad_wq { sum_sq += g * g; }
            for &g in &sg.grad_wk { sum_sq += g * g; }
            for &g in &sg.grad_wv { sum_sq += g * g; }
            for &g in &sg.grad_wo { sum_sq += g * g; }
            for &g in &sg.grad_norm2_gamma { sum_sq += g * g; }
            for &g in &sg.grad_w1 { sum_sq += g * g; }
            for &g in &sg.grad_w1u { sum_sq += g * g; }
            for &g in &sg.grad_w2 { sum_sq += g * g; }
            for &g in &sg.grad_adapter_u { sum_sq += g * g; }
            for &g in &sg.grad_adapter_v { sum_sq += g * g; }
            if let Some(ref mg) = sg.mrm_grads {
                for &g in &mg.grad_wq { sum_sq += g * g; }
                for &g in &mg.grad_wk { sum_sq += g * g; }
                for &g in &mg.grad_wv { sum_sq += g * g; }
                for &g in &mg.grad_wo { sum_sq += g * g; }
                for &g in &mg.grad_wgate { sum_sq += g * g; }
            }
        }
        sum_sq.sqrt()
    }

    pub fn step(&mut self, model: &mut TesseraModel, grads: &mut TesseraModelGrads, current_lr: f32) {
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
        update_p(&mut model.final_norm_gamma, &grads.grad_final_norm_gamma, &mut self.m_final_norm, &mut self.v_final_norm);
        update_p(&mut model.w_mtp_proj, &grads.grad_w_mtp_proj, &mut self.m_mtp_proj, &mut self.v_mtp_proj);
        update_p(&mut model.w_mtp_head, &grads.grad_w_mtp_head, &mut self.m_mtp_head, &mut self.v_mtp_head);

        for ((stage, s_grads), sm) in model.stages.iter_mut().zip(grads.stage_grads.iter_mut()).zip(self.stage_moments.iter_mut()) {
            update_p(&mut stage.norm1_gamma, &s_grads.grad_norm1_gamma, &mut sm.m_n1, &mut sm.v_n1);
            update_p(&mut stage.w_conv, &s_grads.grad_w_conv, &mut sm.m_conv, &mut sm.v_conv);
            update_p(&mut stage.w_gate_attn, &s_grads.grad_w_gate_attn, &mut sm.m_gate_attn, &mut sm.v_gate_attn);
            update_p(&mut stage.lambda_diff, &s_grads.grad_lambda_diff, &mut sm.m_lambda, &mut sm.v_lambda);
            update_p(&mut stage.vres_gate, &s_grads.grad_vres_gate, &mut sm.m_vres_gate, &mut sm.v_vres_gate);
            update_p(&mut stage.eta_rope, &s_grads.grad_eta_rope, &mut sm.m_eta, &mut sm.v_eta);
            update_p(&mut stage.wq, &s_grads.grad_wq, &mut sm.m_wq, &mut sm.v_wq);
            update_p(&mut stage.wk, &s_grads.grad_wk, &mut sm.m_wk, &mut sm.v_wk);
            update_p(&mut stage.wv, &s_grads.grad_wv, &mut sm.m_wv, &mut sm.v_wv);
            update_p(&mut stage.wo, &s_grads.grad_wo, &mut sm.m_wo, &mut sm.v_wo);
            update_p(&mut stage.norm2_gamma, &s_grads.grad_norm2_gamma, &mut sm.m_n2, &mut sm.v_n2);
            update_p(&mut stage.w1, &s_grads.grad_w1, &mut sm.m_w1, &mut sm.v_w1);
            update_p(&mut stage.w1u, &s_grads.grad_w1u, &mut sm.m_w1u, &mut sm.v_w1u);
            update_p(&mut stage.w2, &s_grads.grad_w2, &mut sm.m_w2, &mut sm.v_w2);
            update_p(&mut stage.adapter_u, &s_grads.grad_adapter_u, &mut sm.m_ad_u, &mut sm.v_ad_u);
            update_p(&mut stage.adapter_v, &s_grads.grad_adapter_v, &mut sm.m_ad_v, &mut sm.v_ad_v);

            if let (Some(ref mut mrm), Some(ref s_mrm_grads), Some(ref mut m_wq), Some(ref mut v_wq), Some(ref mut m_wk), Some(ref mut v_wk), Some(ref mut m_wv), Some(ref mut v_wv), Some(ref mut m_wo), Some(ref mut v_wo), Some(ref mut m_wg), Some(ref mut v_wg)) = (
                &mut stage.mrm,
                &s_grads.mrm_grads,
                &mut sm.m_mrm_wq,
                &mut sm.v_mrm_wq,
                &mut sm.m_mrm_wk,
                &mut sm.v_mrm_wk,
                &mut sm.m_mrm_wv,
                &mut sm.v_mrm_wv,
                &mut sm.m_mrm_wo,
                &mut sm.v_mrm_wo,
                &mut sm.m_mrm_wgate,
                &mut sm.v_mrm_wgate,
            ) {
                update_p(&mut mrm.w_q, &s_mrm_grads.grad_wq, m_wq, v_wq);
                update_p(&mut mrm.w_k, &s_mrm_grads.grad_wk, m_wk, v_wk);
                update_p(&mut mrm.w_v, &s_mrm_grads.grad_wv, m_wv, v_wv);
                update_p(&mut mrm.w_o, &s_mrm_grads.grad_wo, m_wo, v_wo);
                update_p(&mut mrm.w_gate, &s_mrm_grads.grad_wgate, m_wg, v_wg);
            }
        }
    }
}

pub fn evaluate_tessera_bpc(
    model: &mut TesseraModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let mut dummy_grads = TesseraModelGrads::new(
        model.vocab_size,
        model.d_model,
        model.max_seq_len,
        &model.stages,
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

/// Multithreaded Batch Training for TESSERA with Warmup-Stable-Decay (WSD) & Multi-Token Prediction (MTP).
pub fn train_tessera(
    model: &mut TesseraModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
    label: &str,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = TesseraAdamW::new(model, base_lr);
    let start_time = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();
    let max_start = train_data.len().saturating_sub(seq_len + 1);

    let (p_tot, p_act, bytes_tok, param_bytes, mem_footprint_bytes) = model.parameter_metrics();
    println!(
        "Training {} | Total P: {:.2}M, Active P: {:.2}M | DRAM/tok: {} B | Param Weights: {:.2} MB | MRM-v2 Memory Footprint: {:.2} MB | Steps: {}",
        label, p_tot as f32 / 1e6, p_act as f32 / 1e6, bytes_tok, param_bytes as f32 / 1e6, mem_footprint_bytes as f32 / 1e6, max_steps
    );

    for step in 1..=max_steps {
        let elapsed_sec = start_time.elapsed().as_secs_f64();
        if elapsed_sec >= max_time_secs as f64 {
            println!("\nReached max time limit of {}s at step {}", max_time_secs, step);
            break;
        }

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
        let thread_results: Vec<(f32, TesseraModelGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut local_model = model_ref.clone();
                let mut grads = TesseraModelGrads::new(
                    local_model.vocab_size,
                    local_model.d_model,
                    local_model.max_seq_len,
                    &local_model.stages,
                );
                let loss = local_model.forward_backward_sequence(&x_seq, &y_seq, &mut grads);
                (loss, grads)
            })
            .collect();

        let mut total_grads = TesseraModelGrads::new(
            model.vocab_size,
            model.d_model,
            model.max_seq_len,
            &model.stages,
        );
        let mut total_loss = 0.0f32;
        let scale = 1.0f32 / (batch_size * seq_len) as f32;

        for (loss, g) in thread_results {
            total_loss += loss;
            total_grads.add(&g);
        }

        axiom_core::tensor::vec_scale(&mut total_grads.grad_embed, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_final_norm_gamma, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_w_mtp_proj, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_w_mtp_head, scale);
        for sg in &mut total_grads.stage_grads {
            axiom_core::tensor::vec_scale(&mut sg.grad_norm1_gamma, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_w_conv, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_w_gate_attn, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_lambda_diff, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_vres_gate, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_eta_rope, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_wq, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_wk, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_wv, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_wo, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_norm2_gamma, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_w1, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_w1u, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_w2, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_adapter_u, scale);
            axiom_core::tensor::vec_scale(&mut sg.grad_adapter_v, scale);
            if let Some(ref mut mg) = sg.mrm_grads {
                axiom_core::tensor::vec_scale(&mut mg.grad_wq, scale);
                axiom_core::tensor::vec_scale(&mut mg.grad_wk, scale);
                axiom_core::tensor::vec_scale(&mut mg.grad_wv, scale);
                axiom_core::tensor::vec_scale(&mut mg.grad_wo, scale);
                axiom_core::tensor::vec_scale(&mut mg.grad_wgate, scale);
            }
        }

        // Warmup-Stable-Decay (WSD): Warmup (1..10) -> Stable Plateau (11..90) -> Cooldown Annealing (91..120)
        let warmup_steps = 10;
        let decay_start = (max_steps as f32 * 0.75) as usize; // Step 90 out of 120
        let min_lr = 4.0e-4f32;
        let current_lr = if step <= warmup_steps {
            let alpha = step as f32 / warmup_steps as f32;
            min_lr + alpha * (base_lr - min_lr)
        } else if step <= decay_start {
            base_lr
        } else {
            let decay_prog = (step - decay_start) as f32 / (max_steps - decay_start).max(1) as f32;
            min_lr + 0.5 * (1.0 + (PI * decay_prog.min(1.0)).cos()) * (base_lr - min_lr)
        };

        optimizer.step(model, &mut total_grads, current_lr);

        if step % 25 == 0 || step == 1 || step == max_steps {
            let mean_train_loss = total_loss * scale;
            let (val_loss, val_bpc) = evaluate_tessera_bpc(model, val_data, 10, seq_len);
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
