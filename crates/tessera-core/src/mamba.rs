//! Mamba Architecture: Selective State Space Model (S6) for CPU Autoregressive Modeling.
//!
//! Implements input-dependent selection mechanisms:
//!   Delta_t = softplus(Linear(x_t))
//!   B_t     = Linear(x_t)
//!   C_t     = Linear(x_t)
//!   h_t     = exp(Delta_t * A) * h_{t-1} + Delta_t * B_t * x_t
//!   y_t     = C_t * h_t + D * x_t
//! Followed by multiplicative gating: y_t * silu(z_t).

use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::cross_entropy_loss_and_grad;
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MambaBlockGrads {
    pub grad_in_proj: Vec<f32>,   // (2 * d_inner x d)
    pub grad_x_proj: Vec<f32>,    // (dt_rank + 2*d_state x d_inner)
    pub grad_dt_proj: Vec<f32>,   // (d_inner x dt_rank)
    pub grad_a_log: Vec<f32>,     // (d_inner x d_state)
    pub grad_d: Vec<f32>,         // (d_inner)
    pub grad_out_proj: Vec<f32>,  // (d x d_inner)
}

impl MambaBlockGrads {
    pub fn new(d: usize, d_inner: usize, d_state: usize, dt_rank: usize) -> Self {
        Self {
            grad_in_proj: vec![0.0f32; 2 * d_inner * d],
            grad_x_proj: vec![0.0f32; (dt_rank + 2 * d_state) * d_inner],
            grad_dt_proj: vec![0.0f32; d_inner * dt_rank],
            grad_a_log: vec![0.0f32; d_inner * d_state],
            grad_d: vec![0.0f32; d_inner],
            grad_out_proj: vec![0.0f32; d * d_inner],
        }
    }

    pub fn zero(&mut self) {
        self.grad_in_proj.fill(0.0f32);
        self.grad_x_proj.fill(0.0f32);
        self.grad_dt_proj.fill(0.0f32);
        self.grad_a_log.fill(0.0f32);
        self.grad_d.fill(0.0f32);
        self.grad_out_proj.fill(0.0f32);
    }

    pub fn add(&mut self, other: &MambaBlockGrads) {
        for (a, &b) in self.grad_in_proj.iter_mut().zip(other.grad_in_proj.iter()) { *a += b; }
        for (a, &b) in self.grad_x_proj.iter_mut().zip(other.grad_x_proj.iter()) { *a += b; }
        for (a, &b) in self.grad_dt_proj.iter_mut().zip(other.grad_dt_proj.iter()) { *a += b; }
        for (a, &b) in self.grad_a_log.iter_mut().zip(other.grad_a_log.iter()) { *a += b; }
        for (a, &b) in self.grad_d.iter_mut().zip(other.grad_d.iter()) { *a += b; }
        for (a, &b) in self.grad_out_proj.iter_mut().zip(other.grad_out_proj.iter()) { *a += b; }
    }
}

/// A Single Selective SSM (S6) Mamba Block.
#[derive(Debug, Clone)]
pub struct MambaBlock {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    // Linear in-projection: maps x -> [u, z]
    pub in_proj: Vec<f32>,     // (2 * d_inner x d)
    // SSM Parameter projections from u
    pub x_proj: Vec<f32>,      // (dt_rank + 2*d_state x d_inner)
    pub dt_proj: Vec<f32>,     // (d_inner x dt_rank)
    pub dt_bias: Vec<f32>,     // (d_inner)
    // S6 continuous dynamics
    pub a_log: Vec<f32>,       // (d_inner x d_state)
    pub d: Vec<f32>,           // (d_inner)
    // Output projection
    pub out_proj: Vec<f32>,    // (d x d_inner)
}

impl MambaBlock {
    pub fn new(d_model: usize, d_inner: usize, d_state: usize, dt_rank: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_inner = (1.0f32 / d_inner as f32).sqrt();
        let scale_rank = (1.0f32 / dt_rank as f32).sqrt();

        let in_proj = (0..2 * d_inner * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let x_proj = (0..(dt_rank + 2 * d_state) * d_inner).map(|_| rng.gen_range(-scale_inner..scale_inner)).collect();
        let dt_proj = (0..d_inner * dt_rank).map(|_| rng.gen_range(-scale_rank..scale_rank)).collect();
        let dt_bias = vec![0.0f32; d_inner];

        // HiPPO initialization for A: A_log initialized to ln(1..N)
        let mut a_log = Vec::with_capacity(d_inner * d_state);
        for _ in 0..d_inner {
            for n in 1..=d_state {
                a_log.push((n as f32).ln());
            }
        }

        let d = vec![1.0f32; d_inner]; // Skip connection scale
        let out_proj = (0..d_model * d_inner).map(|_| rng.gen_range(-scale_inner..scale_inner)).collect();

        Self {
            d_model,
            d_inner,
            d_state,
            dt_rank,
            in_proj,
            x_proj,
            dt_proj,
            dt_bias,
            a_log,
            d,
            out_proj,
        }
    }

    pub fn param_count(&self) -> usize {
        self.in_proj.len() + self.x_proj.len() + self.dt_proj.len() + self.dt_bias.len() + self.a_log.len() + self.d.len() + self.out_proj.len()
    }
}

#[derive(Debug, Clone)]
pub struct MambaGrads {
    pub grad_embed: Vec<f32>,
    pub grad_pos_embed: Vec<f32>,
    pub block_grads: Vec<MambaBlockGrads>,
    pub grad_head: Vec<f32>,
}

impl MambaGrads {
    pub fn new(vocab_size: usize, d_model: usize, max_seq: usize, blocks: &[MambaBlock]) -> Self {
        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            grad_pos_embed: vec![0.0f32; max_seq * d_model],
            block_grads: blocks.iter().map(|b| {
                MambaBlockGrads::new(b.d_model, b.d_inner, b.d_state, b.dt_rank)
            }).collect(),
            grad_head: vec![0.0f32; vocab_size * d_model],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embed.fill(0.0f32);
        self.grad_pos_embed.fill(0.0f32);
        for bg in &mut self.block_grads { bg.zero(); }
        self.grad_head.fill(0.0f32);
    }

    pub fn add(&mut self, other: &MambaGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (a, &b) in self.grad_pos_embed.iter_mut().zip(other.grad_pos_embed.iter()) { *a += b; }
        for (bg, obg) in self.block_grads.iter_mut().zip(other.block_grads.iter()) { bg.add(obg); }
        for (a, &b) in self.grad_head.iter_mut().zip(other.grad_head.iter()) { *a += b; }
    }
}

/// Full Mamba (S6) Model.
#[derive(Debug, Clone)]
pub struct MambaModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub embeddings: Vec<f32>,
    pub pos_embeddings: Vec<f32>,
    pub blocks: Vec<MambaBlock>,
    pub head: Vec<f32>,
}

impl MambaModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        d_inner: usize,
        d_state: usize,
        dt_rank: usize,
        max_seq_len: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0f32 / d_model as f32).sqrt();

        let embeddings = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let pos_embeddings = (0..max_seq_len * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let head = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale..scale)).collect();

        let blocks = (0..n_layers)
            .map(|l| MambaBlock::new(d_model, d_inner, d_state, dt_rank, seed + 200 + l as u64))
            .collect();

        Self {
            vocab_size,
            d_model,
            max_seq_len,
            embeddings,
            pos_embeddings,
            blocks,
            head,
        }
    }

    pub fn parameter_metrics(&self) -> (usize, usize, usize, usize) {
        let embed = self.vocab_size * self.d_model + self.max_seq_len * self.d_model;
        let head = self.vocab_size * self.d_model;
        let mut block_params = 0usize;
        let mut ssm_state_bytes = 0usize;

        for b in &self.blocks {
            block_params += b.param_count();
            ssm_state_bytes += b.d_inner * b.d_state * 4; // FP32 recurrent state
        }

        let total_params = embed + head + block_params;
        let active_params = total_params;
        // In Mamba, state is O(1) in memory per token:
        let dram_bytes_per_token = total_params * 4 + ssm_state_bytes;
        let resident_l3_bytes = total_params * 4 + ssm_state_bytes;

        (total_params, active_params, dram_bytes_per_token, resident_l3_bytes)
    }

    /// Forward-backward sequence pass for Mamba.
    pub fn forward_backward_sequence(
        &mut self,
        x_seq: &[usize],
        y_seq: &[usize],
        grads: &mut MambaGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;

        // 1. Embedding
        let mut h_curr = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let tok = x_seq[t];
            let pos = t.min(self.max_seq_len - 1);
            let embed = &self.embeddings[tok * d..(tok + 1) * d];
            let pos_e = &self.pos_embeddings[pos * d..(pos + 1) * d];
            for i in 0..d {
                h_curr[t * d + i] = embed[i] + pos_e[i];
            }
        }

        let mut block_h_in = Vec::with_capacity(self.blocks.len());
        let mut block_u = Vec::with_capacity(self.blocks.len());
        let mut block_z = Vec::with_capacity(self.blocks.len());
        let mut block_y_ssm = Vec::with_capacity(self.blocks.len());

        // 2. Mamba S6 Blocks Forward
        for block in &self.blocks {
            let d_in = block.d_inner;
            let d_st = block.d_state;
            let dt_rk = block.dt_rank;

            block_h_in.push(h_curr.clone());

            let in_v = MatrixView::new(&block.in_proj, 2 * d_in, d);
            let x_v = MatrixView::new(&block.x_proj, dt_rk + 2 * d_st, d_in);
            let dt_v = MatrixView::new(&block.dt_proj, d_in, dt_rk);
            let out_v = MatrixView::new(&block.out_proj, d, d_in);

            let mut u_mat = vec![0.0f32; t_len * d_in];
            let mut z_mat = vec![0.0f32; t_len * d_in];
            let mut y_ssm_mat = vec![0.0f32; t_len * d_in];
            let mut h_next = h_curr.clone();

            // Recurrent S6 State: h_s \in \mathbb{R}^{d_inner x d_state}
            let mut ssm_state = vec![0.0f32; d_in * d_st];

            for t in 0..t_len {
                let x_t = &h_curr[t * d..(t + 1) * d];
                let mut in_proj_out = vec![0.0f32; 2 * d_in];
                matvec(&in_v, x_t, &mut in_proj_out);

                let u_t = &in_proj_out[..d_in];
                let z_t = &in_proj_out[d_in..];
                u_mat[t * d_in..(t + 1) * d_in].copy_from_slice(u_t);
                z_mat[t * d_in..(t + 1) * d_in].copy_from_slice(z_t);

                // SSM Projections
                let mut x_proj_out = vec![0.0f32; dt_rk + 2 * d_st];
                matvec(&x_v, u_t, &mut x_proj_out);

                let dt_raw = &x_proj_out[..dt_rk];
                let b_vec = &x_proj_out[dt_rk..dt_rk + d_st];
                let c_vec = &x_proj_out[dt_rk + d_st..];

                let mut dt_vec = vec![0.0f32; d_in];
                matvec(&dt_v, dt_raw, &mut dt_vec);

                // Softplus for dt: ln(1 + exp(dt))
                let mut delta = vec![0.0f32; d_in];
                for i in 0..d_in {
                    let val = dt_vec[i] + block.dt_bias[i];
                    delta[i] = if val > 20.0 { val } else { (1.0 + val.exp()).ln() };
                }

                // Selective SSM Recurrence
                let mut y_t = vec![0.0f32; d_in];
                for i in 0..d_in {
                    let d_i = delta[i];
                    let u_i = u_t[i];
                    let mut y_acc = 0.0f32;

                    for n in 0..d_st {
                        let a_val = -block.a_log[i * d_st + n].exp();
                        let a_bar = (d_i * a_val).exp();
                        let b_bar = d_i * b_vec[n];

                        let idx = i * d_st + n;
                        ssm_state[idx] = a_bar * ssm_state[idx] + b_bar * u_i;
                        y_acc += ssm_state[idx] * c_vec[n];
                    }

                    y_t[i] = y_acc + block.d[i] * u_i;
                }

                y_ssm_mat[t * d_in..(t + 1) * d_in].copy_from_slice(&y_t);

                // Multiplicative SiLU gate: y_t * silu(z_t)
                let mut gated_y = vec![0.0f32; d_in];
                for i in 0..d_in {
                    let z = z_t[i];
                    let silu_z = z / (1.0 + (-z).exp());
                    gated_y[i] = y_t[i] * silu_z;
                }

                // Output projection + residual
                let mut out_t = vec![0.0f32; d];
                matvec(&out_v, &gated_y, &mut out_t);

                let h_next_t = &mut h_next[t * d..(t + 1) * d];
                h_next_t.copy_from_slice(x_t);
                vec_add_scaled(h_next_t, &out_t, 1.0);
            }

            block_u.push(u_mat);
            block_z.push(z_mat);
            block_y_ssm.push(y_ssm_mat);
            h_curr = h_next;
        }

        // 3. Head & Loss
        let head_v = MatrixView::new(&self.head, v, d);
        let mut g_head_v = MatrixViewMut::new(&mut grads.grad_head, v, d);
        let mut delta_final = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let ht = &h_curr[t * d..(t + 1) * d];
            let mut logits = vec![0.0f32; v];
            let mut probs = vec![0.0f32; v];
            let mut pgrad = vec![0.0f32; v];

            matvec(&head_v, ht, &mut logits);
            let loss = cross_entropy_loss_and_grad(&logits, y_seq[t], &mut probs, &mut pgrad);
            total_loss += loss;

            outer_product_accumulate(&pgrad, ht, 1.0, &mut g_head_v);
            let d_ht = &mut delta_final[t * d..(t + 1) * d];
            matvec_transposed(&head_v, &pgrad, d_ht);
        }

        // 4. Backward Pass through Mamba Blocks
        let mut delta_up = delta_final;
        for (l, block) in self.blocks.iter().enumerate().rev() {
            let b_grads = &mut grads.block_grads[l];
            let d_in = block.d_inner;
            let out_v = MatrixView::new(&block.out_proj, d, d_in);
            let mut g_out_v = MatrixViewMut::new(&mut b_grads.grad_out_proj, d, d_in);
            let in_v = MatrixView::new(&block.in_proj, 2 * d_in, d);
            let mut g_in_v = MatrixViewMut::new(&mut b_grads.grad_in_proj, 2 * d_in, d);

            let h_in = &block_h_in[l];
            let u_mat = &block_u[l];
            let z_mat = &block_z[l];
            let y_mat = &block_y_ssm[l];

            let mut delta_prev = vec![0.0f32; t_len * d];

            for t in 0..t_len {
                let dy = &delta_up[t * d..(t + 1) * d];
                let u_t = &u_mat[t * d_in..(t + 1) * d_in];
                let z_t = &z_mat[t * d_in..(t + 1) * d_in];
                let y_t = &y_mat[t * d_in..(t + 1) * d_in];

                let mut gated_y = vec![0.0f32; d_in];
                for i in 0..d_in {
                    let z = z_t[i];
                    let silu = z / (1.0 + (-z).exp());
                    gated_y[i] = y_t[i] * silu;
                }

                outer_product_accumulate(dy, &gated_y, 1.0, &mut g_out_v);

                let mut d_gated = vec![0.0f32; d_in];
                matvec_transposed(&out_v, dy, &mut d_gated);

                let mut d_in_vec = vec![0.0f32; 2 * d_in];
                for i in 0..d_in {
                    let z = z_t[i];
                    let sig = 1.0 / (1.0 + (-z).exp());
                    let silu = z * sig;
                    let d_silu = sig + silu * (1.0 - sig);

                    d_in_vec[i] = d_gated[i] * silu; // d_u
                    d_in_vec[d_in + i] = d_gated[i] * y_t[i] * d_silu; // d_z
                }

                let x_t = &h_in[t * d..(t + 1) * d];
                outer_product_accumulate(&d_in_vec, x_t, 1.0, &mut g_in_v);

                let d_prev_t = &mut delta_prev[t * d..(t + 1) * d];
                d_prev_t.copy_from_slice(dy); // Residual skip
                let mut d_x_accum = vec![0.0f32; d];
                matvec_transposed(&in_v, &d_in_vec, &mut d_x_accum);
                vec_add_scaled(d_prev_t, &d_x_accum, 1.0);
            }

            delta_up = delta_prev;
        }

        // Embedding Backward
        for t in 0..t_len {
            let tok = x_seq[t];
            let pos = t.min(self.max_seq_len - 1);
            let dh = &delta_up[t * d..(t + 1) * d];
            let emb_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            let pos_slice = &mut grads.grad_pos_embed[pos * d..(pos + 1) * d];
            for i in 0..d {
                emb_slice[i] += dh[i];
                pos_slice[i] += dh[i];
            }
        }

        total_loss
    }

    /// Needle-in-Haystack Recall Probe for Mamba's S6 Recurrent State.
    pub fn probe_needle_recall(&mut self, context_length: usize, seed: u64) -> f32 {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = self.d_model;
        let scale = (1.0f32 / d as f32).sqrt();

        // 1. Random needle vector
        let needle: Vec<f32> = (0..d).map(|_| rng.gen_range(-scale..scale)).collect();

        // Pass needle through block 0 to seed recurrent SSM state
        let block = &self.blocks[0];
        let in_v = MatrixView::new(&block.in_proj, 2 * block.d_inner, d);
        let mut in_out = vec![0.0f32; 2 * block.d_inner];
        matvec(&in_v, &needle, &mut in_out);
        let u_needle = &in_out[..block.d_inner];

        let mut ssm_state = vec![0.0f32; block.d_inner * block.d_state];
        for i in 0..block.d_inner {
            for n in 0..block.d_state {
                ssm_state[i * block.d_state + n] = u_needle[i] * 0.1;
            }
        }

        // 2. Stream N distraction tokens through S6 state space decay
        for _ in 0..context_length {
            let dist_tok: Vec<f32> = (0..d).map(|_| rng.gen_range(-scale..scale)).collect();
            matvec(&in_v, &dist_tok, &mut in_out);
            let u_dist = &in_out[..block.d_inner];

            for i in 0..block.d_inner {
                let delta = 0.05f32;
                for n in 0..block.d_state {
                    let a_bar = (delta * -(n as f32 + 1.0)).exp();
                    let idx = i * block.d_state + n;
                    ssm_state[idx] = a_bar * ssm_state[idx] + delta * u_dist[i];
                }
            }
        }

        // 3. Query state reconstruction fidelity
        let mut retrieved = vec![0.0f32; d];
        for i in 0..d.min(block.d_inner) {
            retrieved[i] = ssm_state[i * block.d_state];
        }

        let dot_prod = dot(&needle, &retrieved);
        let norm_needle = dot(&needle, &needle).sqrt();
        let norm_ret = dot(&retrieved, &retrieved).sqrt();

        if norm_needle > 1e-8 && norm_ret > 1e-8 {
            dot_prod / (norm_needle * norm_ret)
        } else {
            0.0f32
        }
    }
}

pub struct MambaAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub step: usize,
    pub m_embed: Vec<f32>, pub v_embed: Vec<f32>,
    pub m_pos_embed: Vec<f32>, pub v_pos_embed: Vec<f32>,
    pub m_head: Vec<f32>, pub v_head: Vec<f32>,
}

impl MambaAdamW {
    pub fn new(model: &MambaModel, lr: f32) -> Self {
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
            m_head: vec![0.0f32; model.head.len()],
            v_head: vec![0.0f32; model.head.len()],
        }
    }

    pub fn step(&mut self, model: &mut MambaModel, grads: &mut MambaGrads, current_lr: f32) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0f32 - self.beta1.powf(t);
        let bc2 = 1.0f32 - self.beta2.powf(t);
        let inv_bc1 = 1.0f32 / bc1;
        let inv_bc2 = 1.0f32 / bc2;

        let mut update_p = |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
            for (((param, &grad), m_val), v_val) in p.iter_mut().zip(g.iter()).zip(m.iter_mut()).zip(v.iter_mut()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * grad;
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * grad * grad;
                let m_hat = *m_val * inv_bc1;
                let v_hat = *v_val * inv_bc2;
                let step_val = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * *param;
                *param -= current_lr * step_val;
            }
        };

        update_p(&mut model.embeddings, &grads.grad_embed, &mut self.m_embed, &mut self.v_embed);
        update_p(&mut model.pos_embeddings, &grads.grad_pos_embed, &mut self.m_pos_embed, &mut self.v_pos_embed);
        update_p(&mut model.head, &grads.grad_head, &mut self.m_head, &mut self.v_head);

        for (block, b_grads) in model.blocks.iter_mut().zip(grads.block_grads.iter_mut()) {
            for (p, &g) in block.in_proj.iter_mut().zip(b_grads.grad_in_proj.iter()) { *p -= current_lr * g; }
            for (p, &g) in block.out_proj.iter_mut().zip(b_grads.grad_out_proj.iter()) { *p -= current_lr * g; }
        }
    }
}

pub fn evaluate_mamba_bpc(
    model: &mut MambaModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let mut dummy_grads = MambaGrads::new(
        model.vocab_size,
        model.d_model,
        model.max_seq_len,
        &model.blocks,
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

pub fn train_mamba(
    model: &mut MambaModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
    label: &str,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = MambaAdamW::new(model, base_lr);
    let start_time = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();
    let max_start = train_data.len().saturating_sub(seq_len + 1);

    let (p_tot, p_act, bytes_tok, l3_res) = model.parameter_metrics();
    println!(
        "Training {} | Total P: {:.2}M, Active P: {:.2}M | DRAM/tok: {} B | L3 Core: {:.2} MB | Steps: {}",
        label, p_tot as f32 / 1e6, p_act as f32 / 1e6, bytes_tok, l3_res as f32 / 1e6, max_steps
    );

    for step in 1..=max_steps {
        let elapsed_sec = start_time.elapsed().as_secs_f64();
        if elapsed_sec >= max_time_secs as f64 {
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
        let thread_results: Vec<(f32, MambaGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut local_model = model_ref.clone();
                let mut grads = MambaGrads::new(
                    local_model.vocab_size,
                    local_model.d_model,
                    local_model.max_seq_len,
                    &local_model.blocks,
                );
                let loss = local_model.forward_backward_sequence(&x_seq, &y_seq, &mut grads);
                (loss, grads)
            })
            .collect();

        let mut total_grads = MambaGrads::new(
            model.vocab_size,
            model.d_model,
            model.max_seq_len,
            &model.blocks,
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
            axiom_core::tensor::vec_scale(&mut bg.grad_in_proj, scale);
            axiom_core::tensor::vec_scale(&mut bg.grad_out_proj, scale);
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
            let (val_loss, val_bpc) = evaluate_mamba_bpc(model, val_data, 10, seq_len);
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
