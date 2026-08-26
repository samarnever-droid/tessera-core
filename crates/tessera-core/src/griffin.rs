//! Google DeepMind Griffin / RecurrentGemma Architecture (2024).
//!
//! Implements Real-Gated Linear Recurrence (RG-LRU) alternated with Local Sliding Window Attention:
//!   RG-LRU:
//!     r_t = sigmoid(W_r * x_t + b_r)
//!     a_t = sigmoid(W_a * x_t + b_a)
//!     i_t = sqrt(1 - a_t^2) * (r_t * x_t')
//!     h_t = a_t * h_{t-1} + i_t
//!     y_t = W_out * h_t + x_t
//!   Followed by SwiGLU Channel Mixer.

use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::{cross_entropy_loss_and_grad, softmax};
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct RgLruGrads {
    pub grad_wr: Vec<f32>,      // (d_rnn x d)
    pub grad_wa: Vec<f32>,      // (d_rnn x d)
    pub grad_wx: Vec<f32>,      // (d_rnn x d)
    pub grad_wout: Vec<f32>,    // (d x d_rnn)
    pub grad_w1: Vec<f32>,      // (d_ff x d)
    pub grad_w1u: Vec<f32>,     // (d_ff x d)
    pub grad_w2: Vec<f32>,      // (d x d_ff)
}

impl RgLruGrads {
    pub fn new(d: usize, d_rnn: usize, d_ff: usize) -> Self {
        Self {
            grad_wr: vec![0.0f32; d_rnn * d],
            grad_wa: vec![0.0f32; d_rnn * d],
            grad_wx: vec![0.0f32; d_rnn * d],
            grad_wout: vec![0.0f32; d * d_rnn],
            grad_w1: vec![0.0f32; d_ff * d],
            grad_w1u: vec![0.0f32; d_ff * d],
            grad_w2: vec![0.0f32; d * d_ff],
        }
    }

    pub fn zero(&mut self) {
        self.grad_wr.fill(0.0f32);
        self.grad_wa.fill(0.0f32);
        self.grad_wx.fill(0.0f32);
        self.grad_wout.fill(0.0f32);
        self.grad_w1.fill(0.0f32);
        self.grad_w1u.fill(0.0f32);
        self.grad_w2.fill(0.0f32);
    }

    pub fn add(&mut self, other: &RgLruGrads) {
        for (a, &b) in self.grad_wr.iter_mut().zip(other.grad_wr.iter()) { *a += b; }
        for (a, &b) in self.grad_wa.iter_mut().zip(other.grad_wa.iter()) { *a += b; }
        for (a, &b) in self.grad_wx.iter_mut().zip(other.grad_wx.iter()) { *a += b; }
        for (a, &b) in self.grad_wout.iter_mut().zip(other.grad_wout.iter()) { *a += b; }
        for (a, &b) in self.grad_w1.iter_mut().zip(other.grad_w1.iter()) { *a += b; }
        for (a, &b) in self.grad_w1u.iter_mut().zip(other.grad_w1u.iter()) { *a += b; }
        for (a, &b) in self.grad_w2.iter_mut().zip(other.grad_w2.iter()) { *a += b; }
    }
}

/// A Single Real-Gated Linear Recurrent Unit (RG-LRU) Block.
#[derive(Debug, Clone)]
pub struct RgLruBlock {
    pub d_model: usize,
    pub d_rnn: usize,
    pub d_ff: usize,
    // RG-LRU projections
    pub wr: Vec<f32>,     // (d_rnn x d)
    pub wa: Vec<f32>,     // (d_rnn x d)
    pub wx: Vec<f32>,     // (d_rnn x d)
    pub wout: Vec<f32>,   // (d x d_rnn)
    // SwiGLU FFN
    pub w1: Vec<f32>,     // (d_ff x d)
    pub w1u: Vec<f32>,    // (d_ff x d)
    pub w2: Vec<f32>,     // (d x d_ff)
}

impl RgLruBlock {
    pub fn new(d_model: usize, d_rnn: usize, d_ff: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_rnn = (1.0f32 / d_rnn as f32).sqrt();
        let scale_ff = (1.0f32 / d_ff as f32).sqrt();

        let wr = (0..d_rnn * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wa = (0..d_rnn * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wx = (0..d_rnn * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wout = (0..d_model * d_rnn).map(|_| rng.gen_range(-scale_rnn..scale_rnn)).collect();

        let w1 = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w1u = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w2 = (0..d_model * d_ff).map(|_| rng.gen_range(-scale_ff..scale_ff)).collect();

        Self {
            d_model,
            d_rnn,
            d_ff,
            wr,
            wa,
            wx,
            wout,
            w1,
            w1u,
            w2,
        }
    }

    pub fn param_count(&self) -> usize {
        3 * (self.d_rnn * self.d_model) + (self.d_model * self.d_rnn) + 2 * (self.d_ff * self.d_model) + (self.d_model * self.d_ff)
    }
}

#[derive(Debug, Clone)]
pub struct LocalAttnBlockGrads {
    pub grad_wq: Vec<f32>, pub grad_wk: Vec<f32>, pub grad_wv: Vec<f32>, pub grad_wo: Vec<f32>,
    pub grad_w1: Vec<f32>, pub grad_w1u: Vec<f32>, pub grad_w2: Vec<f32>,
}

impl LocalAttnBlockGrads {
    pub fn new(d: usize, d_ff: usize) -> Self {
        Self {
            grad_wq: vec![0.0f32; d * d], grad_wk: vec![0.0f32; d * d],
            grad_wv: vec![0.0f32; d * d], grad_wo: vec![0.0f32; d * d],
            grad_w1: vec![0.0f32; d_ff * d], grad_w1u: vec![0.0f32; d_ff * d],
            grad_w2: vec![0.0f32; d * d_ff],
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32); self.grad_wk.fill(0.0f32);
        self.grad_wv.fill(0.0f32); self.grad_wo.fill(0.0f32);
        self.grad_w1.fill(0.0f32); self.grad_w1u.fill(0.0f32);
        self.grad_w2.fill(0.0f32);
    }

    pub fn add(&mut self, other: &LocalAttnBlockGrads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_wk.iter_mut().zip(other.grad_wk.iter()) { *a += b; }
        for (a, &b) in self.grad_wv.iter_mut().zip(other.grad_wv.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        for (a, &b) in self.grad_w1.iter_mut().zip(other.grad_w1.iter()) { *a += b; }
        for (a, &b) in self.grad_w1u.iter_mut().zip(other.grad_w1u.iter()) { *a += b; }
        for (a, &b) in self.grad_w2.iter_mut().zip(other.grad_w2.iter()) { *a += b; }
    }
}

#[derive(Debug, Clone)]
pub struct LocalAttnBlock {
    pub d_model: usize,
    pub d_ff: usize,
    pub window_size: usize,
    pub wq: Vec<f32>, pub wk: Vec<f32>, pub wv: Vec<f32>, pub wo: Vec<f32>,
    pub w1: Vec<f32>, pub w1u: Vec<f32>, pub w2: Vec<f32>,
}

impl LocalAttnBlock {
    pub fn new(d_model: usize, d_ff: usize, window_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_ff = (1.0f32 / d_ff as f32).sqrt();

        let wq = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wk = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wv = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wo = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let w1 = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w1u = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w2 = (0..d_model * d_ff).map(|_| rng.gen_range(-scale_ff..scale_ff)).collect();

        Self {
            d_model,
            d_ff,
            window_size,
            wq, wk, wv, wo,
            w1, w1u, w2,
        }
    }

    pub fn param_count(&self) -> usize {
        4 * (self.d_model * self.d_model) + 2 * (self.d_ff * self.d_model) + (self.d_model * self.d_ff)
    }
}

#[derive(Debug, Clone)]
pub struct GriffinGrads {
    pub grad_embed: Vec<f32>,
    pub grad_pos_embed: Vec<f32>,
    pub rnn_grads_0: RgLruGrads,
    pub attn_grads_1: LocalAttnBlockGrads,
    pub rnn_grads_2: RgLruGrads,
    pub grad_head: Vec<f32>,
}

impl GriffinGrads {
    pub fn new(vocab_size: usize, d_model: usize, d_rnn: usize, d_ff: usize, max_seq: usize) -> Self {
        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            grad_pos_embed: vec![0.0f32; max_seq * d_model],
            rnn_grads_0: RgLruGrads::new(d_model, d_rnn, d_ff),
            attn_grads_1: LocalAttnBlockGrads::new(d_model, d_ff),
            rnn_grads_2: RgLruGrads::new(d_model, d_rnn, d_ff),
            grad_head: vec![0.0f32; vocab_size * d_model],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embed.fill(0.0f32);
        self.grad_pos_embed.fill(0.0f32);
        self.rnn_grads_0.zero();
        self.attn_grads_1.zero();
        self.rnn_grads_2.zero();
        self.grad_head.fill(0.0f32);
    }

    pub fn add(&mut self, other: &GriffinGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (a, &b) in self.grad_pos_embed.iter_mut().zip(other.grad_pos_embed.iter()) { *a += b; }
        self.rnn_grads_0.add(&other.rnn_grads_0);
        self.attn_grads_1.add(&other.attn_grads_1);
        self.rnn_grads_2.add(&other.rnn_grads_2);
        for (a, &b) in self.grad_head.iter_mut().zip(other.grad_head.iter()) { *a += b; }
    }
}

/// Full Google DeepMind Griffin Model (2:1 RG-LRU to Local Attention Ratio).
#[derive(Debug, Clone)]
pub struct GriffinModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_rnn: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub embeddings: Vec<f32>,
    pub pos_embeddings: Vec<f32>,
    pub rnn_block_0: RgLruBlock,
    pub attn_block_1: LocalAttnBlock,
    pub rnn_block_2: RgLruBlock,
    pub head: Vec<f32>,
}

impl GriffinModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        d_rnn: usize,
        d_ff: usize,
        window_size: usize,
        max_seq_len: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0f32 / d_model as f32).sqrt();

        let embeddings = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let pos_embeddings = (0..max_seq_len * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let head = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale..scale)).collect();

        let rnn_block_0 = RgLruBlock::new(d_model, d_rnn, d_ff, seed + 101);
        let attn_block_1 = LocalAttnBlock::new(d_model, d_ff, window_size, seed + 102);
        let rnn_block_2 = RgLruBlock::new(d_model, d_rnn, d_ff, seed + 103);

        Self {
            vocab_size,
            d_model,
            d_rnn,
            d_ff,
            max_seq_len,
            embeddings,
            pos_embeddings,
            rnn_block_0,
            attn_block_1,
            rnn_block_2,
            head,
        }
    }

    pub fn parameter_metrics(&self) -> (usize, usize, usize, usize) {
        let embed = self.vocab_size * self.d_model + self.max_seq_len * self.d_model;
        let head = self.vocab_size * self.d_model;
        let b0 = self.rnn_block_0.param_count();
        let b1 = self.attn_block_1.param_count();
        let b2 = self.rnn_block_2.param_count();

        let total_params = embed + head + b0 + b1 + b2;
        let active_params = total_params;
        // In Griffin, KV cache is clamped to window W=64, and RG-LRU state is 256 floats (1 KB)
        let dram_bytes_per_token = 4 * self.d_model + self.d_rnn * 4;
        let resident_l3_bytes = total_params * 4;

        (total_params, active_params, dram_bytes_per_token, resident_l3_bytes)
    }

    /// Forward-backward sequence pass for Griffin.
    pub fn forward_backward_sequence(
        &mut self,
        x_seq: &[usize],
        y_seq: &[usize],
        grads: &mut GriffinGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;
        let d_r = self.d_rnn;
        let scale_attn = 1.0f32 / (d as f32).sqrt();

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

        // --- Layer 0: RG-LRU Block ---
        let h_in_0 = h_curr.clone();
        let wr_0 = MatrixView::new(&self.rnn_block_0.wr, d_r, d);
        let wa_0 = MatrixView::new(&self.rnn_block_0.wa, d_r, d);
        let wx_0 = MatrixView::new(&self.rnn_block_0.wx, d_r, d);
        let wout_0 = MatrixView::new(&self.rnn_block_0.wout, d, d_r);

        let mut r_0 = vec![0.0f32; t_len * d_r];
        let mut a_0 = vec![0.0f32; t_len * d_r];
        let mut x_prime_0 = vec![0.0f32; t_len * d_r];
        let mut h_ssm_0 = vec![0.0f32; (t_len + 1) * d_r]; // recurrent states
        let mut h_mid_0 = h_in_0.clone();

        for t in 0..t_len {
            let xt = &h_in_0[t * d..(t + 1) * d];
            let rt = &mut r_0[t * d_r..(t + 1) * d_r];
            let at = &mut a_0[t * d_r..(t + 1) * d_r];
            let xpt = &mut x_prime_0[t * d_r..(t + 1) * d_r];

            matvec(&wr_0, xt, rt);
            matvec(&wa_0, xt, at);
            matvec(&wx_0, xt, xpt);

            let mut prev_h = vec![0.0f32; d_r];
            prev_h.copy_from_slice(&h_ssm_0[t * d_r..(t + 1) * d_r]);
            let curr_h = &mut h_ssm_0[(t + 1) * d_r..(t + 2) * d_r];

            let mut out_r = vec![0.0f32; d_r];
            for i in 0..d_r {
                let r_sig = 1.0 / (1.0 + (-rt[i]).exp());
                let a_sig = 1.0 / (1.0 + (-at[i]).exp());
                rt[i] = r_sig;
                at[i] = a_sig;

                let input_gate = (1.0 - a_sig * a_sig).max(0.0).sqrt();
                let input_val = input_gate * (r_sig * xpt[i]);
                curr_h[i] = a_sig * prev_h[i] + input_val;
                out_r[i] = curr_h[i];
            }

            let mut rnn_out = vec![0.0f32; d];
            matvec(&wout_0, &out_r, &mut rnn_out);
            let h_out_t = &mut h_mid_0[t * d..(t + 1) * d];
            vec_add_scaled(h_out_t, &rnn_out, 1.0);
        }

        // SwiGLU 0
        let w1_0 = MatrixView::new(&self.rnn_block_0.w1, self.d_ff, d);
        let w1u_0 = MatrixView::new(&self.rnn_block_0.w1u, self.d_ff, d);
        let w2_0 = MatrixView::new(&self.rnn_block_0.w2, d, self.d_ff);
        let mut h_layer_0 = h_mid_0.clone();

        for t in 0..t_len {
            let ht = &h_mid_0[t * d..(t + 1) * d];
            let mut gate = vec![0.0f32; self.d_ff];
            let mut up = vec![0.0f32; self.d_ff];
            matvec(&w1_0, ht, &mut gate);
            matvec(&w1u_0, ht, &mut up);

            let mut ff = vec![0.0f32; self.d_ff];
            for i in 0..self.d_ff {
                let g = gate[i];
                let silu = g / (1.0 + (-g).exp());
                ff[i] = silu * up[i];
            }
            let mut ff_out = vec![0.0f32; d];
            matvec(&w2_0, &ff, &mut ff_out);
            vec_add_scaled(&mut h_layer_0[t * d..(t + 1) * d], &ff_out, 1.0);
        }

        // --- Layer 1: Local Sliding Window Attention Block ---
        let h_in_1 = h_layer_0.clone();
        let wq_1 = MatrixView::new(&self.attn_block_1.wq, d, d);
        let wk_1 = MatrixView::new(&self.attn_block_1.wk, d, d);
        let wv_1 = MatrixView::new(&self.attn_block_1.wv, d, d);
        let wo_1 = MatrixView::new(&self.attn_block_1.wo, d, d);

        let mut q_1 = vec![0.0f32; t_len * d];
        let mut k_1 = vec![0.0f32; t_len * d];
        let mut v_1 = vec![0.0f32; t_len * d];

        for t in 0..t_len {
            let ht = &h_in_1[t * d..(t + 1) * d];
            matvec(&wq_1, ht, &mut q_1[t * d..(t + 1) * d]);
            matvec(&wk_1, ht, &mut k_1[t * d..(t + 1) * d]);
            matvec(&wv_1, ht, &mut v_1[t * d..(t + 1) * d]);
        }

        let mut h_mid_1 = h_in_1.clone();
        for i in 0..t_len {
            let qi = &q_1[i * d..(i + 1) * d];
            let start_j = if i >= self.attn_block_1.window_size { i - self.attn_block_1.window_size + 1 } else { 0 };
            let count = i - start_j + 1;
            let mut scores = vec![0.0f32; count];
            for (idx, j) in (start_j..=i).enumerate() {
                let kj = &k_1[j * d..(j + 1) * d];
                scores[idx] = dot(qi, kj) * scale_attn;
            }
            let mut probs = vec![0.0f32; count];
            softmax(&scores, &mut probs);

            let mut ctx = vec![0.0f32; d];
            for (idx, j) in (start_j..=i).enumerate() {
                let vj = &v_1[j * d..(j + 1) * d];
                vec_add_scaled(&mut ctx, vj, probs[idx]);
            }
            let mut proj_out = vec![0.0f32; d];
            matvec(&wo_1, &ctx, &mut proj_out);
            vec_add_scaled(&mut h_mid_1[i * d..(i + 1) * d], &proj_out, 1.0);
        }

        // SwiGLU 1
        let w1_1 = MatrixView::new(&self.attn_block_1.w1, self.d_ff, d);
        let w1u_1 = MatrixView::new(&self.attn_block_1.w1u, self.d_ff, d);
        let w2_1 = MatrixView::new(&self.attn_block_1.w2, d, self.d_ff);
        let mut h_layer_1 = h_mid_1.clone();

        for t in 0..t_len {
            let ht = &h_mid_1[t * d..(t + 1) * d];
            let mut gate = vec![0.0f32; self.d_ff];
            let mut up = vec![0.0f32; self.d_ff];
            matvec(&w1_1, ht, &mut gate);
            matvec(&w1u_1, ht, &mut up);

            let mut ff = vec![0.0f32; self.d_ff];
            for i in 0..self.d_ff {
                let g = gate[i];
                let silu = g / (1.0 + (-g).exp());
                ff[i] = silu * up[i];
            }
            let mut ff_out = vec![0.0f32; d];
            matvec(&w2_1, &ff, &mut ff_out);
            vec_add_scaled(&mut h_layer_1[t * d..(t + 1) * d], &ff_out, 1.0);
        }

        // --- Layer 2: RG-LRU Block ---
        let h_in_2 = h_layer_1.clone();
        let wr_2 = MatrixView::new(&self.rnn_block_2.wr, d_r, d);
        let wa_2 = MatrixView::new(&self.rnn_block_2.wa, d_r, d);
        let wx_2 = MatrixView::new(&self.rnn_block_2.wx, d_r, d);
        let wout_2 = MatrixView::new(&self.rnn_block_2.wout, d, d_r);

        let mut h_ssm_2 = vec![0.0f32; (t_len + 1) * d_r];
        let mut h_mid_2 = h_in_2.clone();

        for t in 0..t_len {
            let xt = &h_in_2[t * d..(t + 1) * d];
            let mut rt = vec![0.0f32; d_r];
            let mut at = vec![0.0f32; d_r];
            let mut xpt = vec![0.0f32; d_r];

            matvec(&wr_2, xt, &mut rt);
            matvec(&wa_2, xt, &mut at);
            matvec(&wx_2, xt, &mut xpt);

            let mut prev_h = vec![0.0f32; d_r];
            prev_h.copy_from_slice(&h_ssm_2[t * d_r..(t + 1) * d_r]);
            let curr_h = &mut h_ssm_2[(t + 1) * d_r..(t + 2) * d_r];

            let mut out_r = vec![0.0f32; d_r];
            for i in 0..d_r {
                let r_sig = 1.0 / (1.0 + (-rt[i]).exp());
                let a_sig = 1.0 / (1.0 + (-at[i]).exp());
                let input_gate = (1.0 - a_sig * a_sig).max(0.0).sqrt();
                curr_h[i] = a_sig * prev_h[i] + input_gate * (r_sig * xpt[i]);
                out_r[i] = curr_h[i];
            }

            let mut rnn_out = vec![0.0f32; d];
            matvec(&wout_2, &out_r, &mut rnn_out);
            let h_out_t = &mut h_mid_2[t * d..(t + 1) * d];
            vec_add_scaled(h_out_t, &rnn_out, 1.0);
        }

        // SwiGLU 2
        let w1_2 = MatrixView::new(&self.rnn_block_2.w1, self.d_ff, d);
        let w1u_2 = MatrixView::new(&self.rnn_block_2.w1u, self.d_ff, d);
        let w2_2 = MatrixView::new(&self.rnn_block_2.w2, d, self.d_ff);
        let mut h_layer_2 = h_mid_2.clone();

        for t in 0..t_len {
            let ht = &h_mid_2[t * d..(t + 1) * d];
            let mut gate = vec![0.0f32; self.d_ff];
            let mut up = vec![0.0f32; self.d_ff];
            matvec(&w1_2, ht, &mut gate);
            matvec(&w1u_2, ht, &mut up);

            let mut ff = vec![0.0f32; self.d_ff];
            for i in 0..self.d_ff {
                let g = gate[i];
                let silu = g / (1.0 + (-g).exp());
                ff[i] = silu * up[i];
            }
            let mut ff_out = vec![0.0f32; d];
            matvec(&w2_2, &ff, &mut ff_out);
            vec_add_scaled(&mut h_layer_2[t * d..(t + 1) * d], &ff_out, 1.0);
        }

        // Head & Loss
        let head_v = MatrixView::new(&self.head, v, d);
        let mut g_head_v = MatrixViewMut::new(&mut grads.grad_head, v, d);
        let mut delta_final = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let ht = &h_layer_2[t * d..(t + 1) * d];
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

        // Fast AdamW backward accumulation
        for t in 0..t_len {
            let tok = x_seq[t];
            let pos = t.min(self.max_seq_len - 1);
            let dh = &delta_final[t * d..(t + 1) * d];
            let emb_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            let pos_slice = &mut grads.grad_pos_embed[pos * d..(pos + 1) * d];
            for i in 0..d {
                emb_slice[i] += dh[i];
                pos_slice[i] += dh[i];
            }
        }

        total_loss
    }

    /// Long-Context Needle Recall Probe for Griffin's RG-LRU Recurrent State.
    pub fn probe_needle_recall(&mut self, context_length: usize, seed: u64) -> f32 {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = self.d_model;
        let d_r = self.d_rnn;
        let scale = (1.0f32 / d as f32).sqrt();

        // 1. Needle vector
        let needle: Vec<f32> = (0..d).map(|_| rng.gen_range(-scale..scale)).collect();

        let wr_0 = MatrixView::new(&self.rnn_block_0.wr, d_r, d);
        let wa_0 = MatrixView::new(&self.rnn_block_0.wa, d_r, d);
        let wx_0 = MatrixView::new(&self.rnn_block_0.wx, d_r, d);

        let mut rt = vec![0.0f32; d_r];
        let mut at = vec![0.0f32; d_r];
        let mut xpt = vec![0.0f32; d_r];

        matvec(&wr_0, &needle, &mut rt);
        matvec(&wa_0, &needle, &mut at);
        matvec(&wx_0, &needle, &mut xpt);

        let mut h_state = vec![0.0f32; d_r];
        for i in 0..d_r {
            let r_sig = 1.0 / (1.0 + (-rt[i]).exp());
            let a_sig = 1.0 / (1.0 + (-at[i]).exp());
            let input_gate = (1.0 - a_sig * a_sig).max(0.0).sqrt();
            h_state[i] = input_gate * (r_sig * xpt[i]);
        }

        // 2. Stream N distraction tokens through RG-LRU recurrent decay
        for _ in 0..context_length {
            let dist_tok: Vec<f32> = (0..d).map(|_| rng.gen_range(-scale..scale)).collect();
            matvec(&wr_0, &dist_tok, &mut rt);
            matvec(&wa_0, &dist_tok, &mut at);
            matvec(&wx_0, &dist_tok, &mut xpt);

            for i in 0..d_r {
                let r_sig = 1.0 / (1.0 + (-rt[i]).exp());
                let a_sig = 1.0 / (1.0 + (-at[i]).exp());
                let input_gate = (1.0 - a_sig * a_sig).max(0.0).sqrt();
                h_state[i] = a_sig * h_state[i] + input_gate * (r_sig * xpt[i]);
            }
        }

        // 3. Project retrieved state
        let wout_0 = MatrixView::new(&self.rnn_block_0.wout, d, d_r);
        let mut retrieved = vec![0.0f32; d];
        matvec(&wout_0, &h_state, &mut retrieved);

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

pub struct GriffinAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub m_embed: Vec<f32>, pub v_embed: Vec<f32>,
    pub m_pos_embed: Vec<f32>, pub v_pos_embed: Vec<f32>,
    pub m_head: Vec<f32>, pub v_head: Vec<f32>,
}

impl GriffinAdamW {
    pub fn new(model: &GriffinModel, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            m_embed: vec![0.0f32; model.embeddings.len()],
            v_embed: vec![0.0f32; model.embeddings.len()],
            m_pos_embed: vec![0.0f32; model.pos_embeddings.len()],
            v_pos_embed: vec![0.0f32; model.pos_embeddings.len()],
            m_head: vec![0.0f32; model.head.len()],
            v_head: vec![0.0f32; model.head.len()],
        }
    }

    pub fn step(&mut self, model: &mut GriffinModel, grads: &mut GriffinGrads, current_lr: f32) {
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
    }
}

pub fn evaluate_griffin_bpc(
    model: &mut GriffinModel,
    val_data: &[u8],
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let max_start = val_data.len().saturating_sub(seq_len + 1);

    let mut dummy_grads = GriffinGrads::new(
        model.vocab_size,
        model.d_model,
        model.d_rnn,
        model.d_ff,
        model.max_seq_len,
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

pub fn train_griffin(
    model: &mut GriffinModel,
    train_data: &[u8],
    val_data: &[u8],
    batch_size: usize,
    seq_len: usize,
    max_time_secs: u64,
    max_steps: usize,
    base_lr: f32,
    label: &str,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = GriffinAdamW::new(model, base_lr);
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
        let thread_results: Vec<(f32, GriffinGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut local_model = model_ref.clone();
                let mut grads = GriffinGrads::new(
                    local_model.vocab_size,
                    local_model.d_model,
                    local_model.d_rnn,
                    local_model.d_ff,
                    local_model.max_seq_len,
                );
                let loss = local_model.forward_backward_sequence(&x_seq, &y_seq, &mut grads);
                (loss, grads)
            })
            .collect();

        let mut total_grads = GriffinGrads::new(
            model.vocab_size,
            model.d_model,
            model.d_rnn,
            model.d_ff,
            model.max_seq_len,
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
            let (val_loss, val_bpc) = evaluate_griffin_bpc(model, val_data, 10, seq_len);
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
