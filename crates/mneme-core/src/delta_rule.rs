//! Gated Delta-Rule Recurrent Token Mixer (§5.1).
//! State: S_t = α_t * S_{t-1} * (I - β_t * k_t * k_t^T) + β_t * v_t * k_t^T
//! Output: o_t = S_t * q_t

use axiom_core::activations::sigmoid;
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Delta-rule ablation mode for Experiment E5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaRuleMode {
    NoDelta,          // Token mixer is a simple feedforward projection (no recurrent state)
    Ungated,          // α=1.0, β=1.0 fixed, with erasure
    GatedNoErasure,   // Learned α_t, β_t, but S_t = α S_{t-1} + β v k^T (no -β k k^T erasure)
    FullGatedErasure, // Full MNEME gated delta rule with erasure
}

#[derive(Debug, Clone)]
pub struct DeltaRuleGrads {
    pub grad_wq: Vec<f32>,
    pub grad_wk: Vec<f32>,
    pub grad_wv: Vec<f32>,
    pub grad_wo: Vec<f32>,
    pub grad_w_alpha: Vec<f32>,
    pub grad_w_beta: Vec<f32>,
}

impl DeltaRuleGrads {
    pub fn new(d: usize, d_state: usize) -> Self {
        Self {
            grad_wq: vec![0.0f32; d_state * d],
            grad_wk: vec![0.0f32; d_state * d],
            grad_wv: vec![0.0f32; d_state * d],
            grad_wo: vec![0.0f32; d * d_state],
            grad_w_alpha: vec![0.0f32; d],
            grad_w_beta: vec![0.0f32; d],
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32);
        self.grad_wk.fill(0.0f32);
        self.grad_wv.fill(0.0f32);
        self.grad_wo.fill(0.0f32);
        self.grad_w_alpha.fill(0.0f32);
        self.grad_w_beta.fill(0.0f32);
    }

    pub fn add(&mut self, other: &DeltaRuleGrads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_wk.iter_mut().zip(other.grad_wk.iter()) { *a += b; }
        for (a, &b) in self.grad_wv.iter_mut().zip(other.grad_wv.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        for (a, &b) in self.grad_w_alpha.iter_mut().zip(other.grad_w_alpha.iter()) { *a += b; }
        for (a, &b) in self.grad_w_beta.iter_mut().zip(other.grad_w_beta.iter()) { *a += b; }
    }
}

/// Gated Delta-Rule Layer.
#[derive(Debug, Clone)]
pub struct GatedDeltaRule {
    pub d_model: usize,
    pub d_state: usize,
    pub mode: DeltaRuleMode,
    pub wq: Vec<f32>,       // (d_state x d_model)
    pub wk: Vec<f32>,       // (d_state x d_model)
    pub wv: Vec<f32>,       // (d_state x d_model)
    pub wo: Vec<f32>,       // (d_model x d_state)
    pub w_alpha: Vec<f32>,  // (d_model)
    pub w_beta: Vec<f32>,   // (d_model)
}

impl GatedDeltaRule {
    pub fn new(d_model: usize, d_state: usize, mode: DeltaRuleMode, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_s = (1.0f32 / d_state as f32).sqrt();

        let wq = (0..d_state * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wk = (0..d_state * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wv = (0..d_state * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wo = (0..d_model * d_state).map(|_| rng.gen_range(-scale_s..scale_s)).collect();
        let w_alpha = (0..d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w_beta = (0..d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        Self {
            d_model,
            d_state,
            mode,
            wq,
            wk,
            wv,
            wo,
            w_alpha,
            w_beta,
        }
    }

    pub fn param_count(&self) -> usize {
        match self.mode {
            DeltaRuleMode::NoDelta => self.d_model * self.d_model,
            _ => 3 * (self.d_state * self.d_model) + (self.d_model * self.d_state) + 2 * self.d_model,
        }
    }

    /// Forward pass over sequence of vectors H_in (T x d_model).
    pub fn forward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        h_out: &mut [f32],
        q_cache: &mut [f32],
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        alpha_cache: &mut [f32],
        beta_cache: &mut [f32],
        s_cache: &mut [f32], // (seq_len + 1) * d_state * d_state
    ) {
        let d = self.d_model;
        let ds = self.d_state;

        if self.mode == DeltaRuleMode::NoDelta {
            // Simple feedforward pass without recurrent state
            let wo_v = MatrixView::new(&self.wo[..d * ds.min(d)], d, ds.min(d));
            for t in 0..seq_len {
                let x_t = &h_in[t * d..(t + 1) * d];
                let out_t = &mut h_out[t * d..(t + 1) * d];
                out_t.copy_from_slice(x_t);
            }
            return;
        }

        let wq_v = MatrixView::new(&self.wq, ds, d);
        let wk_v = MatrixView::new(&self.wk, ds, d);
        let wv_v = MatrixView::new(&self.wv, ds, d);
        let wo_v = MatrixView::new(&self.wo, d, ds);

        // Initial state S_0 = 0
        s_cache[..ds * ds].fill(0.0f32);

        let mut o_token = vec![0.0f32; ds];
        let mut sk_t = vec![0.0f32; ds];

        for t in 0..seq_len {
            let x_t = &h_in[t * d..(t + 1) * d];
            let q_t = &mut q_cache[t * ds..(t + 1) * ds];
            let k_t = &mut k_cache[t * ds..(t + 1) * ds];
            let v_t = &mut v_cache[t * ds..(t + 1) * ds];

            matvec(&wq_v, x_t, q_t);
            matvec(&wk_v, x_t, k_t);
            matvec(&wv_v, x_t, v_t);

            // Normalize key vector
            let k_norm = dot(k_t, k_t).sqrt().max(1e-6);
            for elem in k_t.iter_mut() { *elem /= k_norm; }

            // Gates
            let (alpha, beta) = match self.mode {
                DeltaRuleMode::Ungated => (0.95f32, 0.2f32),
                _ => {
                    let mut alpha_raw = [dot(x_t, &self.w_alpha)];
                    let mut alpha_act = [0.0f32; 1];
                    sigmoid(&alpha_raw, &mut alpha_act);
                    let a = 0.8f32 + 0.199f32 * alpha_act[0];

                    let mut beta_raw = [dot(x_t, &self.w_beta)];
                    let mut beta_act = [0.0f32; 1];
                    sigmoid(&beta_raw, &mut beta_act);
                    let b = 0.5f32 * beta_act[0];
                    (a, b)
                }
            };
            alpha_cache[t] = alpha;
            beta_cache[t] = beta;

            let s_prev = &s_cache[t * ds * ds..(t + 1) * ds * ds];
            let s_prev_view = MatrixView::new(s_prev, ds, ds);

            // sk_t = S_{t-1} * k_t
            matvec(&s_prev_view, k_t, &mut sk_t);

            // Compute S_t update:
            // FullGatedErasure: S_t = α S_{t-1} + (β v_t - α β sk_t) ⊗ k_t^T
            // GatedNoErasure:   S_t = α S_{t-1} + (β v_t) ⊗ k_t^T
            for i in 0..ds * ds {
                s_cache[(t + 1) * ds * ds + i] = alpha * s_cache[t * ds * ds + i];
            }

            let mut u_t = vec![0.0f32; ds];
            for i in 0..ds {
                if self.mode == DeltaRuleMode::GatedNoErasure {
                    u_t[i] = beta * v_t[i];
                } else {
                    u_t[i] = beta * v_t[i] - alpha * beta * sk_t[i];
                }
            }

            let s_curr = &mut s_cache[(t + 1) * ds * ds..(t + 2) * ds * ds];
            let mut s_curr_view = MatrixViewMut::new(s_curr, ds, ds);
            outer_product_accumulate(&u_t, k_t, 1.0, &mut s_curr_view);

            // Read output: o_t = S_t * q_t
            let s_curr_read = MatrixView::new(s_curr, ds, ds);
            matvec(&s_curr_read, q_t, &mut o_token);

            // Output projection + residual
            let out_t = &mut h_out[t * d..(t + 1) * d];
            matvec(&wo_v, &o_token, out_t);
            vec_add_scaled(out_t, x_t, 1.0);
        }
    }

    /// Backward pass over sequence.
    pub fn backward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        delta_out: &[f32], // (T x d_model)
        delta_in: &mut [f32], // (T x d_model)
        q_cache: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        alpha_cache: &[f32],
        beta_cache: &[f32],
        s_cache: &[f32],
        grads: &mut DeltaRuleGrads,
    ) {
        let d = self.d_model;
        let ds = self.d_state;

        if self.mode == DeltaRuleMode::NoDelta {
            delta_in.copy_from_slice(delta_out);
            return;
        }

        let wq_v = MatrixView::new(&self.wq, ds, d);
        let wk_v = MatrixView::new(&self.wk, ds, d);
        let wv_v = MatrixView::new(&self.wv, ds, d);
        let wo_v = MatrixView::new(&self.wo, d, ds);

        let mut grad_wq = MatrixViewMut::new(&mut grads.grad_wq, ds, d);
        let mut grad_wk = MatrixViewMut::new(&mut grads.grad_wk, ds, d);
        let mut grad_wv = MatrixViewMut::new(&mut grads.grad_wv, ds, d);
        let mut grad_wo = MatrixViewMut::new(&mut grads.grad_wo, d, ds);

        let mut delta_s = vec![0.0f32; ds * ds];
        let mut o_token = vec![0.0f32; ds];
        let mut d_o = vec![0.0f32; ds];
        let mut d_q = vec![0.0f32; ds];
        let mut d_k = vec![0.0f32; ds];
        let mut d_v = vec![0.0f32; ds];
        let mut sk_t = vec![0.0f32; ds];

        for t in (0..seq_len).rev() {
            let x_t = &h_in[t * d..(t + 1) * d];
            let dy_t = &delta_out[t * d..(t + 1) * d];
            let q_t = &q_cache[t * ds..(t + 1) * ds];
            let k_t = &k_cache[t * ds..(t + 1) * ds];
            let v_t = &v_cache[t * ds..(t + 1) * ds];
            let alpha = alpha_cache[t];
            let beta = beta_cache[t];

            let s_prev = &s_cache[t * ds * ds..(t + 1) * ds * ds];
            let s_curr = &s_cache[(t + 1) * ds * ds..(t + 2) * ds * ds];
            let s_curr_view = MatrixView::new(s_curr, ds, ds);
            let s_prev_view = MatrixView::new(s_prev, ds, ds);

            // Recompute o_t = S_t * q_t
            matvec(&s_curr_view, q_t, &mut o_token);

            // W_o grad: dW_o += dy_t ⊗ o_t^T
            outer_product_accumulate(dy_t, &o_token, 1.0, &mut grad_wo);

            // d_o = W_o^T * dy_t
            matvec_transposed(&wo_v, dy_t, &mut d_o);

            // dS_t += d_o ⊗ q_t^T, d_q = S_t^T * d_o
            let mut delta_s_view = MatrixViewMut::new(&mut delta_s, ds, ds);
            outer_product_accumulate(&d_o, q_t, 1.0, &mut delta_s_view);
            matvec_transposed(&s_curr_view, &d_o, &mut d_q);

            // Recompute u_t
            matvec(&s_prev_view, k_t, &mut sk_t);
            let mut u_t = vec![0.0f32; ds];
            for i in 0..ds {
                if self.mode == DeltaRuleMode::GatedNoErasure {
                    u_t[i] = beta * v_t[i];
                } else {
                    u_t[i] = beta * v_t[i] - alpha * beta * sk_t[i];
                }
            }

            // S_t = alpha * S_{t-1} + u_t ⊗ k_t^T
            let delta_s_read = MatrixView::new(&delta_s, ds, ds);
            let mut d_u = vec![0.0f32; ds];
            matvec(&delta_s_read, k_t, &mut d_u);

            d_k.fill(0.0f32);
            matvec_transposed(&delta_s_read, &u_t, &mut d_k);

            for i in 0..ds {
                d_v[i] = beta * d_u[i];
            }

            let mut next_delta_s = vec![0.0f32; ds * ds];
            for i in 0..ds * ds {
                next_delta_s[i] = alpha * delta_s[i];
            }

            if self.mode != DeltaRuleMode::GatedNoErasure {
                let mut d_sk = vec![0.0f32; ds];
                for i in 0..ds {
                    d_sk[i] = -alpha * beta * d_u[i];
                }
                matvec_transposed(&s_prev_view, &d_sk, &mut o_token);
                for i in 0..ds {
                    d_k[i] += o_token[i];
                }
                let mut next_ds_view = MatrixViewMut::new(&mut next_delta_s, ds, ds);
                outer_product_accumulate(&d_sk, k_t, 1.0, &mut next_ds_view);
            }
            delta_s = next_delta_s;

            // Parameter gradients
            outer_product_accumulate(&d_q, x_t, 1.0, &mut grad_wq);
            outer_product_accumulate(&d_k, x_t, 1.0, &mut grad_wk);
            outer_product_accumulate(&d_v, x_t, 1.0, &mut grad_wv);

            // Input gradient
            let d_in_t = &mut delta_in[t * d..(t + 1) * d];
            d_in_t.copy_from_slice(dy_t);

            let mut d_x_accum = vec![0.0f32; d];
            matvec_transposed(&wq_v, &d_q, &mut d_x_accum);
            vec_add_scaled(d_in_t, &d_x_accum, 1.0);

            matvec_transposed(&wk_v, &d_k, &mut d_x_accum);
            vec_add_scaled(d_in_t, &d_x_accum, 1.0);

            matvec_transposed(&wv_v, &d_v, &mut d_x_accum);
            vec_add_scaled(d_in_t, &d_x_accum, 1.0);
        }
    }
}
