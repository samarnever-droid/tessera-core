//! STRATUM Delta-Rule Linear Recurrence with Matrix Associative State (§3.2).
//! State: S_t = α_t * S_{t-1} * (I - β_t * k_t * k_t^T) + β_t * v_t * k_t^T
//! Output: o_t = S_t * q_t

use axiom_core::activations::sigmoid;
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Gradients for Delta-Rule Layer.
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
    pub fn new(d: usize) -> Self {
        Self {
            grad_wq: vec![0.0f32; d * d],
            grad_wk: vec![0.0f32; d * d],
            grad_wv: vec![0.0f32; d * d],
            grad_wo: vec![0.0f32; d * d],
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

/// Delta-Rule Linear Recurrence Layer.
#[derive(Debug, Clone)]
pub struct DeltaRuleLayer {
    pub d_model: usize,
    pub wq: Vec<f32>,       // (d x d)
    pub wk: Vec<f32>,       // (d x d)
    pub wv: Vec<f32>,       // (d x d)
    pub wo: Vec<f32>,       // (d x d)
    pub w_alpha: Vec<f32>,  // (d)
    pub w_beta: Vec<f32>,   // (d)
}

impl DeltaRuleLayer {
    pub fn new(d_model: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0f32 / d_model as f32).sqrt();

        let wq = (0..d_model * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let wk = (0..d_model * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let wv = (0..d_model * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let wo = (0..d_model * d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_alpha = (0..d_model).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_beta = (0..d_model).map(|_| rng.gen_range(-scale..scale)).collect();

        Self { d_model, wq, wk, wv, wo, w_alpha, w_beta }
    }

    /// Forward pass over sequence of vectors H_in (T x d).
    /// Stores outputs H_out (T x d) and caches states and activations for backward pass.
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
        s_cache: &mut [f32], // (seq_len + 1) * d * d
    ) {
        let d = self.d_model;
        let wq_v = MatrixView::new(&self.wq, d, d);
        let wk_v = MatrixView::new(&self.wk, d, d);
        let wv_v = MatrixView::new(&self.wv, d, d);
        let wo_v = MatrixView::new(&self.wo, d, d);

        // Initial matrix state S_0 = 0
        s_cache[..d * d].fill(0.0f32);

        let mut o_token = vec![0.0f32; d];
        let mut sk_t = vec![0.0f32; d]; // S_{t-1} * k_t

        for t in 0..seq_len {
            let x_t = &h_in[t * d..(t + 1) * d];
            let q_t = &mut q_cache[t * d..(t + 1) * d];
            let k_t = &mut k_cache[t * d..(t + 1) * d];
            let v_t = &mut v_cache[t * d..(t + 1) * d];

            matvec(&wq_v, x_t, q_t);
            matvec(&wk_v, x_t, k_t);
            matvec(&wv_v, x_t, v_t);

            // Normalise k_t to unit length
            let k_norm = dot(k_t, k_t).sqrt().max(1e-6);
            for elem in k_t.iter_mut() { *elem /= k_norm; }

            // Gates α_t, β_t ∈ (0, 1)
            let alpha_raw = [dot(x_t, &self.w_alpha)];
            let mut alpha_act = [0.0f32; 1];
            sigmoid(&alpha_raw, &mut alpha_act);
            let alpha = 0.8f32 + 0.199f32 * alpha_act[0]; // Bound decay to [0.8, 0.999]
            alpha_cache[t] = alpha;

            let beta_raw = [dot(x_t, &self.w_beta)];
            let mut beta_act = [0.0f32; 1];
            sigmoid(&beta_raw, &mut beta_act);
            let beta = 0.5f32 * beta_act[0]; // Bound write strength to [0, 0.5]
            beta_cache[t] = beta;

            let s_prev = &s_cache[t * d * d..(t + 1) * d * d];
            let s_prev_view = MatrixView::new(s_prev, d, d);

            // sk_t = S_{t-1} * k_t
            matvec(&s_prev_view, k_t, &mut sk_t);

            // Compute S_t = α * S_{t-1} - α * β * (sk_t * k_t^T) + β * (v_t * k_t^T)
            //             = α * S_{t-1} + (β * v_t - α * β * sk_t) * k_t^T
            for i in 0..d * d {
                s_cache[(t + 1) * d * d + i] = alpha * s_cache[t * d * d + i];
            }

            let mut u_t = vec![0.0f32; d];
            for i in 0..d {
                u_t[i] = beta * v_t[i] - alpha * beta * sk_t[i];
            }

            let s_curr = &mut s_cache[(t + 1) * d * d..(t + 2) * d * d];
            let mut s_curr_view = MatrixViewMut::new(s_curr, d, d);
            outer_product_accumulate(&u_t, k_t, 1.0, &mut s_curr_view);

            // o_t = S_t * q_t
            let s_curr_read = MatrixView::new(s_curr, d, d);
            matvec(&s_curr_read, q_t, &mut o_token);

            // Output projection + residual connection
            let out_t = &mut h_out[t * d..(t + 1) * d];
            matvec(&wo_v, &o_token, out_t);
            vec_add_scaled(out_t, x_t, 1.0);
        }
    }

    /// Exact BPTT Backward Pass across sequence.
    pub fn backward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        delta_out: &[f32], // (T x d) gradient from upstream
        delta_in: &mut [f32], // (T x d) gradient back to input
        q_cache: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        alpha_cache: &[f32],
        beta_cache: &[f32],
        s_cache: &[f32],
        grads: &mut DeltaRuleGrads,
    ) {
        let d = self.d_model;
        let wq_v = MatrixView::new(&self.wq, d, d);
        let wk_v = MatrixView::new(&self.wk, d, d);
        let wv_v = MatrixView::new(&self.wv, d, d);
        let wo_v = MatrixView::new(&self.wo, d, d);

        let mut grad_wq = MatrixViewMut::new(&mut grads.grad_wq, d, d);
        let mut grad_wk = MatrixViewMut::new(&mut grads.grad_wk, d, d);
        let mut grad_wv = MatrixViewMut::new(&mut grads.grad_wv, d, d);
        let mut grad_wo = MatrixViewMut::new(&mut grads.grad_wo, d, d);

        let mut delta_s = vec![0.0f32; d * d]; // Matrix state adjoint dS_t
        let mut o_token = vec![0.0f32; d];
        let mut d_o = vec![0.0f32; d];
        let mut d_q = vec![0.0f32; d];
        let mut d_k = vec![0.0f32; d];
        let mut d_v = vec![0.0f32; d];
        let mut sk_t = vec![0.0f32; d];

        for t in (0..seq_len).rev() {
            let x_t = &h_in[t * d..(t + 1) * d];
            let dy_t = &delta_out[t * d..(t + 1) * d];
            let q_t = &q_cache[t * d..(t + 1) * d];
            let k_t = &k_cache[t * d..(t + 1) * d];
            let v_t = &v_cache[t * d..(t + 1) * d];
            let alpha = alpha_cache[t];
            let beta = beta_cache[t];

            let s_prev = &s_cache[t * d * d..(t + 1) * d * d];
            let s_curr = &s_cache[(t + 1) * d * d..(t + 2) * d * d];
            let s_curr_view = MatrixView::new(s_curr, d, d);
            let s_prev_view = MatrixView::new(s_prev, d, d);

            // Recompute o_t = S_t * q_t
            matvec(&s_curr_view, q_t, &mut o_token);

            // W_o gradient: dW_o += dy_t ⊗ o_t^T
            outer_product_accumulate(dy_t, &o_token, 1.0, &mut grad_wo);

            // d_o = W_o^T * dy_t
            matvec_transposed(&wo_v, dy_t, &mut d_o);

            // o_t = S_t * q_t => dS_t += d_o ⊗ q_t^T, d_q = S_t^T * d_o
            let mut delta_s_view = MatrixViewMut::new(&mut delta_s, d, d);
            outer_product_accumulate(&d_o, q_t, 1.0, &mut delta_s_view);
            matvec_transposed(&s_curr_view, &d_o, &mut d_q);

            // Recompute u_t = beta * v_t - alpha * beta * (S_{t-1} * k_t)
            matvec(&s_prev_view, k_t, &mut sk_t);
            let mut u_t = vec![0.0f32; d];
            for i in 0..d {
                u_t[i] = beta * v_t[i] - alpha * beta * sk_t[i];
            }

            // S_t = alpha * S_{t-1} + u_t ⊗ k_t^T
            // d_u_t = dS_t * k_t
            let delta_s_read = MatrixView::new(&delta_s, d, d);
            let mut d_u = vec![0.0f32; d];
            matvec(&delta_s_read, k_t, &mut d_u);

            // d_k_t from outer product: d_k += dS_t^T * u_t
            d_k.fill(0.0f32);
            matvec_transposed(&delta_s_read, &u_t, &mut d_k);

            // From u_t = beta * v_t - alpha * beta * sk_t:
            // d_v_t = beta * d_u
            for i in 0..d {
                d_v[i] = beta * d_u[i];
            }

            // d_sk_t = -alpha * beta * d_u
            let mut d_sk = vec![0.0f32; d];
            for i in 0..d {
                d_sk[i] = -alpha * beta * d_u[i];
            }

            // sk_t = S_{t-1} * k_t => dS_{t-1}_from_sk = d_sk ⊗ k_t^T, d_k += S_{t-1}^T * d_sk
            matvec_transposed(&s_prev_view, &d_sk, &mut o_token);
            for i in 0..d {
                d_k[i] += o_token[i];
            }

            // Next adjoint: dS_{t-1} = alpha * dS_t + d_sk ⊗ k_t^T
            let mut next_delta_s = vec![0.0f32; d * d];
            for i in 0..d * d {
                next_delta_s[i] = alpha * delta_s[i];
            }
            let mut next_ds_view = MatrixViewMut::new(&mut next_delta_s, d, d);
            outer_product_accumulate(&d_sk, k_t, 1.0, &mut next_ds_view);
            delta_s = next_delta_s;

            // Parameter gradients for W_q, W_k, W_v
            outer_product_accumulate(&d_q, x_t, 1.0, &mut grad_wq);
            outer_product_accumulate(&d_k, x_t, 1.0, &mut grad_wk);
            outer_product_accumulate(&d_v, x_t, 1.0, &mut grad_wv);

            // Input gradient delta_in[t] = dy_t (residual) + W_q^T d_q + W_k^T d_k + W_v^T d_v
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
