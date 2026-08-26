//! Product-Key Sparse Expert Tier with Sigmoid Gates (§5.3).
//! E total experts, addressed via factored subkey tables with O(√E) cost.

use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Detailed routing diagnostics for Experiment E6.
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub total_experts: usize,
    pub top_k: usize,
    pub total_lookups: usize,
    pub expert_hit_counts: Vec<usize>,
    pub routing_entropies: Vec<f32>,
    pub history_distributions: Vec<(usize, Vec<f32>)>, // (step, distribution)
}

impl RoutingStats {
    pub fn new(total_experts: usize, top_k: usize) -> Self {
        Self {
            total_experts,
            top_k,
            total_lookups: 0,
            expert_hit_counts: vec![0; total_experts],
            routing_entropies: Vec::new(),
            history_distributions: Vec::new(),
        }
    }

    pub fn record_step(&mut self, active_indices: &[usize], weights: &[f32]) {
        self.total_lookups += 1;
        for &idx in active_indices {
            if idx < self.total_experts {
                self.expert_hit_counts[idx] += 1;
            }
        }

        // Routing entropy: - \sum p_i ln(p_i)
        let sum_w: f32 = weights.iter().sum();
        if sum_w > 1e-8 {
            let mut ent = 0.0f32;
            for &w in weights {
                let p = w / sum_w;
                if p > 1e-8 {
                    ent -= p * p.ln();
                }
            }
            self.routing_entropies.push(ent);
        }
    }

    pub fn checkpoint(&mut self, step: usize) {
        let total: usize = self.expert_hit_counts.iter().sum();
        let dist = if total > 0 {
            self.expert_hit_counts.iter().map(|&c| c as f32 / total as f32).collect()
        } else {
            vec![1.0f32 / self.total_experts as f32; self.total_experts]
        };
        self.history_distributions.push((step, dist));
    }

    pub fn dead_experts(&self) -> usize {
        let avg = self.expert_hit_counts.iter().sum::<usize>() as f32 / self.total_experts.max(1) as f32;
        self.expert_hit_counts.iter().filter(|&&c| (c as f32) < (avg * 0.01).max(1.0)).count()
    }

    pub fn hot_experts(&self) -> usize {
        let avg = self.expert_hit_counts.iter().sum::<usize>() as f32 / self.total_experts.max(1) as f32;
        self.expert_hit_counts.iter().filter(|&&c| (c as f32) > (avg * 5.0).max(5.0)).count()
    }

    pub fn mean_entropy(&self) -> f32 {
        if self.routing_entropies.is_empty() {
            0.0f32
        } else {
            self.routing_entropies.iter().sum::<f32>() / self.routing_entropies.len() as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertGrads {
    pub grad_wq: Vec<f32>,
    pub grad_c1: Vec<f32>,
    pub grad_c2: Vec<f32>,
    pub grad_wo: Vec<f32>,
    pub sparse_expert_grads: Vec<(usize, Vec<f32>)>, // (expert_id, gradient)
}

impl ExpertGrads {
    pub fn new(d_model: usize, m: usize) -> Self {
        let d_half = d_model / 2;
        Self {
            grad_wq: vec![0.0f32; d_model * d_model],
            grad_c1: vec![0.0f32; m * d_half],
            grad_c2: vec![0.0f32; m * d_half],
            grad_wo: vec![0.0f32; d_model * d_model],
            sparse_expert_grads: Vec::new(),
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32);
        self.grad_c1.fill(0.0f32);
        self.grad_c2.fill(0.0f32);
        self.grad_wo.fill(0.0f32);
        self.sparse_expert_grads.clear();
    }

    pub fn add(&mut self, other: &ExpertGrads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_c1.iter_mut().zip(other.grad_c1.iter()) { *a += b; }
        for (a, &b) in self.grad_c2.iter_mut().zip(other.grad_c2.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        self.sparse_expert_grads.extend_from_slice(&other.sparse_expert_grads);
    }
}

/// Sparse Product-Key Expert Bank.
#[derive(Debug, Clone)]
pub struct ProductKeyExpertBank {
    pub d_model: usize,
    pub d_half: usize,
    pub m: usize,            // sqrt(E)
    pub total_experts: usize,// E = m * m
    pub top_k: usize,
    pub wq: Vec<f32>,        // (d x d)
    pub c1: Vec<f32>,        // (m x d_half)
    pub c2: Vec<f32>,        // (m x d_half)
    pub expert_values: Vec<f32>, // (E x d)
    pub wo: Vec<f32>,        // (d x d)
    pub stats: RoutingStats,
}

impl ProductKeyExpertBank {
    pub fn new(d_model: usize, n_experts: usize, top_k: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let m = ((n_experts as f32).sqrt().round() as usize).max(2);
        let total_experts = m * m;
        let d_half = d_model / 2;

        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_h = (1.0f32 / d_half as f32).sqrt();

        let wq = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let c1 = (0..m * d_half).map(|_| rng.gen_range(-scale_h..scale_h)).collect();
        let c2 = (0..m * d_half).map(|_| rng.gen_range(-scale_h..scale_h)).collect();
        let expert_values = (0..total_experts * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wo = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        Self {
            d_model,
            d_half,
            m,
            total_experts,
            top_k: top_k.min(total_experts),
            wq,
            c1,
            c2,
            expert_values,
            wo,
            stats: RoutingStats::new(total_experts, top_k),
        }
    }

    pub fn param_count(&self) -> (usize, usize) {
        let dense_params = 2 * (self.d_model * self.d_model) + 2 * (self.m * self.d_half);
        let sparse_params = self.total_experts * self.d_model;
        (dense_params, sparse_params)
    }

    /// Forward pass for token x (d_model).
    pub fn forward_token(
        &self,
        x: &[f32],
        out: &mut [f32],
        active_indices: &mut [usize],
        active_weights: &mut [f32],
        gathered_vals: &mut [f32],
        query_buf: &mut [f32],
    ) {
        let d = self.d_model;
        let d_half = self.d_half;
        let m = self.m;
        let k = self.top_k;
        let k_sub = ((k as f32).sqrt().ceil() as usize).max(2);

        // 1. Query projection
        let wq_view = MatrixView::new(&self.wq, d, d);
        matvec(&wq_view, x, query_buf);

        let q1 = &query_buf[..d_half];
        let q2 = &query_buf[d_half..];

        // 2. Factored scores
        let c1_view = MatrixView::new(&self.c1, m, d_half);
        let c2_view = MatrixView::new(&self.c2, m, d_half);

        let mut s1 = vec![0.0f32; m];
        let mut s2 = vec![0.0f32; m];
        matvec(&c1_view, q1, &mut s1);
        matvec(&c2_view, q2, &mut s2);

        // 3. Top-k_sub candidates
        let mut top1: Vec<(f32, usize)> = s1.into_iter().enumerate().map(|(i, s)| (s, i)).collect();
        let mut top2: Vec<(f32, usize)> = s2.into_iter().enumerate().map(|(i, s)| (s, i)).collect();
        top1.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        top2.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let top1_sub = &top1[..k_sub.min(m)];
        let top2_sub = &top2[..k_sub.min(m)];

        let mut cands = Vec::with_capacity(top1_sub.len() * top2_sub.len());
        for &(v1, i) in top1_sub {
            for &(v2, j) in top2_sub {
                cands.push((v1 + v2, i * m + j));
            }
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let top_cands = &cands[..k.min(cands.len())];

        // 4. Sigmoid gating per expert (HARDPOINT §2.3)
        let mut y_accum = vec![0.0f32; d];
        for (r, &(score, exp_id)) in top_cands.iter().enumerate() {
            active_indices[r] = exp_id;
            let gate = 1.0f32 / (1.0f32 + (-score).exp());
            active_weights[r] = gate;

            let exp_slice = &self.expert_values[exp_id * d..(exp_id + 1) * d];
            gathered_vals[r * d..(r + 1) * d].copy_from_slice(exp_slice);
            vec_add_scaled(&mut y_accum, exp_slice, gate);
        }

        // 5. Output projection + residual
        let wo_view = MatrixView::new(&self.wo, d, d);
        matvec(&wo_view, &y_accum, out);
        vec_add_scaled(out, x, 1.0);
    }

    /// Forward sequence
    pub fn forward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        h_out: &mut [f32],
        indices_cache: &mut [usize],
        weights_cache: &mut [f32],
        vals_cache: &mut [f32],
        queries_cache: &mut [f32],
    ) {
        let d = self.d_model;
        let k = self.top_k;
        for t in 0..seq_len {
            let x_t = &h_in[t * d..(t + 1) * d];
            let out_t = &mut h_out[t * d..(t + 1) * d];
            let idx_t = &mut indices_cache[t * k..(t + 1) * k];
            let w_t = &mut weights_cache[t * k..(t + 1) * k];
            let v_t = &mut vals_cache[t * k * d..(t + 1) * k * d];
            let q_t = &mut queries_cache[t * d..(t + 1) * d];

            self.forward_token(x_t, out_t, idx_t, w_t, v_t, q_t);
        }
    }

    /// Backward sequence
    pub fn backward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        delta_out: &[f32],
        delta_in: &mut [f32],
        indices_cache: &[usize],
        weights_cache: &[f32],
        vals_cache: &[f32],
        _queries_cache: &[f32],
        grads: &mut ExpertGrads,
    ) {
        let d = self.d_model;
        let d_half = self.d_half;
        let m = self.m;
        let k = self.top_k;

        let wq_view = MatrixView::new(&self.wq, d, d);
        let wo_view = MatrixView::new(&self.wo, d, d);
        let mut grad_wq = MatrixViewMut::new(&mut grads.grad_wq, d, d);
        let mut grad_wo = MatrixViewMut::new(&mut grads.grad_wo, d, d);

        let mut d_y = vec![0.0f32; d];
        let mut y_accum = vec![0.0f32; d];
        let mut d_q = vec![0.0f32; d];
        let mut d_x_accum = vec![0.0f32; d];

        for t in (0..seq_len).rev() {
            let x_t = &h_in[t * d..(t + 1) * d];
            let dy_t = &delta_out[t * d..(t + 1) * d];
            let idx_t = &indices_cache[t * k..(t + 1) * k];
            let w_t = &weights_cache[t * k..(t + 1) * k];
            let v_t = &vals_cache[t * k * d..(t + 1) * k * d];

            // Recompute y_accum
            y_accum.fill(0.0f32);
            for r in 0..k {
                let v_r = &v_t[r * d..(r + 1) * d];
                vec_add_scaled(&mut y_accum, v_r, w_t[r]);
            }

            // W_o grad: dy_t ⊗ y_accum^T
            outer_product_accumulate(dy_t, &y_accum, 1.0, &mut grad_wo);

            // d_y = W_o^T * dy_t
            matvec_transposed(&wo_view, dy_t, &mut d_y);

            // Sparse expert gradients: dExpert[id] += w_r * d_y
            for r in 0..k {
                let exp_id = idx_t[r];
                let w = w_t[r];
                let mut d_exp = vec![0.0f32; d];
                for i in 0..d { d_exp[i] = w * d_y[i]; }
                grads.sparse_expert_grads.push((exp_id, d_exp));
            }

            // Gated gradients into query Q
            d_q.fill(0.0f32);
            for r in 0..k {
                let exp_id = idx_t[r];
                let w = w_t[r];
                let v_r = &v_t[r * d..(r + 1) * d];
                let dw = dot(&d_y, v_r);
                let d_score = dw * (w * (1.0 - w)); // sigmoid derivative

                let i_idx = exp_id / m;
                let j_idx = exp_id % m;

                let c1_slice = &self.c1[i_idx * d_half..(i_idx + 1) * d_half];
                let c2_slice = &self.c2[j_idx * d_half..(j_idx + 1) * d_half];

                for i in 0..d_half {
                    d_q[i] += d_score * c1_slice[i];
                    d_q[d_half + i] += d_score * c2_slice[i];
                }
            }

            // W_q grad: dQ_t ⊗ x_t^T
            outer_product_accumulate(&d_q, x_t, 1.0, &mut grad_wq);

            // delta_in[t] = dy_t (residual) + W_q^T * dQ_t
            let d_in_t = &mut delta_in[t * d..(t + 1) * d];
            d_in_t.copy_from_slice(dy_t);
            matvec_transposed(&wq_view, &d_q, &mut d_x_accum);
            vec_add_scaled(d_in_t, &d_x_accum, 1.0);
        }
    }

    pub fn apply_sparse_expert_updates(&mut self, sparse_grads: &[(usize, Vec<f32>)], lr: f32) {
        let d = self.d_model;
        for &(exp_id, ref g) in sparse_grads {
            if exp_id < self.total_experts {
                let start = exp_id * d;
                let slice = &mut self.expert_values[start..start + d];
                for (v, &grad) in slice.iter_mut().zip(g.iter()) {
                    *v -= lr * grad;
                }
            }
        }
    }
}
