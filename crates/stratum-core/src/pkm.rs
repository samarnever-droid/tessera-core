//! STRATUM Product-Key Sparse Memory (PKM) Layer (§3.4).
//! N = m^2 total addressable slots, top-k selection via 2D factored sub-keys,
//! O(√N) addressing cost, and sparse gradient scatter updates.

use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::softmax;
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Diagnostics and statistics for PKM routing and slot utilization.
#[derive(Debug, Clone)]
pub struct PkmStats {
    pub total_slots: usize,
    pub active_k: usize,
    pub total_lookups: usize,
    pub slot_hit_counts: Vec<usize>,
    pub routing_entropies: Vec<f32>,
}

impl PkmStats {
    pub fn new(total_slots: usize, active_k: usize) -> Self {
        Self {
            total_slots,
            active_k,
            total_lookups: 0,
            slot_hit_counts: vec![0; total_slots],
            routing_entropies: Vec::new(),
        }
    }

    pub fn record_routing(&mut self, active_indices: &[usize], weights: &[f32]) {
        self.total_lookups += 1;
        for &idx in active_indices {
            if idx < self.total_slots {
                self.slot_hit_counts[idx] += 1;
            }
        }

        // Routing entropy: - \sum w_i ln(w_i)
        let mut ent = 0.0f32;
        for &w in weights {
            if w > 1e-8 {
                ent -= w * w.ln();
            }
        }
        self.routing_entropies.push(ent);
    }

    /// Compute slot utilization (% of slots activated at least once).
    pub fn slot_utilization(&self) -> f32 {
        let active = self.slot_hit_counts.iter().filter(|&&c| c > 0).count();
        (active as f32 / self.total_slots as f32) * 100.0
    }

    /// Compute median-to-mean slot update ratio (detects Zipfian routing collapse).
    pub fn median_to_mean_ratio(&self) -> f32 {
        let mut sorted = self.slot_hit_counts.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2] as f32;
        let sum: usize = sorted.iter().sum();
        let mean = sum as f32 / sorted.len() as f32;
        if mean > 1e-6 {
            median / mean
        } else {
            0.0f32
        }
    }

    /// Compute 10-bin histogram of slot hit counts.
    pub fn hit_count_histogram(&self) -> Vec<(usize, usize, usize)> {
        let max_hits = *self.slot_hit_counts.iter().max().unwrap_or(&0);
        if max_hits == 0 {
            return vec![(0, 0, self.total_slots)];
        }

        let num_bins = 10;
        let bin_width = (max_hits + num_bins - 1) / num_bins;
        let mut bins = vec![0usize; num_bins];

        for &c in &self.slot_hit_counts {
            let b = (c / bin_width.max(1)).min(num_bins - 1);
            bins[b] += 1;
        }

        bins.into_iter()
            .enumerate()
            .map(|(i, count)| (i * bin_width, (i + 1) * bin_width, count))
            .collect()
    }
}

/// Gradients for Product-Key Memory Layer.
#[derive(Debug, Clone)]
pub struct PkmGrads {
    pub grad_wq: Vec<f32>,       // (d x d)
    pub grad_w_out: Vec<f32>,    // (d x d_v)
    pub grad_keys1: Vec<f32>,    // (m x (d/2))
    pub grad_keys2: Vec<f32>,    // (m x (d/2))
    pub sparse_value_grads: Vec<(usize, Vec<f32>)>, // (slot_id, gradient vector)
}

impl PkmGrads {
    pub fn new(d: usize, d_v: usize, m: usize) -> Self {
        Self {
            grad_wq: vec![0.0f32; d * d],
            grad_w_out: vec![0.0f32; d * d_v],
            grad_keys1: vec![0.0f32; m * (d / 2)],
            grad_keys2: vec![0.0f32; m * (d / 2)],
            sparse_value_grads: Vec::new(),
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32);
        self.grad_w_out.fill(0.0f32);
        self.grad_keys1.fill(0.0f32);
        self.grad_keys2.fill(0.0f32);
        self.sparse_value_grads.clear();
    }

    pub fn add(&mut self, other: &PkmGrads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_w_out.iter_mut().zip(other.grad_w_out.iter()) { *a += b; }
        for (a, &b) in self.grad_keys1.iter_mut().zip(other.grad_keys1.iter()) { *a += b; }
        for (a, &b) in self.grad_keys2.iter_mut().zip(other.grad_keys2.iter()) { *a += b; }
        self.sparse_value_grads.extend_from_slice(&other.sparse_value_grads);
    }
}

/// Product-Key Memory (PKM) Layer.
#[derive(Debug, Clone)]
pub struct ProductKeyMemory {
    pub d_model: usize,
    pub d_half: usize,
    pub d_v: usize,
    pub m: usize,
    pub total_slots: usize, // N = m^2
    pub k_active: usize,
    pub k_sub: usize,       // sqrt(k) candidates per half
    pub wq: Vec<f32>,       // (d x d)
    pub w_out: Vec<f32>,    // (d x d_v)
    pub keys1: Vec<f32>,    // (m x d_half)
    pub keys2: Vec<f32>,    // (m x d_half)
    pub values: Vec<f32>,   // (N x d_v)
    pub stats: PkmStats,
}

impl ProductKeyMemory {
    pub fn new(d_model: usize, d_v: usize, m: usize, k_active: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let d_half = d_model / 2;
        let total_slots = m * m;
        let k_sub = ((k_active as f32).sqrt().ceil() as usize).max(2);

        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_half = (1.0f32 / d_half as f32).sqrt();
        let scale_v = (1.0f32 / d_v as f32).sqrt();

        let wq = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w_out = (0..d_model * d_v).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let keys1 = (0..m * d_half).map(|_| rng.gen_range(-scale_half..scale_half)).collect();
        let keys2 = (0..m * d_half).map(|_| rng.gen_range(-scale_half..scale_half)).collect();
        let values = (0..total_slots * d_v).map(|_| rng.gen_range(-scale_v..scale_v)).collect();

        Self {
            d_model,
            d_half,
            d_v,
            m,
            total_slots,
            k_active,
            k_sub,
            wq,
            w_out,
            keys1,
            keys2,
            values,
            stats: PkmStats::new(total_slots, k_active),
        }
    }

    /// Parameter count calculation.
    pub fn param_count(&self) -> (usize, usize) {
        let dense_params = (self.d_model * self.d_model) // W_q
            + (self.d_model * self.d_v)                  // W_out
            + 2 * (self.m * self.d_half);                // Keys1 + Keys2
        let sparse_slot_params = self.total_slots * self.d_v;
        (dense_params, sparse_slot_params)
    }

    /// Forward pass for a single token query x (d).
    /// Returns: out (d), active_indices (k), active_weights (k), gathered_values (k x d_v), query (d).
    pub fn forward_token(
        &self,
        x: &[f32],
        out: &mut [f32],
        active_indices: &mut [usize],
        active_weights: &mut [f32],
        gathered_val_buf: &mut [f32],
        query_buf: &mut [f32],
    ) {
        let d = self.d_model;
        let d_half = self.d_half;
        let d_v = self.d_v;
        let m = self.m;
        let k = self.k_active;
        let k_sub = self.k_sub;

        // 1. Query projection
        let wq_view = MatrixView::new(&self.wq, d, d);
        matvec(&wq_view, x, query_buf);

        let q1 = &query_buf[..d_half];
        let q2 = &query_buf[d_half..];

        // 2. Score sub-keys: s1 = K1 * q1, s2 = K2 * q2
        let k1_view = MatrixView::new(&self.keys1, m, d_half);
        let k2_view = MatrixView::new(&self.keys2, m, d_half);

        let mut scores1 = vec![0.0f32; m];
        let mut scores2 = vec![0.0f32; m];
        matvec(&k1_view, q1, &mut scores1);
        matvec(&k2_view, q2, &mut scores2);

        // 3. Top-k_sub for each half
        let mut top1: Vec<(f32, usize)> = scores1.into_iter().enumerate().map(|(i, s)| (s, i)).collect();
        let mut top2: Vec<(f32, usize)> = scores2.into_iter().enumerate().map(|(i, s)| (s, i)).collect();
        top1.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        top2.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let top1_sub = &top1[..k_sub.min(m)];
        let top2_sub = &top2[..k_sub.min(m)];

        // 4. Cartesian product candidates: s_{ij} = s1[i] + s2[j]
        let mut candidates = Vec::with_capacity(top1_sub.len() * top2_sub.len());
        for &(s1_val, i_idx) in top1_sub {
            for &(s2_val, j_idx) in top2_sub {
                let slot_idx = i_idx * m + j_idx;
                candidates.push((s1_val + s2_val, slot_idx));
            }
        }

        // 5. Select top-k from candidates
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let top_k_cand = &candidates[..k.min(candidates.len())];

        let mut candidate_scores = vec![0.0f32; top_k_cand.len()];
        for (idx, &(score, slot_id)) in top_k_cand.iter().enumerate() {
            candidate_scores[idx] = score;
            active_indices[idx] = slot_id;
        }

        // Softmax routing weights
        softmax(&candidate_scores, active_weights);

        // 6. Gather active value vectors and compute y = \sum w_r V[slot_id]
        let mut y_accum = vec![0.0f32; d_v];
        for (r, (&slot_id, &w)) in active_indices.iter().zip(active_weights.iter()).enumerate() {
            let slot_start = slot_id * d_v;
            let val_slice = &self.values[slot_start..slot_start + d_v];
            gathered_val_buf[r * d_v..(r + 1) * d_v].copy_from_slice(val_slice);

            vec_add_scaled(&mut y_accum, val_slice, w);
        }

        // 7. Output projection: out = W_out * y + x (residual connection)
        let w_out_view = MatrixView::new(&self.w_out, d, d_v);
        matvec(&w_out_view, &y_accum, out);
        vec_add_scaled(out, x, 1.0);
    }

    /// Forward pass across full sequence (T x d).
    pub fn forward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        h_out: &mut [f32],
        indices_cache: &mut [usize], // (T x k)
        weights_cache: &mut [f32],   // (T x k)
        val_cache: &mut [f32],       // (T x k x d_v)
        query_cache: &mut [f32],     // (T x d)
    ) {
        let d = self.d_model;
        let d_v = self.d_v;
        let k = self.k_active;

        for t in 0..seq_len {
            let x_t = &h_in[t * d..(t + 1) * d];
            let out_t = &mut h_out[t * d..(t + 1) * d];
            let idx_t = &mut indices_cache[t * k..(t + 1) * k];
            let w_t = &mut weights_cache[t * k..(t + 1) * k];
            let val_t = &mut val_cache[t * k * d_v..(t + 1) * k * d_v];
            let q_t = &mut query_cache[t * d..(t + 1) * d];

            self.forward_token(x_t, out_t, idx_t, w_t, val_t, q_t);
        }
    }

    /// Backward pass across full sequence (T x d).
    pub fn backward_sequence(
        &self,
        h_in: &[f32],
        seq_len: usize,
        delta_out: &[f32],   // (T x d)
        delta_in: &mut [f32], // (T x d)
        indices_cache: &[usize],
        weights_cache: &[f32],
        val_cache: &[f32],
        query_cache: &[f32],
        grads: &mut PkmGrads,
    ) {
        let d = self.d_model;
        let d_v = self.d_v;
        let k = self.k_active;

        let wq_view = MatrixView::new(&self.wq, d, d);
        let w_out_view = MatrixView::new(&self.w_out, d, d_v);
        let mut grad_wq = MatrixViewMut::new(&mut grads.grad_wq, d, d);
        let mut grad_w_out = MatrixViewMut::new(&mut grads.grad_w_out, d, d_v);

        let mut d_y = vec![0.0f32; d_v];
        let mut y_accum = vec![0.0f32; d_v];
        let mut d_q = vec![0.0f32; d];
        let mut d_x_accum = vec![0.0f32; d];

        for t in (0..seq_len).rev() {
            let x_t = &h_in[t * d..(t + 1) * d];
            let dy_t = &delta_out[t * d..(t + 1) * d];
            let idx_t = &indices_cache[t * k..(t + 1) * k];
            let w_t = &weights_cache[t * k..(t + 1) * k];
            let val_t = &val_cache[t * k * d_v..(t + 1) * k * d_v];
            let _q_t = &query_cache[t * d..(t + 1) * d];

            // Recompute y_accum = \sum w_r V[r]
            y_accum.fill(0.0f32);
            for r in 0..k {
                let v_r = &val_t[r * d_v..(r + 1) * d_v];
                vec_add_scaled(&mut y_accum, v_r, w_t[r]);
            }

            // W_out gradient: dW_out += dy_t ⊗ y_accum^T
            outer_product_accumulate(dy_t, &y_accum, 1.0, &mut grad_w_out);

            // d_y = W_out^T * dy_t
            matvec_transposed(&w_out_view, dy_t, &mut d_y);

            // Sparse Value gradients: dV[slot_id] += w_r * d_y
            for r in 0..k {
                let slot_id = idx_t[r];
                let w = w_t[r];
                let mut d_val = vec![0.0f32; d_v];
                for i in 0..d_v {
                    d_val[i] = w * d_y[i];
                }
                grads.sparse_value_grads.push((slot_id, d_val));
            }

            // d_w_r = d_y^T * V[r]
            let mut dw = vec![0.0f32; k];
            for r in 0..k {
                let v_r = &val_t[r * d_v..(r + 1) * d_v];
                dw[r] = dot(&d_y, v_r);
            }

            // Softmax backward: d_score_r = w_r * (dw_r - \sum w_j dw_j)
            let mean_dw: f32 = w_t.iter().zip(dw.iter()).map(|(&w, &d)| w * d).sum();
            let mut d_score = vec![0.0f32; k];
            for r in 0..k {
                d_score[r] = w_t[r] * (dw[r] - mean_dw);
            }

            // Approximate gradient into query Q
            // dQ_t ≈ \sum d_score_r * (K1[i] + K2[j])
            d_q.fill(0.0f32);
            let d_half = self.d_half;
            let m = self.m;
            for r in 0..k {
                let slot_id = idx_t[r];
                let i_idx = slot_id / m;
                let j_idx = slot_id % m;
                let ds = d_score[r];

                let k1_slice = &self.keys1[i_idx * d_half..(i_idx + 1) * d_half];
                let k2_slice = &self.keys2[j_idx * d_half..(j_idx + 1) * d_half];

                for i in 0..d_half {
                    d_q[i] += ds * k1_slice[i];
                    d_q[d_half + i] += ds * k2_slice[i];
                }
            }

            // W_q gradient: dW_q += dQ_t ⊗ x_t^T
            outer_product_accumulate(&d_q, x_t, 1.0, &mut grad_wq);

            // delta_in[t] = dy_t (residual) + W_q^T * dQ_t
            let d_in_t = &mut delta_in[t * d..(t + 1) * d];
            d_in_t.copy_from_slice(dy_t);
            matvec_transposed(&wq_view, &d_q, &mut d_x_accum);
            vec_add_scaled(d_in_t, &d_x_accum, 1.0);
        }
    }

    /// Apply sparse SGD updates directly to the value store (§6.4).
    pub fn apply_sparse_value_updates(&mut self, sparse_grads: &[(usize, Vec<f32>)], lr: f32) {
        let d_v = self.d_v;
        for &(slot_id, ref g) in sparse_grads {
            if slot_id < self.total_slots {
                let slot_start = slot_id * d_v;
                let slot_slice = &mut self.values[slot_start..slot_start + d_v];
                for (v, &grad) in slot_slice.iter_mut().zip(g.iter()) {
                    *v -= lr * grad;
                }
            }
        }
    }
}
