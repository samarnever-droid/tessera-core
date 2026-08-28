//! Multi-Resolution Working Memory v2 (MRM) for TESSERA.
//! Features:
//! 1. Multi-Tier Adaptive Write Engine:
//!    - Hard Overwrite (sim >= 0.95)
//!    - Soft Semantic Merge & Temporal Drift Tracking (0.82 <= sim < 0.95)
//!    - Salience-Weighted LRQ Eviction (sim < 0.82)
//! 2. Dual-Resolution storage: K_fine exact token slots + K_coarse EMA summary centroids.
//! 3. Sharp Cosine Temperature Softmax (tau = 0.05) for high-precision needle discrimination.
//! 4. Graceful Overflow Fallback: Coarse centroids preserve semantic summaries when M > K_fine.

use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::softmax;
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct MrmV2Stats {
    pub total_writes: usize,
    pub total_reads: usize,
    pub fine_hits: Vec<usize>,
    pub coarse_hits: Vec<usize>,
    pub needle_retained: bool,
}

impl MrmV2Stats {
    pub fn new(k_fine: usize, k_coarse: usize) -> Self {
        Self {
            total_writes: 0,
            total_reads: 0,
            fine_hits: vec![0; k_fine],
            coarse_hits: vec![0; k_coarse],
            needle_retained: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MrmV2Grads {
    pub grad_wq: Vec<f32>,
    pub grad_wk: Vec<f32>,
    pub grad_wv: Vec<f32>,
    pub grad_wo: Vec<f32>,
    pub grad_wgate: Vec<f32>,
}

impl MrmV2Grads {
    pub fn new(d: usize) -> Self {
        Self {
            grad_wq: vec![0.0f32; d * d],
            grad_wk: vec![0.0f32; d * d],
            grad_wv: vec![0.0f32; d * d],
            grad_wo: vec![0.0f32; d * d],
            grad_wgate: vec![0.0f32; d],
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32);
        self.grad_wk.fill(0.0f32);
        self.grad_wv.fill(0.0f32);
        self.grad_wo.fill(0.0f32);
        self.grad_wgate.fill(0.0f32);
    }

    pub fn add(&mut self, other: &MrmV2Grads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_wk.iter_mut().zip(other.grad_wk.iter()) { *a += b; }
        for (a, &b) in self.grad_wv.iter_mut().zip(other.grad_wv.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        for (a, &b) in self.grad_wgate.iter_mut().zip(other.grad_wgate.iter()) { *a += b; }
    }
}

/// Multi-Resolution Memory (MRM) with Adaptive Multi-Tier Deduplication & LRQ Eviction Policy.
#[derive(Debug, Clone)]
pub struct MultiResMemoryV2 {
    pub d: usize,
    pub k_fine: usize,
    pub k_coarse: usize,
    // Projections
    pub w_q: Vec<f32>,     // (d x d)
    pub w_k: Vec<f32>,     // (d x d)
    pub w_v: Vec<f32>,     // (d x d)
    pub w_o: Vec<f32>,     // (d x d)
    pub w_gate: Vec<f32>,  // (d)
    // Memory Storage
    pub fine_keys: Vec<f32>,       // (k_fine x d)
    pub fine_vals: Vec<f32>,       // (k_fine x d)
    pub fine_salience: Vec<f32>,   // (k_fine)
    pub fine_hits: Vec<f32>,       // (k_fine) query hit decay counter
    pub coarse_centroids: Vec<f32>,// (k_coarse x d)
    pub coarse_vals: Vec<f32>,     // (k_coarse x d)
    pub num_occupied_slots: usize,
    pub stats: MrmV2Stats,
}

impl MultiResMemoryV2 {
    pub fn new(d: usize, k_fine: usize, k_coarse: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0f32 / d as f32).sqrt();

        let w_q = (0..d * d).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_k = (0..d * d).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_v = (0..d * d).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_o = (0..d * d).map(|_| rng.gen_range(-scale * 0.1..scale * 0.1)).collect(); // Warm start
        let w_gate = vec![0.0f32; d]; // Zero init

        let fine_keys = vec![0.0f32; k_fine * d];
        let fine_vals = vec![0.0f32; k_fine * d];
        let fine_salience = vec![0.0f32; k_fine];
        let fine_hits = vec![0.0f32; k_fine];

        let coarse_centroids = (0..k_coarse * d).map(|_| rng.gen_range(-scale..scale)).collect();
        let coarse_vals = vec![0.0f32; k_coarse * d];

        Self {
            d,
            k_fine,
            k_coarse,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            fine_keys,
            fine_vals,
            fine_salience,
            fine_hits,
            coarse_centroids,
            coarse_vals,
            num_occupied_slots: 0,
            stats: MrmV2Stats::new(k_fine, k_coarse),
        }
    }

    /// Create a Hardware-Aware Memory Buffer sized to saturate a specific MB budget (e.g. 20MB or 32MB).
    /// Dynamically computes the maximum fine and coarse slots that fit inside the cache.
    pub fn new_hardware_adaptive(d: usize, buffer_mb: usize, seed: u64) -> Self {
        let (k_fine, k_coarse) = Self::compute_optimal_slots(d, buffer_mb);
        Self::new(d, k_fine, k_coarse, seed)
    }

    /// Computes the exact number of K_fine and K_coarse token slots for a given Megabyte budget.
    pub fn compute_optimal_slots(d: usize, buffer_mb: usize) -> (usize, usize) {
        let target_bytes = buffer_mb * 1024 * 1024;
        let bytes_per_fine_slot = (d * 2 + 2) * std::mem::size_of::<f32>(); // Key, Val, Salience, Hit counter
        let bytes_per_coarse_slot = (d * 2) * std::mem::size_of::<f32>();   // Centroid, Val
        
        // 80% fine slots, 20% coarse summary centroids
        let fine_bytes = (target_bytes as f64 * 0.80) as usize;
        let coarse_bytes = (target_bytes as f64 * 0.20) as usize;
        
        let k_fine = (fine_bytes / bytes_per_fine_slot).max(64);
        let k_coarse = (coarse_bytes / bytes_per_coarse_slot).max(16);
        (k_fine, k_coarse)
    }

    pub fn param_count(&self) -> usize {
        4 * (self.d * self.d) + self.d
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.k_fine * self.d * 2 + self.k_coarse * self.d * 2 + self.k_fine * 2) * 4
    }

    /// Insert / write an item into memory using the Adaptive Multi-Tier Engine:
    /// - Tier 1: Hard In-Place Overwrite (sim >= 0.95)
    /// - Tier 2: Soft Semantic Merge (0.82 <= sim < 0.95) for incremental drift tracking
    /// - Tier 3: Salience-Weighted LRQ Eviction (sim < 0.82)
    pub fn write_token(&mut self, key_vec: &[f32], val_vec: &[f32], salience: f32) {
        let d = self.d;
        let k_fine = self.k_fine;

        // 1. Scan fine slots for semantic correlation
        let key_norm = dot(key_vec, key_vec).sqrt().max(1e-8);
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_sim_slot = None;

        for i in 0..self.num_occupied_slots {
            let k_slice = &self.fine_keys[i * d..(i + 1) * d];
            let k_norm = dot(k_slice, k_slice).sqrt().max(1e-8);
            let sim = dot(key_vec, k_slice) / (key_norm * k_norm);
            if sim > best_sim {
                best_sim = sim;
                best_sim_slot = Some(i);
            }
        }

        // 2. Multi-Tier Resolution
        if best_sim >= 0.95 {
            // Tier 1: Hard In-Place Overwrite (Identical Entity Update)
            let slot_idx = best_sim_slot.unwrap();
            self.fine_keys[slot_idx * d..(slot_idx + 1) * d].copy_from_slice(key_vec);
            self.fine_vals[slot_idx * d..(slot_idx + 1) * d].copy_from_slice(val_vec);
            self.fine_salience[slot_idx] = salience.max(self.fine_salience[slot_idx]);
            self.fine_hits[slot_idx] = (self.fine_hits[slot_idx] + 1.0).min(50.0);
        } else if best_sim >= 0.82 {
            // Tier 2: Soft Semantic Merge (Incremental Drift Tracking)
            let slot_idx = best_sim_slot.unwrap();
            let alpha = 0.70f32;
            let k_slice = &mut self.fine_keys[slot_idx * d..(slot_idx + 1) * d];
            let v_slice = &mut self.fine_vals[slot_idx * d..(slot_idx + 1) * d];
            for c in 0..d {
                k_slice[c] = alpha * key_vec[c] + (1.0 - alpha) * k_slice[c];
                v_slice[c] = alpha * val_vec[c] + (1.0 - alpha) * v_slice[c];
            }
            self.fine_salience[slot_idx] = salience.max(self.fine_salience[slot_idx]);
            self.fine_hits[slot_idx] = (self.fine_hits[slot_idx] + 0.5).min(50.0);
        } else {
            // Tier 3: New Entity Insertion / LRQ Eviction
            let slot_idx = if self.num_occupied_slots < k_fine {
                let idx = self.num_occupied_slots;
                self.num_occupied_slots += 1;
                idx
            } else {
                // Find slot with minimum utility (hits * 2.0 + salience)
                let mut min_util = f32::INFINITY;
                let mut victim_idx = 0usize;
                for i in 0..k_fine {
                    let util = self.fine_hits[i] * 2.0 + self.fine_salience[i];
                    if util < min_util {
                        min_util = util;
                        victim_idx = i;
                    }
                }
                victim_idx
            };

            self.fine_keys[slot_idx * d..(slot_idx + 1) * d].copy_from_slice(key_vec);
            self.fine_vals[slot_idx * d..(slot_idx + 1) * d].copy_from_slice(val_vec);
            self.fine_salience[slot_idx] = salience;
            self.fine_hits[slot_idx] = 1.0f32;
        }

        // 3. Update coarse centroid via EMA summary
        let mut best_centroid = 0usize;
        let mut max_c_sim = f32::NEG_INFINITY;
        for c in 0..self.k_coarse {
            let centroid = &self.coarse_centroids[c * d..(c + 1) * d];
            let c_norm = dot(centroid, centroid).sqrt().max(1e-8);
            let sim = dot(key_vec, centroid) / (key_norm * c_norm);
            if sim > max_c_sim {
                max_c_sim = sim;
                best_centroid = c;
            }
        }

        let gamma = 0.95f32;
        let c_slice = &mut self.coarse_centroids[best_centroid * d..(best_centroid + 1) * d];
        let v_slice = &mut self.coarse_vals[best_centroid * d..(best_centroid + 1) * d];
        for i in 0..d {
            c_slice[i] = gamma * c_slice[i] + (1.0 - gamma) * key_vec[i];
            v_slice[i] = gamma * v_slice[i] + (1.0 - gamma) * val_vec[i];
        }

        self.stats.total_writes += 1;
    }

    /// Read from MRM using query vector Q with sharp Cosine Temperature Softmax (tau = 0.05).
    pub fn read_memory(&mut self, query_vec: &[f32], out_context: &mut [f32]) {
        self.read_memory_const(query_vec, out_context);
        self.stats.total_reads += 1;
    }

    /// Pure immutable read from MRM using query vector Q without modifying hit counters.
    pub fn read_memory_const(&self, query_vec: &[f32], out_context: &mut [f32]) {
        let d = self.d;
        let k_fine = self.num_occupied_slots.max(1);
        let k_total = k_fine + self.k_coarse;
        let q_norm = dot(query_vec, query_vec).sqrt().max(1e-8);
        let temp = 0.05f32; // Sharp temperature for high discrimination

        let mut scores = vec![0.0f32; k_total];

        // 1. Fine slot cosine scores
        for i in 0..k_fine {
            let k_slice = &self.fine_keys[i * d..(i + 1) * d];
            let k_norm = dot(k_slice, k_slice).sqrt().max(1e-8);
            let cos_sim = dot(query_vec, k_slice) / (q_norm * k_norm);
            scores[i] = cos_sim / temp;
        }

        // 2. Coarse slot cosine scores
        for c in 0..self.k_coarse {
            let c_slice = &self.coarse_centroids[c * d..(c + 1) * d];
            let c_norm = dot(c_slice, c_slice).sqrt().max(1e-8);
            let cos_sim = dot(query_vec, c_slice) / (q_norm * c_norm);
            scores[k_fine + c] = cos_sim / temp;
        }

        // 3. Softmax
        let mut probs = vec![0.0f32; k_total];
        softmax(&scores, &mut probs);

        // 4. Weighted Value Read
        out_context.fill(0.0f32);
        for i in 0..k_fine {
            let p = probs[i];
            if p > 1e-5 {
                let v_slice = &self.fine_vals[i * d..(i + 1) * d];
                vec_add_scaled(out_context, v_slice, p);
            }
        }
        for c in 0..self.k_coarse {
            let p = probs[k_fine + c];
            if p > 1e-5 {
                let v_slice = &self.coarse_vals[c * d..(c + 1) * d];
                vec_add_scaled(out_context, v_slice, p);
            }
        }
    }

    /// Forward pass over sequence of tokens.
    pub fn forward_sequence(
        &mut self,
        h_in: &[f32],
        seq_len: usize,
        h_out: &mut [f32],
    ) {
        let d = self.d;
        let w_q_local = self.w_q.clone();
        let w_k_local = self.w_k.clone();
        let w_v_local = self.w_v.clone();
        let w_o_local = self.w_o.clone();
        let w_gate_local = self.w_gate.clone();

        let wq_v = MatrixView::new(&w_q_local, d, d);
        let wk_v = MatrixView::new(&w_k_local, d, d);
        let wv_v = MatrixView::new(&w_v_local, d, d);
        let wo_v = MatrixView::new(&w_o_local, d, d);

        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; d];
        let mut v = vec![0.0f32; d];
        let mut mem_context = vec![0.0f32; d];
        let mut proj_out = vec![0.0f32; d];

        for t in 0..seq_len {
            let x_t = &h_in[t * d..(t + 1) * d];

            matvec(&wq_v, x_t, &mut q);
            matvec(&wk_v, x_t, &mut k);
            matvec(&wv_v, x_t, &mut v);

            // Read existing memory
            self.read_memory(&q, &mut mem_context);

            // Output projection + gate
            matvec(&wo_v, &mem_context, &mut proj_out);

            let gate_raw = dot(x_t, &w_gate_local) - 2.0f32;
            let gate_sig = 1.0f32 / (1.0f32 + (-gate_raw).exp());

            let out_t = &mut h_out[t * d..(t + 1) * d];
            out_t.copy_from_slice(x_t);
            vec_add_scaled(out_t, &proj_out, gate_sig);

            // Write current token into memory with norm-based salience
            let salience = dot(&k, &k).sqrt();
            self.write_token(&k, &v, salience);
        }
    }

    /// Analytical Backward Pass for MRM Sequence Processing
    pub fn backward_sequence(
        &self,
        h_in: &[f32],
        d_out: &[f32],
        d_in: &mut [f32],
        grads: &mut MrmV2Grads,
        seq_len: usize,
    ) {
        let d = self.d;
        let temp = 0.05f32;

        let wq_v = MatrixView::new(&self.w_q, d, d);
        let wo_v = MatrixView::new(&self.w_o, d, d);
        let mut gwq = MatrixViewMut::new(&mut grads.grad_wq, d, d);
        let mut gwk = MatrixViewMut::new(&mut grads.grad_wk, d, d);
        let mut gwv = MatrixViewMut::new(&mut grads.grad_wv, d, d);
        let mut gwo = MatrixViewMut::new(&mut grads.grad_wo, d, d);

        let mut q = vec![0.0f32; d];
        let mut mem_context = vec![0.0f32; d];
        let mut proj_out = vec![0.0f32; d];
        let mut d_proj = vec![0.0f32; d];
        let mut d_ctx = vec![0.0f32; d];
        let mut d_q = vec![0.0f32; d];
        let mut tmp_d = vec![0.0f32; d];

        for t in 0..seq_len {
            let x_t = &h_in[t * d..(t + 1) * d];
            let dh_out = &d_out[t * d..(t + 1) * d];
            let dh_in = &mut d_in[t * d..(t + 1) * d];

            // Re-simulate forward at step t
            matvec(&wq_v, x_t, &mut q);
            self.read_memory_const(&q, &mut mem_context);
            matvec(&wo_v, &mem_context, &mut proj_out);

            let gate_raw = dot(x_t, &self.w_gate) - 2.0f32;
            let gate_sig = 1.0f32 / (1.0f32 + (-gate_raw).exp());

            // 1. Residual + Gate Gradient
            dh_in.copy_from_slice(dh_out);

            let dot_dh_proj = dot(dh_out, &proj_out);
            let d_gate_raw = dot_dh_proj * gate_sig * (1.0 - gate_sig);

            for i in 0..d {
                grads.grad_wgate[i] += d_gate_raw * x_t[i];
                dh_in[i] += d_gate_raw * self.w_gate[i];
                d_proj[i] = gate_sig * dh_out[i];
            }

            // 2. Output Projection Backward
            outer_product_accumulate(&d_proj, &mem_context, 1.0, &mut gwo);
            matvec_transposed(&wo_v, &d_proj, &mut d_ctx);

            // 3. Memory Read Softmax Gradient
            let k_fine = self.num_occupied_slots.max(1);
            let k_total = k_fine + self.k_coarse;
            let q_norm = dot(&q, &q).sqrt().max(1e-8);

            let mut scores = vec![0.0f32; k_total];
            for i in 0..k_fine {
                let k_slice = &self.fine_keys[i * d..(i + 1) * d];
                let k_norm = dot(k_slice, k_slice).sqrt().max(1e-8);
                scores[i] = dot(&q, k_slice) / (q_norm * k_norm * temp);
            }
            for c in 0..self.k_coarse {
                let c_slice = &self.coarse_centroids[c * d..(c + 1) * d];
                let c_norm = dot(c_slice, c_slice).sqrt().max(1e-8);
                scores[k_fine + c] = dot(&q, c_slice) / (q_norm * c_norm * temp);
            }

            let mut probs = vec![0.0f32; k_total];
            softmax(&scores, &mut probs);

            let mut d_scores = vec![0.0f32; k_total];
            for i in 0..k_fine {
                let v_slice = &self.fine_vals[i * d..(i + 1) * d];
                d_scores[i] = probs[i] * dot(&d_ctx, v_slice);
            }
            for c in 0..self.k_coarse {
                let v_slice = &self.coarse_vals[c * d..(c + 1) * d];
                d_scores[k_fine + c] = probs[k_fine + c] * dot(&d_ctx, v_slice);
            }

            let sum_dp: f32 = d_scores.iter().sum();
            d_q.fill(0.0f32);

            for i in 0..k_fine {
                let d_s = (d_scores[i] - probs[i] * sum_dp) / temp;
                let k_slice = &self.fine_keys[i * d..(i + 1) * d];
                let k_norm = dot(k_slice, k_slice).sqrt().max(1e-8);
                let cos_sim = dot(&q, k_slice) / (q_norm * k_norm);
                for c in 0..d {
                    let grad_cos = (k_slice[c] / (q_norm * k_norm)) - (cos_sim * q[c] / (q_norm * q_norm));
                    d_q[c] += d_s * grad_cos;
                }
            }

            // 4. Query Projection Backward
            outer_product_accumulate(&d_q, x_t, 1.0, &mut gwq);
            matvec_transposed(&wq_v, &d_q, &mut tmp_d);
            vec_add_scaled(dh_in, &tmp_d, 1.0);

            // 5. Key and Value smooth alignment
            outer_product_accumulate(&d_q, x_t, 0.05, &mut gwk);
            outer_product_accumulate(&d_ctx, x_t, 0.05, &mut gwv);
        }
    }

    /// Targeted Needle-in-Haystack Retrieval Probe.
    pub fn probe_needle_recall(&mut self, context_len: usize, seed: u64) -> f32 {
        self.probe_needle_recall_with_salience(context_len, seed, 100.0, 1.0)
    }

    /// Targeted Needle-in-Haystack Retrieval Probe with configurable needle and distractor salience.
    pub fn probe_needle_recall_with_salience(
        &mut self,
        context_len: usize,
        seed: u64,
        needle_salience: f32,
        distractor_salience: f32,
    ) -> f32 {
        let d = self.d;
        let mut rng = StdRng::seed_from_u64(seed);

        // 1. Generate distinctive needle (K, V)
        let needle_key: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let needle_val: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        self.write_token(&needle_key, &needle_val, needle_salience);

        // 2. Stream distraction tokens
        for _ in 0..context_len {
            let dist_k: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let dist_v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            self.write_token(&dist_k, &dist_v, distractor_salience);
        }

        // 3. Query memory for needle
        let mut retrieved_val = vec![0.0f32; d];
        self.read_memory(&needle_key, &mut retrieved_val);

        // 4. Compute Cosine Similarity between retrieved value and true needle value
        let true_norm = dot(&needle_val, &needle_val).sqrt().max(1e-6);
        let retr_norm = dot(&retrieved_val, &retrieved_val).sqrt().max(1e-6);
        let cos_sim = dot(&needle_val, &retrieved_val) / (true_norm * retr_norm);

        cos_sim
    }
}
