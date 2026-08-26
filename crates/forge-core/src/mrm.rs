//! Multi-Resolution Memory (MRM): fixed-slot content-addressable store with
//! two resolution levels (fine-grain and coarse-summary), surprise-gated writes,
//! and rank-1 fast-weight adaptation.
//!
//! Design invariants:
//! - O(1) read/write per token (slot count fixed, independent of context length)
//! - Surprise gate: write to fine slots only when |h_t - predicted| > threshold
//! - Fast weights: rank-1 outer product update to W_o at test time, zero gradient steps

use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::softmax;
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ────────────────────────────────────────────────────────────────────
//  Routing & diagnostic stats
// ────────────────────────────────────────────────────────────────────

/// Per-token gate/write diagnostic record for Experiment E3.
#[derive(Debug, Clone)]
pub struct GateRecord {
    pub surprise_magnitude: f32,
    pub gate_value: f32,
    pub written: bool,
}

/// Aggregated per-run diagnostics.
#[derive(Debug, Clone, Default)]
pub struct MrmStats {
    pub total_tokens: usize,
    pub total_writes: usize,
    pub gate_records: Vec<GateRecord>,
    /// Recall experiment: position → recalled correctly?
    pub recall_results: Vec<(usize, bool)>,
}

impl MrmStats {
    pub fn skip_rate(&self) -> f32 {
        if self.total_tokens == 0 { return 0.0; }
        let skips = self.total_tokens - self.total_writes;
        skips as f32 / self.total_tokens as f32
    }
    pub fn write_rate(&self) -> f32 {
        if self.total_tokens == 0 { return 0.0; }
        self.total_writes as f32 / self.total_tokens as f32
    }
}

// ────────────────────────────────────────────────────────────────────
//  Multi-Resolution Memory
// ────────────────────────────────────────────────────────────────────

/// Fixed-capacity content-addressable memory with two resolution tiers.
///
/// Fine slots  (K_f): high-fidelity, surprise-gated recent storage.
/// Coarse slots(K_c): compressed summary, always updated (EMA-style).
///
/// Read:  query both banks, weighted-sum by softmax attention scores.
/// Write: fine ← gated by surprise; coarse ← EMA always.
#[derive(Debug, Clone)]
pub struct MultiResMemory {
    pub d: usize,
    pub k_fine: usize,   // number of fine slots
    pub k_coarse: usize, // number of coarse slots
    // Slot stores: (slots × d) row-major
    pub fine_keys:   Vec<f32>,
    pub fine_vals:   Vec<f32>,
    pub coarse_keys: Vec<f32>,
    pub coarse_vals: Vec<f32>,
    // Write pointer (ring buffer for fine slots)
    pub write_ptr: usize,
    // Surprise gate threshold (learned scalar, but treated as hyperparam here)
    pub surprise_threshold: f32,
    // EMA decay for coarse slots
    pub coarse_decay: f32,
    // Projections: W_k, W_v, W_q  each (d × d)
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_o: Vec<f32>,
    // Fast-weight rank-1 delta: W_fw = W_o + outer(u, v)
    pub fw_u: Vec<f32>, // (d)
    pub fw_v: Vec<f32>, // (d)
    pub fw_active: bool,
    pub stats: MrmStats,
    // Running mean for surprise detection (EMA of h)
    pub mean_h: Vec<f32>,
    mean_decay: f32,
}

impl MultiResMemory {
    pub fn new(d: usize, k_fine: usize, k_coarse: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0f32 / d as f32).sqrt();

        let rand_vec = |rng: &mut StdRng, n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
        };

        let fine_keys   = vec![0.0f32; k_fine * d];
        let fine_vals   = vec![0.0f32; k_fine * d];
        let coarse_keys = vec![0.0f32; k_coarse * d];
        let coarse_vals = vec![0.0f32; k_coarse * d];

        Self {
            d,
            k_fine,
            k_coarse,
            fine_keys,
            fine_vals,
            coarse_keys,
            coarse_vals,
            write_ptr: 0,
            surprise_threshold: 0.3,
            coarse_decay: 0.99,
            w_k: rand_vec(&mut rng, d * d),
            w_v: rand_vec(&mut rng, d * d),
            w_q: rand_vec(&mut rng, d * d),
            w_o: rand_vec(&mut rng, d * d),
            fw_u: vec![0.0f32; d],
            fw_v: vec![0.0f32; d],
            fw_active: false,
            stats: MrmStats::default(),
            mean_h: vec![0.0f32; d],
            mean_decay: 0.98,
        }
    }

    /// Compute surprise magnitude: L2 distance between h and running EMA mean.
    pub fn surprise(&self, h: &[f32]) -> f32 {
        let mut sq = 0.0f32;
        for i in 0..self.d {
            let diff = h[i] - self.mean_h[i];
            sq += diff * diff;
        }
        sq.sqrt()
    }

    /// Update running EMA mean of h.
    fn update_mean(&mut self, h: &[f32]) {
        let alpha = 1.0 - self.mean_decay;
        for i in 0..self.d {
            self.mean_h[i] = self.mean_decay * self.mean_h[i] + alpha * h[i];
        }
    }

    /// Full MRM forward: read + gated write. Returns output vector (d).
    /// record_gate: if true, appends to stats.gate_records (for E3).
    pub fn forward(&mut self, h: &[f32], output: &mut [f32], record_gate: bool) {
        let d = self.d;
        // Clone weight slices so we don't hold an immutable borrow on self
        // while also mutating (update_mean, stats, etc.).
        let w_q_local = self.w_q.clone();
        let w_k_local = self.w_k.clone();
        let w_v_local = self.w_v.clone();
        let w_o_local = self.w_o.clone();
        let fw_u_local = self.fw_u.clone();
        let fw_v_local = self.fw_v.clone();
        let fw_active  = self.fw_active;
        let wq = MatrixView::new(&w_q_local, d, d);
        let wk = MatrixView::new(&w_k_local, d, d);
        let wv = MatrixView::new(&w_v_local, d, d);
        let wo = MatrixView::new(&w_o_local, d, d);

        // ── 1. Project query ──────────────────────────────────────────
        let mut q = vec![0.0f32; d];
        matvec(&wq, h, &mut q);

        // ── 2. Project key and value for new token ────────────────────
        let mut new_k = vec![0.0f32; d];
        let mut new_v = vec![0.0f32; d];
        matvec(&wk, h, &mut new_k);
        matvec(&wv, h, &mut new_v);

        // ── 3. Surprise gate ─────────────────────────────────────────
        let surp = self.surprise(h);
        // Soft gate: g = sigmoid((surp - threshold) * 10)
        let gate = 1.0f32 / (1.0f32 + (-(surp - self.surprise_threshold) * 10.0).exp());
        let written = gate > 0.5;

        if record_gate {
            self.stats.gate_records.push(GateRecord {
                surprise_magnitude: surp,
                gate_value: gate,
                written,
            });
        }
        self.stats.total_tokens += 1;
        if written {
            self.stats.total_writes += 1;
        }
        self.update_mean(h);

        // ── 4. Write to fine slots (ring buffer, gated) ───────────────
        if written {
            let ptr = self.write_ptr % self.k_fine;
            self.fine_keys[ptr * d..(ptr + 1) * d].copy_from_slice(&new_k);
            self.fine_vals[ptr * d..(ptr + 1) * d].copy_from_slice(&new_v);
            self.write_ptr = self.write_ptr.wrapping_add(1);
        }

        // ── 5. EMA update coarse slots ────────────────────────────────
        // Update the slot with highest cosine similarity to new_k.
        let mut best_c = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for s in 0..self.k_coarse {
            let ck = &self.coarse_keys[s * d..(s + 1) * d];
            let sim = dot(&new_k, ck);
            if sim > best_s { best_s = sim; best_c = s; }
        }
        let decay = self.coarse_decay;
        let alpha = 1.0 - decay;
        for i in 0..d {
            self.coarse_keys[best_c * d + i] = decay * self.coarse_keys[best_c * d + i] + alpha * new_k[i];
            self.coarse_vals[best_c * d + i] = decay * self.coarse_vals[best_c * d + i] + alpha * new_v[i];
        }

        // ── 6. Read: attention over fine + coarse ────────────────────
        let total_slots = self.k_fine + self.k_coarse;
        let mut scores = vec![f32::NEG_INFINITY; total_slots];
        let mut attn_vals = vec![0.0f32; d];

        // Fine slots
        for s in 0..self.k_fine {
            let fk = &self.fine_keys[s * d..(s + 1) * d];
            // Only attend to written slots (non-zero norm)
            let norm_sq: f32 = fk.iter().map(|&x| x * x).sum();
            if norm_sq > 1e-6 {
                scores[s] = dot(&q, fk);
            }
        }
        // Coarse slots
        for s in 0..self.k_coarse {
            let ck = &self.coarse_keys[s * d..(s + 1) * d];
            let norm_sq: f32 = ck.iter().map(|&x| x * x).sum();
            if norm_sq > 1e-6 {
                scores[self.k_fine + s] = dot(&q, ck);
            }
        }

        let mut attn_probs = vec![0.0f32; total_slots];
        softmax(&scores, &mut attn_probs);

        // Weighted value read
        for s in 0..self.k_fine {
            let w = attn_probs[s];
            if w > 1e-6 {
                let fv = &self.fine_vals[s * d..(s + 1) * d];
                vec_add_scaled(&mut attn_vals, fv, w);
            }
        }
        for s in 0..self.k_coarse {
            let w = attn_probs[self.k_fine + s];
            if w > 1e-6 {
                let cv = &self.coarse_vals[s * d..(s + 1) * d];
                vec_add_scaled(&mut attn_vals, cv, w);
            }
        }

        // ── 7. Output projection + residual ──────────────────────────
        // If fast weights active: W_eff = W_o + fw_u ⊗ fw_v^T
        // output = W_eff * attn_vals + h
        matvec(&wo, &attn_vals, output);
        if fw_active {
            // Apply rank-1 delta: output += (fw_v · attn_vals) * fw_u
            let scale = dot(&fw_v_local, &attn_vals);
            vec_add_scaled(output, &fw_u_local, scale);
        }
        vec_add_scaled(output, h, 1.0); // residual
    }

    /// Rank-1 fast-weight update (test-time, zero gradient steps).
    /// Computes u, v from (key_example, val_example) and stores as rank-1 delta.
    /// u = W_o * val_example  (direction to add)
    /// v = key_example        (condition key)
    pub fn fast_weight_update(&mut self, key_example: &[f32], val_example: &[f32]) {
        let d = self.d;
        let wo = MatrixView::new(&self.w_o, d, d);

        // u = W_o * val_example
        matvec(&wo, val_example, &mut self.fw_u);
        self.fw_v.copy_from_slice(key_example);
        self.fw_active = true;
    }

    pub fn reset_fast_weights(&mut self) {
        self.fw_u.fill(0.0f32);
        self.fw_v.fill(0.0f32);
        self.fw_active = false;
    }

    /// Write a specific key/value pair directly (for recall experiments).
    pub fn force_write(&mut self, key: &[f32], val: &[f32]) {
        let d = self.d;
        let ptr = self.write_ptr % self.k_fine;
        let wk = MatrixView::new(&self.w_k, d, d);
        let wv = MatrixView::new(&self.w_v, d, d);
        let mut pk = vec![0.0f32; d];
        let mut pv = vec![0.0f32; d];
        matvec(&wk, key, &mut pk);
        matvec(&wv, val, &mut pv);
        self.fine_keys[ptr * d..(ptr + 1) * d].copy_from_slice(&pk);
        self.fine_vals[ptr * d..(ptr + 1) * d].copy_from_slice(&pv);
        self.write_ptr = self.write_ptr.wrapping_add(1);
        self.stats.total_tokens += 1;
        self.stats.total_writes += 1;
    }

    /// Recall: retrieve the nearest value for a query h. Returns cosine sim to best slot.
    pub fn recall(&self, query_h: &[f32]) -> (Vec<f32>, f32) {
        let d = self.d;
        let wq = MatrixView::new(&self.w_q, d, d);
        let wo = MatrixView::new(&self.w_o, d, d);

        let mut q = vec![0.0f32; d];
        matvec(&wq, query_h, &mut q);

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_v = vec![0.0f32; d];

        for s in 0..self.k_fine {
            let fk = &self.fine_keys[s * d..(s + 1) * d];
            let norm_sq: f32 = fk.iter().map(|&x| x * x).sum();
            if norm_sq < 1e-6 { continue; }
            let sim = dot(&q, fk) / (norm_sq.sqrt().max(1e-6) * (dot(&q, &q).sqrt().max(1e-6)));
            if sim > best_sim {
                best_sim = sim;
                best_v = self.fine_vals[s * d..(s + 1) * d].to_vec();
            }
        }

        let mut out = vec![0.0f32; d];
        matvec(&wo, &best_v, &mut out);
        vec_add_scaled(&mut out, query_h, 1.0);
        (out, best_sim)
    }

    pub fn param_count(&self) -> usize {
        4 * self.d * self.d  // w_k, w_v, w_q, w_o
    }
}
