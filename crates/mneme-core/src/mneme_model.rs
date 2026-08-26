//! MNEME Full Architecture: Cache-Resident Recurrent Trunk + Depth-Recurrence + Sparse Knowledge Tier.

use crate::delta_rule::{DeltaRuleGrads, DeltaRuleMode, GatedDeltaRule};
use crate::experts::{ExpertGrads, ProductKeyExpertBank};
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::cross_entropy_loss_and_grad;
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Architecture configuration for MNEME.
#[derive(Debug, Clone, Copy)]
pub struct MnemeConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub n_unique_blocks: usize, // S = 2-5
    pub n_passes: usize,        // R = 1-8
    pub d_ff: usize,
    pub n_experts: usize,       // E = 0 (R0), 16 (R1), 64 (R2), 256 (R3)
    pub top_k_experts: usize,   // k = 4
    pub delta_mode: DeltaRuleMode,
    pub adapter_rank: usize,    // r = 4 (HARDPOINT §2.1)
    pub quant_bits: usize,      // 32 (FP32), 8 (INT8), 6 (W6), 4 (W4)
}

impl MnemeConfig {
    pub fn nano_default() -> Self {
        Self {
            d_model: 128,
            d_state: 32,
            n_unique_blocks: 2,
            n_passes: 2,
            d_ff: 256,
            n_experts: 64,
            top_k_experts: 4,
            delta_mode: DeltaRuleMode::FullGatedErasure,
            adapter_rank: 4,
            quant_bits: 32,
        }
    }
}

/// Gradients for a single MNEME block.
#[derive(Debug, Clone)]
pub struct MnemeBlockGrads {
    pub delta_grads: DeltaRuleGrads,
    pub grad_w1: Vec<f32>,
    pub grad_w1u: Vec<f32>,
    pub grad_w2: Vec<f32>,
    pub expert_grads: Option<ExpertGrads>,
    pub grad_adapters_a: Vec<Vec<f32>>, // per pass: (d x r)
    pub grad_adapters_b: Vec<Vec<f32>>, // per pass: (r x d)
}

impl MnemeBlockGrads {
    pub fn new(d: usize, d_state: usize, d_ff: usize, n_experts: usize, n_passes: usize, r: usize) -> Self {
        let m = ((n_experts as f32).sqrt().round() as usize).max(2);
        let expert_grads = if n_experts > 0 {
            Some(ExpertGrads::new(d, m))
        } else {
            None
        };

        let grad_adapters_a = (0..n_passes).map(|_| vec![0.0f32; d * r]).collect();
        let grad_adapters_b = (0..n_passes).map(|_| vec![0.0f32; r * d]).collect();

        Self {
            delta_grads: DeltaRuleGrads::new(d, d_state),
            grad_w1: vec![0.0f32; d_ff * d],
            grad_w1u: vec![0.0f32; d_ff * d],
            grad_w2: vec![0.0f32; d * d_ff],
            expert_grads,
            grad_adapters_a,
            grad_adapters_b,
        }
    }

    pub fn zero(&mut self) {
        self.delta_grads.zero();
        self.grad_w1.fill(0.0f32);
        self.grad_w1u.fill(0.0f32);
        self.grad_w2.fill(0.0f32);
        if let Some(ref mut eg) = self.expert_grads {
            eg.zero();
        }
        for ga in &mut self.grad_adapters_a { ga.fill(0.0f32); }
        for gb in &mut self.grad_adapters_b { gb.fill(0.0f32); }
    }

    pub fn add(&mut self, other: &MnemeBlockGrads) {
        self.delta_grads.add(&other.delta_grads);
        for (a, &b) in self.grad_w1.iter_mut().zip(other.grad_w1.iter()) { *a += b; }
        for (a, &b) in self.grad_w1u.iter_mut().zip(other.grad_w1u.iter()) { *a += b; }
        for (a, &b) in self.grad_w2.iter_mut().zip(other.grad_w2.iter()) { *a += b; }
        if let (Some(ref mut eg), Some(ref oeg)) = (&mut self.expert_grads, &other.expert_grads) {
            eg.add(oeg);
        }
        for (ga, oga) in self.grad_adapters_a.iter_mut().zip(other.grad_adapters_a.iter()) {
            for (a, &b) in ga.iter_mut().zip(oga.iter()) { *a += b; }
        }
        for (gb, ogb) in self.grad_adapters_b.iter_mut().zip(other.grad_adapters_b.iter()) {
            for (a, &b) in gb.iter_mut().zip(ogb.iter()) { *a += b; }
        }
    }
}

/// Single unique MNEME Block (reused across R passes).
#[derive(Debug, Clone)]
pub struct MnemeBlock {
    pub d_model: usize,
    pub d_state: usize,
    pub d_ff: usize,
    pub delta_layer: GatedDeltaRule,
    pub w1: Vec<f32>,  // SwiGLU gate (d_ff x d)
    pub w1u: Vec<f32>, // SwiGLU up   (d_ff x d)
    pub w2: Vec<f32>,  // SwiGLU down (d x d_ff)
    pub expert_tier: Option<ProductKeyExpertBank>,
    pub adapters_a: Vec<Vec<f32>>, // per pass: (d x r)
    pub adapters_b: Vec<Vec<f32>>, // per pass: (r x d)
    pub adapter_rank: usize,
}

impl MnemeBlock {
    pub fn new(
        d_model: usize,
        d_state: usize,
        d_ff: usize,
        n_experts: usize,
        top_k: usize,
        delta_mode: DeltaRuleMode,
        n_passes: usize,
        r: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_ff = (1.0f32 / d_ff as f32).sqrt();
        let scale_r = (1.0f32 / r as f32).sqrt();

        let delta_layer = GatedDeltaRule::new(d_model, d_state, delta_mode, seed + 1);

        let w1 = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w1u = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w2 = (0..d_model * d_ff).map(|_| rng.gen_range(-scale_ff..scale_ff)).collect();

        let expert_tier = if n_experts > 0 {
            Some(ProductKeyExpertBank::new(d_model, n_experts, top_k, seed + 2))
        } else {
            None
        };

        let mut adapters_a = Vec::with_capacity(n_passes);
        let mut adapters_b = Vec::with_capacity(n_passes);
        for _ in 0..n_passes {
            let a: Vec<f32> = (0..d_model * r).map(|_| rng.gen_range(-scale_r..scale_r)).collect();
            let b: Vec<f32> = vec![0.0f32; r * d_model]; // zero init B so adapter starts as identity
            adapters_a.push(a);
            adapters_b.push(b);
        }

        Self {
            d_model,
            d_state,
            d_ff,
            delta_layer,
            w1,
            w1u,
            w2,
            expert_tier,
            adapters_a,
            adapters_b,
            adapter_rank: r,
        }
    }
}

/// Gradients for full MNEME Model.
#[derive(Debug, Clone)]
pub struct MnemeModelGrads {
    pub grad_embed: Vec<f32>,
    pub grad_pos_embed: Vec<f32>,
    pub block_grads: Vec<MnemeBlockGrads>,
    pub grad_head: Vec<f32>,
}

impl MnemeModelGrads {
    pub fn new(vocab_size: usize, d_model: usize, max_seq: usize, blocks: &[MnemeBlock], n_passes: usize) -> Self {
        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            grad_pos_embed: vec![0.0f32; max_seq * d_model],
            block_grads: blocks.iter().map(|b| {
                let n_exp = b.expert_tier.as_ref().map(|e| e.total_experts).unwrap_or(0);
                MnemeBlockGrads::new(d_model, b.d_state, b.d_ff, n_exp, n_passes, b.adapter_rank)
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

    pub fn add(&mut self, other: &MnemeModelGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (a, &b) in self.grad_pos_embed.iter_mut().zip(other.grad_pos_embed.iter()) { *a += b; }
        for (bg, obg) in self.block_grads.iter_mut().zip(other.block_grads.iter()) { bg.add(obg); }
        for (a, &b) in self.grad_head.iter_mut().zip(other.grad_head.iter()) { *a += b; }
    }
}

/// Complete MNEME Model.
#[derive(Debug, Clone)]
pub struct MnemeModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub config: MnemeConfig,
    pub embeddings: Vec<f32>,
    pub pos_embeddings: Vec<f32>,
    pub unique_blocks: Vec<MnemeBlock>, // S unique blocks
    pub head: Vec<f32>,
}

impl MnemeModel {
    pub fn new(vocab_size: usize, max_seq_len: usize, config: MnemeConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / config.d_model as f32).sqrt();

        let embeddings = (0..vocab_size * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let pos_embeddings = (0..max_seq_len * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let head = (0..vocab_size * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let unique_blocks = (0..config.n_unique_blocks)
            .map(|s| {
                MnemeBlock::new(
                    config.d_model,
                    config.d_state,
                    config.d_ff,
                    config.n_experts,
                    config.top_k_experts,
                    config.delta_mode,
                    config.n_passes,
                    config.adapter_rank,
                    seed + 100 + s as u64,
                )
            })
            .collect();

        Self {
            vocab_size,
            d_model: config.d_model,
            max_seq_len,
            config,
            embeddings,
            pos_embeddings,
            unique_blocks,
            head,
        }
    }

    /// Parameter accounting: returns (total_params, active_params, inference_dram_bytes, resident_l3_bytes).
    pub fn parameter_metrics(&self) -> (usize, usize, usize, usize) {
        let d = self.d_model;
        let v = self.vocab_size;

        let embed_params = v * d + self.max_seq_len * d;
        let head_params = v * d;

        let mut resident_trunk_params = embed_params + head_params;
        let mut sparse_expert_params = 0usize;
        let mut active_expert_params = 0usize;

        for block in &self.unique_blocks {
            resident_trunk_params += block.delta_layer.param_count();
            resident_trunk_params += (2 * block.d_ff * d) + (d * block.d_ff); // SwiGLU
            resident_trunk_params += self.config.n_passes * 2 * (d * block.adapter_rank); // adapters

            if let Some(ref exp) = block.expert_tier {
                let (dense_p, sparse_p) = exp.param_count();
                resident_trunk_params += dense_p;
                sparse_expert_params += sparse_p;
                active_expert_params += exp.top_k * d;
            }
        }

        let total_params = resident_trunk_params + sparse_expert_params;
        let active_params = resident_trunk_params + active_expert_params;

        // In MNEME: resident trunk sits in L3 cache (0 DRAM bytes after warm-up).
        // Only sparse expert rows read from DRAM per token!
        let dram_bytes_per_token = active_expert_params * 4 + 4 * d; // active expert vectors + embed
        let resident_l3_bytes = resident_trunk_params * 4;

        (total_params, active_params, dram_bytes_per_token, resident_l3_bytes)
    }

    /// Forward & Backward pass with Depth-Recurrence (R passes over S unique blocks).
    pub fn forward_backward_sequence(
        &mut self,
        x_seq: &[usize],
        y_seq: &[usize],
        passes_r: usize,
        grads: &mut MnemeModelGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;
        let s_blocks = self.config.n_unique_blocks;

        // 1. Initial Embedding
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

        // Cache for all (pass p, block s) activations for BPTT
        // Storing activations: [p * S + s] => (h_in, h_after_delta, h_after_swiglu, h_after_expert)
        let total_sites = passes_r * s_blocks;
        let mut h_sites = Vec::with_capacity(total_sites + 1);
        h_sites.push(h_curr.clone());

        for p in 0..passes_r {
            for s in 0..s_blocks {
                let block = &self.unique_blocks[s];
                let h_in = h_sites.last().unwrap().clone();
                let mut h_delta_out = vec![0.0f32; t_len * d];

                let mut q_c = vec![0.0f32; t_len * block.d_state];
                let mut k_c = vec![0.0f32; t_len * block.d_state];
                let mut v_c = vec![0.0f32; t_len * block.d_state];
                let mut a_c = vec![0.0f32; t_len];
                let mut b_c = vec![0.0f32; t_len];
                let mut s_c = vec![0.0f32; (t_len + 1) * block.d_state * block.d_state];

                block.delta_layer.forward_sequence(
                    &h_in,
                    t_len,
                    &mut h_delta_out,
                    &mut q_c,
                    &mut k_c,
                    &mut v_c,
                    &mut a_c,
                    &mut b_c,
                    &mut s_c,
                );

                // SwiGLU Dense Channel Mixer Forward + Residual
                let mut h_swiglu_out = h_delta_out.clone();
                let w1_v = MatrixView::new(&block.w1, block.d_ff, d);
                let w1u_v = MatrixView::new(&block.w1u, block.d_ff, d);
                let w2_v = MatrixView::new(&block.w2, d, block.d_ff);

                for t in 0..t_len {
                    let ht = &h_delta_out[t * d..(t + 1) * d];
                    let mut gate = vec![0.0f32; block.d_ff];
                    let mut up = vec![0.0f32; block.d_ff];
                    matvec(&w1_v, ht, &mut gate);
                    matvec(&w1u_v, ht, &mut up);

                    let mut ff = vec![0.0f32; block.d_ff];
                    for i in 0..block.d_ff {
                        let g = gate[i];
                        let silu = g / (1.0 + (-g).exp());
                        ff[i] = silu * up[i];
                    }
                    let mut ff_out = vec![0.0f32; d];
                    matvec(&w2_v, &ff, &mut ff_out);

                    // Add per-site adapter: h += A_p * (B_p * h)
                    let r = block.adapter_rank;
                    let p_idx = p % block.adapters_b.len().max(1);
                    let b_view = MatrixView::new(&block.adapters_b[p_idx], r, d);
                    let a_view = MatrixView::new(&block.adapters_a[p_idx], d, r);
                    let mut adapt_mid = vec![0.0f32; r];
                    let mut adapt_out = vec![0.0f32; d];
                    matvec(&b_view, ht, &mut adapt_mid);
                    matvec(&a_view, &adapt_mid, &mut adapt_out);

                    let out_slice = &mut h_swiglu_out[t * d..(t + 1) * d];
                    vec_add_scaled(out_slice, &ff_out, 1.0);
                    vec_add_scaled(out_slice, &adapt_out, 1.0);
                }

                // Sparse Expert Bank (if present)
                let mut h_next = h_swiglu_out.clone();
                if let Some(ref exp) = block.expert_tier {
                    let mut exp_out = vec![0.0f32; t_len * d];
                    let mut idx_c = vec![0usize; t_len * exp.top_k];
                    let mut w_c = vec![0.0f32; t_len * exp.top_k];
                    let mut v_c = vec![0.0f32; t_len * exp.top_k * d];
                    let mut q_c = vec![0.0f32; t_len * d];

                    exp.forward_sequence(
                        &h_swiglu_out,
                        t_len,
                        &mut exp_out,
                        &mut idx_c,
                        &mut w_c,
                        &mut v_c,
                        &mut q_c,
                    );
                    h_next = exp_out;
                }

                h_sites.push(h_next);
            }
        }

        // 4. Output Head Loss & Backward
        let final_h = h_sites.last().unwrap();
        let head_view = MatrixView::new(&self.head, v, d);
        let mut grad_head_view = MatrixViewMut::new(&mut grads.grad_head, v, d);
        let mut delta_final = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let ht = &final_h[t * d..(t + 1) * d];
            let mut logits = vec![0.0f32; v];
            let mut probs = vec![0.0f32; v];
            let mut pred_grad = vec![0.0f32; v];

            matvec(&head_view, ht, &mut logits);
            let loss = cross_entropy_loss_and_grad(&logits, y_seq[t], &mut probs, &mut pred_grad);
            total_loss += loss;

            outer_product_accumulate(&pred_grad, ht, 1.0, &mut grad_head_view);
            let d_ht = &mut delta_final[t * d..(t + 1) * d];
            matvec_transposed(&head_view, &pred_grad, d_ht);
        }

        // Backward through depth-recurrence
        let mut delta_upstream = delta_final;
        for p in (0..passes_r).rev() {
            for s in (0..s_blocks).rev() {
                let block = &self.unique_blocks[s];
                let b_grads = &mut grads.block_grads[s];

                // Expert backward (if present)
                let delta_expert_in = delta_upstream.clone();

                // SwiGLU backward
                let w1_v = MatrixView::new(&block.w1, block.d_ff, d);
                let w1u_v = MatrixView::new(&block.w1u, block.d_ff, d);
                let w2_v = MatrixView::new(&block.w2, d, block.d_ff);
                let mut gw1 = MatrixViewMut::new(&mut b_grads.grad_w1, block.d_ff, d);
                let mut gw1u = MatrixViewMut::new(&mut b_grads.grad_w1u, block.d_ff, d);
                let mut gw2 = MatrixViewMut::new(&mut b_grads.grad_w2, d, block.d_ff);

                let mut delta_delta_out = vec![0.0f32; t_len * d];

                for t in 0..t_len {
                    let dh = &delta_expert_in[t * d..(t + 1) * d];
                    let site_idx = p * s_blocks + s;
                    let ht = &h_sites[site_idx][t * d..(t + 1) * d];

                    let mut gate = vec![0.0f32; block.d_ff];
                    let mut up = vec![0.0f32; block.d_ff];
                    matvec(&w1_v, ht, &mut gate);
                    matvec(&w1u_v, ht, &mut up);

                    let mut ff = vec![0.0f32; block.d_ff];
                    for i in 0..block.d_ff {
                        let g = gate[i];
                        ff[i] = (g / (1.0 + (-g).exp())) * up[i];
                    }

                    outer_product_accumulate(dh, &ff, 1.0, &mut gw2);

                    let mut d_ff = vec![0.0f32; block.d_ff];
                    matvec_transposed(&w2_v, dh, &mut d_ff);

                    let mut d_gate = vec![0.0f32; block.d_ff];
                    let mut d_up = vec![0.0f32; block.d_ff];
                    for i in 0..block.d_ff {
                        let g = gate[i];
                        let sig = 1.0 / (1.0 + (-g).exp());
                        let silu = g * sig;
                        d_gate[i] = d_ff[i] * up[i] * (sig + silu * (1.0 - sig));
                        d_up[i] = d_ff[i] * silu;
                    }

                    outer_product_accumulate(&d_gate, ht, 1.0, &mut gw1);
                    outer_product_accumulate(&d_up, ht, 1.0, &mut gw1u);

                    let mut d_ht_accum = vec![0.0f32; d];
                    matvec_transposed(&w1_v, &d_gate, &mut d_ht_accum);
                    matvec_transposed(&w1u_v, &d_up, &mut d_ht_accum);

                    // Residual from SwiGLU + Delta layer
                    let d_delta_t = &mut delta_delta_out[t * d..(t + 1) * d];
                    d_delta_t.copy_from_slice(dh);
                    vec_add_scaled(d_delta_t, &d_ht_accum, 1.0);
                }

                // Delta mixer backward
                let site_in_idx = p * s_blocks + s;
                let h_site_in = &h_sites[site_in_idx];
                let mut delta_site_in = vec![0.0f32; t_len * d];

                let mut q_c = vec![0.0f32; t_len * block.d_state];
                let mut k_c = vec![0.0f32; t_len * block.d_state];
                let mut v_c = vec![0.0f32; t_len * block.d_state];
                let mut a_c = vec![0.0f32; t_len];
                let mut b_c = vec![0.0f32; t_len];
                let mut s_c = vec![0.0f32; (t_len + 1) * block.d_state * block.d_state];

                block.delta_layer.forward_sequence(
                    h_site_in,
                    t_len,
                    &mut vec![0.0f32; t_len * d],
                    &mut q_c,
                    &mut k_c,
                    &mut v_c,
                    &mut a_c,
                    &mut b_c,
                    &mut s_c,
                );

                block.delta_layer.backward_sequence(
                    h_site_in,
                    t_len,
                    &delta_delta_out,
                    &mut delta_site_in,
                    &q_c,
                    &k_c,
                    &v_c,
                    &a_c,
                    &b_c,
                    &s_c,
                    &mut b_grads.delta_grads,
                );

                delta_upstream = delta_site_in;
            }
        }

        // Embedding Backward
        for t in 0..t_len {
            let tok = x_seq[t];
            let pos = t.min(self.max_seq_len - 1);
            let dh = &delta_upstream[t * d..(t + 1) * d];
            let emb_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            let pos_slice = &mut grads.grad_pos_embed[pos * d..(pos + 1) * d];
            for i in 0..d {
                emb_slice[i] += dh[i];
                pos_slice[i] += dh[i];
            }
        }

        total_loss
    }

    /// Quantize weights in-place for Experiment E7 (INT8, W6, W4).
    pub fn apply_quantization_simulation(&mut self, bits: usize) {
        if bits >= 32 { return; }
        let levels = (1 << bits) as f32;
        let quantize_slice = |slice: &mut [f32]| {
            let max_val = slice.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())).max(1e-8);
            let scale = (levels / 2.0 - 1.0) / max_val;
            for val in slice.iter_mut() {
                let q = (*val * scale).round().clamp(-(levels / 2.0), levels / 2.0 - 1.0);
                *val = q / scale;
            }
        };

        quantize_slice(&mut self.embeddings);
        quantize_slice(&mut self.pos_embeddings);
        quantize_slice(&mut self.head);

        for block in &mut self.unique_blocks {
            quantize_slice(&mut block.delta_layer.wq);
            quantize_slice(&mut block.delta_layer.wk);
            quantize_slice(&mut block.delta_layer.wv);
            quantize_slice(&mut block.delta_layer.wo);
            quantize_slice(&mut block.w1);
            quantize_slice(&mut block.w1u);
            quantize_slice(&mut block.w2);
        }
    }
}
