//! TESSERA-Q: Quality-First Architecture with Causal Temporal Attention + Pre-LN RMSNorm + Progressive Folding + MRM-v2.

use crate::mrm_v2::{MrmV2Grads, MultiResMemoryV2};
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::{cross_entropy_loss_and_grad, softmax};
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Element-wise RMSNorm forward
#[inline]
pub fn rms_norm(x: &[f32], out: &mut [f32], eps: f32) -> f32 {
    let d = x.len() as f32;
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms = (sum_sq / d + eps).sqrt();
    let inv_rms = 1.0f32 / rms;
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v * inv_rms;
    }
    rms
}

/// Element-wise RMSNorm backward
#[inline]
pub fn rms_norm_backward(x: &[f32], rms: f32, grad_out: &[f32], grad_in: &mut [f32]) {
    let d = x.len() as f32;
    let inv_rms = 1.0f32 / rms;
    let inv_rms3 = inv_rms * inv_rms * inv_rms;
    let mut dot_go_x = 0.0f32;
    for (&go, &xi) in grad_out.iter().zip(x.iter()) {
        dot_go_x += go * xi;
    }
    let scale_dot = dot_go_x / (d * (rms * rms * rms));
    for ((gi, &go), &xi) in grad_in.iter_mut().zip(grad_out.iter()).zip(x.iter()) {
        *gi += go * inv_rms - xi * scale_dot;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TesseraConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_stages: usize,      // P = 2 progressive hierarchy stages
    pub adapter_rank: usize,  // r = 8 per-stage low-rank modulation
    pub use_mrm_v2: bool,     // Ablation flag (Arm B vs Arm C)
    pub k_fine_slots: usize,  // 128 fine slots
    pub k_coarse_slots: usize,// 16 coarse slots
}

impl TesseraConfig {
    pub fn nano_default() -> Self {
        Self {
            d_model: 128,
            d_ff: 512,
            n_stages: 2,
            adapter_rank: 8,
            use_mrm_v2: true,
            k_fine_slots: 128,
            k_coarse_slots: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TesseraStageGrads {
    pub grad_wq: Vec<f32>,
    pub grad_wk: Vec<f32>,
    pub grad_wv: Vec<f32>,
    pub grad_wo: Vec<f32>,
    pub grad_w1: Vec<f32>,
    pub grad_w1u: Vec<f32>,
    pub grad_w2: Vec<f32>,
    pub grad_adapter_u: Vec<f32>,
    pub grad_adapter_v: Vec<f32>,
    pub mrm_grads: Option<MrmV2Grads>,
}

impl TesseraStageGrads {
    pub fn new(d: usize, d_ff: usize, r: usize, use_mrm: bool) -> Self {
        Self {
            grad_wq: vec![0.0f32; d * d],
            grad_wk: vec![0.0f32; d * d],
            grad_wv: vec![0.0f32; d * d],
            grad_wo: vec![0.0f32; d * d],
            grad_w1: vec![0.0f32; d_ff * d],
            grad_w1u: vec![0.0f32; d_ff * d],
            grad_w2: vec![0.0f32; d * d_ff],
            grad_adapter_u: vec![0.0f32; d * r],
            grad_adapter_v: vec![0.0f32; r * d],
            mrm_grads: if use_mrm { Some(MrmV2Grads::new(d)) } else { None },
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32);
        self.grad_wk.fill(0.0f32);
        self.grad_wv.fill(0.0f32);
        self.grad_wo.fill(0.0f32);
        self.grad_w1.fill(0.0f32);
        self.grad_w1u.fill(0.0f32);
        self.grad_w2.fill(0.0f32);
        self.grad_adapter_u.fill(0.0f32);
        self.grad_adapter_v.fill(0.0f32);
        if let Some(ref mut mg) = self.mrm_grads { mg.zero(); }
    }

    pub fn add(&mut self, other: &TesseraStageGrads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_wk.iter_mut().zip(other.grad_wk.iter()) { *a += b; }
        for (a, &b) in self.grad_wv.iter_mut().zip(other.grad_wv.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        for (a, &b) in self.grad_w1.iter_mut().zip(other.grad_w1.iter()) { *a += b; }
        for (a, &b) in self.grad_w1u.iter_mut().zip(other.grad_w1u.iter()) { *a += b; }
        for (a, &b) in self.grad_w2.iter_mut().zip(other.grad_w2.iter()) { *a += b; }
        for (a, &b) in self.grad_adapter_u.iter_mut().zip(other.grad_adapter_u.iter()) { *a += b; }
        for (a, &b) in self.grad_adapter_v.iter_mut().zip(other.grad_adapter_v.iter()) { *a += b; }
        if let (Some(ref mut mg), Some(ref omg)) = (&mut self.mrm_grads, &other.mrm_grads) {
            mg.add(omg);
        }
    }
}

/// A Progressive Hierarchy Stage with Temporal Causal Attention + Pre-LN RMSNorm + SwiGLU + MRM-v2.
#[derive(Debug, Clone)]
pub struct TesseraStage {
    pub d_model: usize,
    pub d_ff: usize,
    pub adapter_rank: usize,
    // Temporal Causal Self-Attention
    pub wq: Vec<f32>, // (d x d)
    pub wk: Vec<f32>, // (d x d)
    pub wv: Vec<f32>, // (d x d)
    pub wo: Vec<f32>, // (d x d)
    // Dense Channel Mixer (SwiGLU)
    pub w1: Vec<f32>,  // (d_ff x d)
    pub w1u: Vec<f32>, // (d_ff x d)
    pub w2: Vec<f32>,  // (d x d_ff)
    // Stage-private low-rank modulation adapter
    pub adapter_u: Vec<f32>, // (d x r)
    pub adapter_v: Vec<f32>, // (r x d)
    // Working Memory (MRM-v2)
    pub mrm: Option<MultiResMemoryV2>,
}

impl TesseraStage {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        adapter_rank: usize,
        use_mrm: bool,
        k_fine: usize,
        k_coarse: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();
        let scale_ff = (1.0f32 / d_ff as f32).sqrt();
        let scale_r = (1.0f32 / adapter_rank as f32).sqrt();

        let wq = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wk = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wv = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wo = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let w1 = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w1u = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w2 = (0..d_model * d_ff).map(|_| rng.gen_range(-scale_ff..scale_ff)).collect();

        let adapter_u = (0..d_model * adapter_rank).map(|_| rng.gen_range(-scale_r..scale_r)).collect();
        let adapter_v = vec![0.0f32; adapter_rank * d_model]; // zero init

        let mrm = if use_mrm {
            Some(MultiResMemoryV2::new(d_model, k_fine, k_coarse, seed + 10))
        } else {
            None
        };

        Self {
            d_model,
            d_ff,
            adapter_rank,
            wq,
            wk,
            wv,
            wo,
            w1,
            w1u,
            w2,
            adapter_u,
            adapter_v,
            mrm,
        }
    }

    pub fn param_count(&self) -> usize {
        let attn = 4 * (self.d_model * self.d_model);
        let dense = 2 * (self.d_ff * self.d_model) + (self.d_model * self.d_ff);
        let adapter = 2 * (self.d_model * self.adapter_rank);
        let mrm_p = self.mrm.as_ref().map(|m| m.param_count()).unwrap_or(0);
        attn + dense + adapter + mrm_p
    }
}

#[derive(Debug, Clone)]
pub struct TesseraModelGrads {
    pub grad_embed: Vec<f32>,
    pub grad_pos_embed: Vec<f32>,
    pub stage_grads: Vec<TesseraStageGrads>,
    pub grad_head: Vec<f32>,
}

impl TesseraModelGrads {
    pub fn new(vocab_size: usize, d_model: usize, max_seq: usize, stages: &[TesseraStage]) -> Self {
        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            grad_pos_embed: vec![0.0f32; max_seq * d_model],
            stage_grads: stages.iter().map(|s| {
                TesseraStageGrads::new(d_model, s.d_ff, s.adapter_rank, s.mrm.is_some())
            }).collect(),
            grad_head: vec![0.0f32; vocab_size * d_model],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embed.fill(0.0f32);
        self.grad_pos_embed.fill(0.0f32);
        for sg in &mut self.stage_grads { sg.zero(); }
        self.grad_head.fill(0.0f32);
    }

    pub fn add(&mut self, other: &TesseraModelGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (a, &b) in self.grad_pos_embed.iter_mut().zip(other.grad_pos_embed.iter()) { *a += b; }
        for (sg, osg) in self.stage_grads.iter_mut().zip(other.stage_grads.iter()) { sg.add(osg); }
        for (a, &b) in self.grad_head.iter_mut().zip(other.grad_head.iter()) { *a += b; }
    }
}

/// Full TESSERA Architecture with Pre-LN RMSNorm.
#[derive(Debug, Clone)]
pub struct TesseraModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub config: TesseraConfig,
    pub embeddings: Vec<f32>,
    pub pos_embeddings: Vec<f32>,
    pub stages: Vec<TesseraStage>,
    pub head: Vec<f32>,
}

impl TesseraModel {
    pub fn new(vocab_size: usize, max_seq_len: usize, config: TesseraConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / config.d_model as f32).sqrt();

        let embeddings = (0..vocab_size * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let pos_embeddings = (0..max_seq_len * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let head = (0..vocab_size * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let stages = (0..config.n_stages)
            .map(|p| {
                TesseraStage::new(
                    config.d_model,
                    config.d_ff,
                    config.adapter_rank,
                    config.use_mrm_v2 && p == config.n_stages - 1, // Attach MRM to final stage
                    config.k_fine_slots,
                    config.k_coarse_slots,
                    seed + 100 + p as u64,
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
            stages,
            head,
        }
    }

    pub fn parameter_metrics(&self) -> (usize, usize, usize, usize) {
        let embed = self.vocab_size * self.d_model + self.max_seq_len * self.d_model;
        let head = self.vocab_size * self.d_model;
        let stage_params: usize = self.stages.iter().map(|s| s.param_count()).sum();

        let total_params = embed + head + stage_params;
        let active_params = total_params;
        let dram_bytes_per_token = if self.config.use_mrm_v2 {
            self.config.k_fine_slots * 64 + self.config.k_coarse_slots * 32
        } else {
            512
        };
        let resident_l3_bytes = total_params * 4;

        (total_params, active_params, dram_bytes_per_token, resident_l3_bytes)
    }

    /// Full forward-backward pass through TESSERA with Pre-LN RMSNorm.
    pub fn forward_backward_sequence(
        &mut self,
        x_seq: &[usize],
        y_seq: &[usize],
        grads: &mut TesseraModelGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;
        let scale_attn = 1.0f32 / (d as f32).sqrt();
        let scale_embed = (d as f32).sqrt();
        let eps = 1e-5f32;

        // 1. Initial Embedding with scale
        let mut h_curr = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let tok = x_seq[t];
            let pos = t.min(self.max_seq_len - 1);
            let embed = &self.embeddings[tok * d..(tok + 1) * d];
            let pos_e = &self.pos_embeddings[pos * d..(pos + 1) * d];
            for i in 0..d {
                h_curr[t * d + i] = (embed[i] + pos_e[i]) * scale_embed;
            }
        }

        // Cache for backpropagation
        let mut stage_h_in = Vec::with_capacity(self.stages.len());
        let mut stage_hnorm1 = Vec::with_capacity(self.stages.len());
        let mut stage_rms1 = Vec::with_capacity(self.stages.len());
        let mut stage_q = Vec::with_capacity(self.stages.len());
        let mut stage_k = Vec::with_capacity(self.stages.len());
        let mut stage_v = Vec::with_capacity(self.stages.len());
        let mut stage_attn_probs = Vec::with_capacity(self.stages.len());
        let mut stage_h_mid = Vec::with_capacity(self.stages.len());
        let mut stage_hnorm2 = Vec::with_capacity(self.stages.len());
        let mut stage_rms2 = Vec::with_capacity(self.stages.len());

        // 2. Progressive Folding Stages with Pre-LN RMSNorm
        for stage in &mut self.stages {
            let h_in = h_curr.clone();
            stage_h_in.push(h_in.clone());

            // A. Pre-LN RMSNorm 1
            let mut h_norm1 = vec![0.0f32; t_len * d];
            let mut rms1 = vec![0.0f32; t_len];
            for t in 0..t_len {
                let xt = &h_in[t * d..(t + 1) * d];
                let out_t = &mut h_norm1[t * d..(t + 1) * d];
                rms1[t] = rms_norm(xt, out_t, eps);
            }
            stage_hnorm1.push(h_norm1.clone());
            stage_rms1.push(rms1);

            // Causal Self-Attention
            let wq_v = MatrixView::new(&stage.wq, d, d);
            let wk_v = MatrixView::new(&stage.wk, d, d);
            let wv_v = MatrixView::new(&stage.wv, d, d);
            let wo_v = MatrixView::new(&stage.wo, d, d);

            let mut q_mat = vec![0.0f32; t_len * d];
            let mut k_mat = vec![0.0f32; t_len * d];
            let mut v_mat = vec![0.0f32; t_len * d];

            for t in 0..t_len {
                let ht = &h_norm1[t * d..(t + 1) * d];
                matvec(&wq_v, ht, &mut q_mat[t * d..(t + 1) * d]);
                matvec(&wk_v, ht, &mut k_mat[t * d..(t + 1) * d]);
                matvec(&wv_v, ht, &mut v_mat[t * d..(t + 1) * d]);
            }

            let mut attn_probs = vec![0.0f32; t_len * t_len];
            let mut attn_out = vec![0.0f32; t_len * d];

            for i in 0..t_len {
                let qi = &q_mat[i * d..(i + 1) * d];
                let mut scores = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let kj = &k_mat[j * d..(j + 1) * d];
                    scores[j] = dot(qi, kj) * scale_attn;
                }
                let mut probs = vec![0.0f32; i + 1];
                softmax(&scores, &mut probs);

                for j in 0..=i {
                    attn_probs[i * t_len + j] = probs[j];
                    let vj = &v_mat[j * d..(j + 1) * d];
                    vec_add_scaled(&mut attn_out[i * d..(i + 1) * d], vj, probs[j]);
                }
            }

            let mut h_after_attn = h_in.clone();
            for t in 0..t_len {
                let ctx_t = &attn_out[t * d..(t + 1) * d];
                let mut proj_out = vec![0.0f32; d];
                matvec(&wo_v, ctx_t, &mut proj_out);
                let out_slice = &mut h_after_attn[t * d..(t + 1) * d];
                vec_add_scaled(out_slice, &proj_out, 1.0);
            }

            stage_q.push(q_mat);
            stage_k.push(k_mat);
            stage_v.push(v_mat);
            stage_attn_probs.push(attn_probs);
            stage_h_mid.push(h_after_attn.clone());

            // B. Pre-LN RMSNorm 2
            let mut h_norm2 = vec![0.0f32; t_len * d];
            let mut rms2 = vec![0.0f32; t_len];
            for t in 0..t_len {
                let ht = &h_after_attn[t * d..(t + 1) * d];
                let out_t = &mut h_norm2[t * d..(t + 1) * d];
                rms2[t] = rms_norm(ht, out_t, eps);
            }
            stage_hnorm2.push(h_norm2.clone());
            stage_rms2.push(rms2);

            // SwiGLU + Adapter
            let w1_v = MatrixView::new(&stage.w1, stage.d_ff, d);
            let w1u_v = MatrixView::new(&stage.w1u, stage.d_ff, d);
            let w2_v = MatrixView::new(&stage.w2, d, stage.d_ff);

            let mut h_stage_out = h_after_attn.clone();

            for t in 0..t_len {
                let ht = &h_norm2[t * d..(t + 1) * d];
                let mut gate = vec![0.0f32; stage.d_ff];
                let mut up = vec![0.0f32; stage.d_ff];
                matvec(&w1_v, ht, &mut gate);
                matvec(&w1u_v, ht, &mut up);

                let mut ff = vec![0.0f32; stage.d_ff];
                for i in 0..stage.d_ff {
                    let g = gate[i];
                    let silu = g / (1.0 + (-g).exp());
                    ff[i] = silu * up[i];
                }
                let mut ff_out = vec![0.0f32; d];
                matvec(&w2_v, &ff, &mut ff_out);

                let r = stage.adapter_rank;
                let v_view = MatrixView::new(&stage.adapter_v, r, d);
                let u_view = MatrixView::new(&stage.adapter_u, d, r);
                let mut adapt_mid = vec![0.0f32; r];
                let mut adapt_out = vec![0.0f32; d];
                matvec(&v_view, ht, &mut adapt_mid);
                matvec(&u_view, &adapt_mid, &mut adapt_out);

                let out_slice = &mut h_stage_out[t * d..(t + 1) * d];
                vec_add_scaled(out_slice, &ff_out, 1.0);
                vec_add_scaled(out_slice, &adapt_out, 1.0);
            }

            // C. MRM-v2 Active Working Memory
            if let Some(ref mut mrm) = stage.mrm {
                let mut mrm_out = vec![0.0f32; t_len * d];
                mrm.forward_sequence(&h_stage_out, t_len, &mut mrm_out);
                h_stage_out = mrm_out;
            }

            h_curr = h_stage_out;
        }

        // 3. Final RMSNorm + Output Head
        let mut h_final_norm = vec![0.0f32; t_len * d];
        let mut rms_final = vec![0.0f32; t_len];
        for t in 0..t_len {
            let ht = &h_curr[t * d..(t + 1) * d];
            let out_t = &mut h_final_norm[t * d..(t + 1) * d];
            rms_final[t] = rms_norm(ht, out_t, eps);
        }

        let head_view = MatrixView::new(&self.head, v, d);
        let mut grad_head_view = MatrixViewMut::new(&mut grads.grad_head, v, d);
        let mut delta_head = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let ht = &h_final_norm[t * d..(t + 1) * d];
            let mut logits = vec![0.0f32; v];
            let mut probs = vec![0.0f32; v];
            let mut pred_grad = vec![0.0f32; v];

            matvec(&head_view, ht, &mut logits);
            let loss = cross_entropy_loss_and_grad(&logits, y_seq[t], &mut probs, &mut pred_grad);
            total_loss += loss;

            outer_product_accumulate(&pred_grad, ht, 1.0, &mut grad_head_view);
            let d_ht = &mut delta_head[t * d..(t + 1) * d];
            matvec_transposed(&head_view, &pred_grad, d_ht);
        }

        // Backprop through Final RMSNorm
        let mut delta_upstream = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let ht = &h_curr[t * d..(t + 1) * d];
            let dh_norm = &delta_head[t * d..(t + 1) * d];
            let dh_in = &mut delta_upstream[t * d..(t + 1) * d];
            rms_norm_backward(ht, rms_final[t], dh_norm, dh_in);
        }

        // 4. Backward Pass through Progressive Stages
        for (s_idx, stage) in self.stages.iter().enumerate().rev() {
            let s_grads = &mut grads.stage_grads[s_idx];
            let h_stage_in = &stage_h_in[s_idx];
            let h_norm1 = &stage_hnorm1[s_idx];
            let rms1 = &stage_rms1[s_idx];
            let h_mid = &stage_h_mid[s_idx];
            let h_norm2 = &stage_hnorm2[s_idx];
            let rms2 = &stage_rms2[s_idx];
            let q_mat = &stage_q[s_idx];
            let k_mat = &stage_k[s_idx];
            let v_mat = &stage_v[s_idx];
            let attn_probs = &stage_attn_probs[s_idx];

            // SwiGLU + Adapter Backward
            let w1_v = MatrixView::new(&stage.w1, stage.d_ff, d);
            let w1u_v = MatrixView::new(&stage.w1u, stage.d_ff, d);
            let w2_v = MatrixView::new(&stage.w2, d, stage.d_ff);
            let mut gw1 = MatrixViewMut::new(&mut s_grads.grad_w1, stage.d_ff, d);
            let mut gw1u = MatrixViewMut::new(&mut s_grads.grad_w1u, stage.d_ff, d);
            let mut gw2 = MatrixViewMut::new(&mut s_grads.grad_w2, d, stage.d_ff);

            let mut delta_hnorm2 = vec![0.0f32; t_len * d];

            for t in 0..t_len {
                let dh = &delta_upstream[t * d..(t + 1) * d];
                let ht_norm = &h_norm2[t * d..(t + 1) * d];

                let mut gate = vec![0.0f32; stage.d_ff];
                let mut up = vec![0.0f32; stage.d_ff];
                matvec(&w1_v, ht_norm, &mut gate);
                matvec(&w1u_v, ht_norm, &mut up);

                let mut ff = vec![0.0f32; stage.d_ff];
                for i in 0..stage.d_ff {
                    let g = gate[i];
                    ff[i] = (g / (1.0 + (-g).exp())) * up[i];
                }

                outer_product_accumulate(dh, &ff, 1.0, &mut gw2);

                let mut d_ff = vec![0.0f32; stage.d_ff];
                matvec_transposed(&w2_v, dh, &mut d_ff);

                let mut d_gate = vec![0.0f32; stage.d_ff];
                let mut d_up = vec![0.0f32; stage.d_ff];
                for i in 0..stage.d_ff {
                    let g = gate[i];
                    let sig = 1.0 / (1.0 + (-g).exp());
                    let silu = g * sig;
                    let silu_grad = sig * (1.0 + g * (1.0 - sig));
                    d_up[i] = d_ff[i] * silu;
                    d_gate[i] = d_ff[i] * up[i] * silu_grad;
                }

                outer_product_accumulate(&d_gate, ht_norm, 1.0, &mut gw1);
                outer_product_accumulate(&d_up, ht_norm, 1.0, &mut gw1u);

                let d_hn = &mut delta_hnorm2[t * d..(t + 1) * d];
                matvec_transposed(&w1_v, &d_gate, d_hn);
                let mut d_hn_up = vec![0.0f32; d];
                matvec_transposed(&w1u_v, &d_up, &mut d_hn_up);
                vec_add_scaled(d_hn, &d_hn_up, 1.0);

                // Adapter backward
                let r = stage.adapter_rank;
                let v_view = MatrixView::new(&stage.adapter_v, r, d);
                let u_view = MatrixView::new(&stage.adapter_u, d, r);
                let mut gu = MatrixViewMut::new(&mut s_grads.grad_adapter_u, d, r);
                let mut gv = MatrixViewMut::new(&mut s_grads.grad_adapter_v, r, d);

                let mut adapt_mid = vec![0.0f32; r];
                matvec(&v_view, ht_norm, &mut adapt_mid);
                outer_product_accumulate(dh, &adapt_mid, 1.0, &mut gu);

                let mut d_adapt_mid = vec![0.0f32; r];
                matvec_transposed(&u_view, dh, &mut d_adapt_mid);
                outer_product_accumulate(&d_adapt_mid, ht_norm, 1.0, &mut gv);

                let mut d_hn_adapt = vec![0.0f32; d];
                matvec_transposed(&v_view, &d_adapt_mid, &mut d_hn_adapt);
                vec_add_scaled(d_hn, &d_hn_adapt, 1.0);
            }

            // Backprop through RMSNorm 2 + Residual
            let mut delta_mid = delta_upstream.clone();
            for t in 0..t_len {
                let ht_mid = &h_mid[t * d..(t + 1) * d];
                let d_hn = &delta_hnorm2[t * d..(t + 1) * d];
                let d_m = &mut delta_mid[t * d..(t + 1) * d];
                rms_norm_backward(ht_mid, rms2[t], d_hn, d_m);
            }

            // Attention Backward
            let wo_v = MatrixView::new(&stage.wo, d, d);
            let mut gwo = MatrixViewMut::new(&mut s_grads.grad_wo, d, d);
            let mut gwq = MatrixViewMut::new(&mut s_grads.grad_wq, d, d);
            let mut gwk = MatrixViewMut::new(&mut s_grads.grad_wk, d, d);
            let mut gwv = MatrixViewMut::new(&mut s_grads.grad_wv, d, d);

            let mut d_attn_out = vec![0.0f32; t_len * d];
            for t in 0..t_len {
                let dh = &delta_mid[t * d..(t + 1) * d];
                let mut ctx_t = vec![0.0f32; d];
                for j in 0..=t {
                    let vj = &v_mat[j * d..(j + 1) * d];
                    vec_add_scaled(&mut ctx_t, vj, attn_probs[t * t_len + j]);
                }
                outer_product_accumulate(dh, &ctx_t, 1.0, &mut gwo);
                matvec_transposed(&wo_v, dh, &mut d_attn_out[t * d..(t + 1) * d]);
            }

            let mut d_q = vec![0.0f32; t_len * d];
            let mut d_k = vec![0.0f32; t_len * d];
            let mut d_v = vec![0.0f32; t_len * d];

            for i in 0..t_len {
                let d_out_i = &d_attn_out[i * d..(i + 1) * d];
                let qi = &q_mat[i * d..(i + 1) * d];

                let mut d_scores = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let p_ij = attn_probs[i * t_len + j];
                    let vj = &v_mat[j * d..(j + 1) * d];
                    let dot_v = dot(d_out_i, vj);
                    d_scores[j] = p_ij * dot_v;
                    vec_add_scaled(&mut d_v[j * d..(j + 1) * d], d_out_i, p_ij);
                }

                let sum_dp: f32 = d_scores.iter().sum();
                for j in 0..=i {
                    let p_ij = attn_probs[i * t_len + j];
                    let d_score_j = (d_scores[j] - p_ij * sum_dp) * scale_attn;
                    let kj = &k_mat[j * d..(j + 1) * d];

                    vec_add_scaled(&mut d_q[i * d..(i + 1) * d], kj, d_score_j);
                    vec_add_scaled(&mut d_k[j * d..(j + 1) * d], qi, d_score_j);
                }
            }

            let mut delta_hnorm1 = vec![0.0f32; t_len * d];
            let wq_v = MatrixView::new(&stage.wq, d, d);
            let wk_v = MatrixView::new(&stage.wk, d, d);
            let wv_v = MatrixView::new(&stage.wv, d, d);

            for t in 0..t_len {
                let ht_norm = &h_norm1[t * d..(t + 1) * d];
                let dq_t = &d_q[t * d..(t + 1) * d];
                let dk_t = &d_k[t * d..(t + 1) * d];
                let dv_t = &d_v[t * d..(t + 1) * d];

                outer_product_accumulate(dq_t, ht_norm, 1.0, &mut gwq);
                outer_product_accumulate(dk_t, ht_norm, 1.0, &mut gwk);
                outer_product_accumulate(dv_t, ht_norm, 1.0, &mut gwv);

                let d_hn = &mut delta_hnorm1[t * d..(t + 1) * d];
                let mut tmp = vec![0.0f32; d];
                matvec_transposed(&wq_v, dq_t, &mut tmp);
                vec_add_scaled(d_hn, &tmp, 1.0);
                matvec_transposed(&wk_v, dk_t, &mut tmp);
                vec_add_scaled(d_hn, &tmp, 1.0);
                matvec_transposed(&wv_v, dv_t, &mut tmp);
                vec_add_scaled(d_hn, &tmp, 1.0);
            }

            // Backprop through RMSNorm 1 + Residual
            let mut delta_in = delta_mid.clone();
            for t in 0..t_len {
                let ht_in = &h_stage_in[t * d..(t + 1) * d];
                let d_hn = &delta_hnorm1[t * d..(t + 1) * d];
                let d_in_t = &mut delta_in[t * d..(t + 1) * d];
                rms_norm_backward(ht_in, rms1[t], d_hn, d_in_t);
            }

            delta_upstream = delta_in;
        }

        // 5. Backprop to Embeddings
        for t in 0..t_len {
            let tok = x_seq[t];
            let pos = t.min(self.max_seq_len - 1);
            let dh = &delta_upstream[t * d..(t + 1) * d];
            let emb_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            let pos_slice = &mut grads.grad_pos_embed[pos * d..(pos + 1) * d];
            for i in 0..d {
                let grad_scaled = dh[i] * scale_embed;
                emb_slice[i] += grad_scaled;
                pos_slice[i] += grad_scaled;
            }
        }

        total_loss
    }
}
