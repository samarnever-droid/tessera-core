//! TESSERA-Q: Quality-First Architecture with 4-Head Causal Attention + RoPE + Affine RMSNorm + Tied Embeddings + MRM-v2.
//! Designed to beat DeepMind Griffin on BPC while retaining 100% 8K long-context recall.

use crate::mrm_v2::{MrmV2Grads, MultiResMemoryV2};
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::{cross_entropy_loss_and_grad, softmax};
use axiom_core::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Element-wise Affine RMSNorm forward: y = (x / RMS(x)) * gamma
#[inline]
pub fn rms_norm_affine(x: &[f32], gamma: &[f32], out: &mut [f32], eps: f32) -> f32 {
    let d = x.len() as f32;
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms = (sum_sq / d + eps).sqrt();
    let inv_rms = 1.0f32 / rms;
    for ((o, &v), &g) in out.iter_mut().zip(x.iter()).zip(gamma.iter()) {
        *o = v * inv_rms * g;
    }
    rms
}

/// Element-wise Affine RMSNorm backward
#[inline]
pub fn rms_norm_affine_backward(
    x: &[f32],
    gamma: &[f32],
    rms: f32,
    grad_out: &[f32],
    grad_in: &mut [f32],
    grad_gamma: &mut [f32],
) {
    let d = x.len() as f32;
    let inv_rms = 1.0f32 / rms;
    let mut dot_go_gamma_x = 0.0f32;

    for (((&go, &g), &xi), gg) in grad_out.iter().zip(gamma.iter()).zip(x.iter()).zip(grad_gamma.iter_mut()) {
        *gg += go * xi * inv_rms;
        dot_go_gamma_x += go * g * xi;
    }

    let scale_dot = dot_go_gamma_x / (d * (rms * rms * rms));
    for (((gi, &go), &g), &xi) in grad_in.iter_mut().zip(grad_out.iter()).zip(gamma.iter()).zip(x.iter()) {
        *gi += go * g * inv_rms - xi * scale_dot;
    }
}

/// Apply Rotary Position Embedding (RoPE) forward
#[inline]
pub fn apply_rope(vec: &mut [f32], pos: usize, d_k: usize) {
    let half = d_k / 2;
    for i in 0..half {
        let theta = 1.0f32 / (10000.0f32.powf((2 * i) as f32 / d_k as f32));
        let angle = pos as f32 * theta;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let v0 = vec[2 * i];
        let v1 = vec[2 * i + 1];
        vec[2 * i] = v0 * cos_a - v1 * sin_a;
        vec[2 * i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

/// Apply Rotary Position Embedding (RoPE) backward
#[inline]
pub fn apply_rope_backward(d_out: &[f32], d_in: &mut [f32], pos: usize, d_k: usize) {
    let half = d_k / 2;
    for i in 0..half {
        let theta = 1.0f32 / (10000.0f32.powf((2 * i) as f32 / d_k as f32));
        let angle = pos as f32 * theta;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let g0 = d_out[2 * i];
        let g1 = d_out[2 * i + 1];
        d_in[2 * i] = g0 * cos_a + g1 * sin_a;
        d_in[2 * i + 1] = -g0 * sin_a + g1 * cos_a;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TesseraConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,       // H = 4 attention heads
    pub n_stages: usize,      // P = 3 progressive hierarchy stages
    pub adapter_rank: usize,  // r = 8 per-stage low-rank modulation
    pub use_mrm_v2: bool,     // Ablation flag (Arm B vs Arm C)
    pub k_fine_slots: usize,  // 128 fine slots
    pub k_coarse_slots: usize,// 16 coarse slots
}

impl TesseraConfig {
    pub fn nano_default() -> Self {
        Self {
            d_model: 128,
            d_ff: 768,        // 6x expansion for peak expressivity
            n_heads: 4,
            n_stages: 3,
            adapter_rank: 8,
            use_mrm_v2: true,
            k_fine_slots: 128,
            k_coarse_slots: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TesseraStageGrads {
    pub grad_norm1_gamma: Vec<f32>,
    pub grad_wq: Vec<f32>,
    pub grad_wk: Vec<f32>,
    pub grad_wv: Vec<f32>,
    pub grad_wo: Vec<f32>,
    pub grad_norm2_gamma: Vec<f32>,
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
            grad_norm1_gamma: vec![0.0f32; d],
            grad_wq: vec![0.0f32; d * d],
            grad_wk: vec![0.0f32; d * d],
            grad_wv: vec![0.0f32; d * d],
            grad_wo: vec![0.0f32; d * d],
            grad_norm2_gamma: vec![0.0f32; d],
            grad_w1: vec![0.0f32; d_ff * d],
            grad_w1u: vec![0.0f32; d_ff * d],
            grad_w2: vec![0.0f32; d * d_ff],
            grad_adapter_u: vec![0.0f32; d * r],
            grad_adapter_v: vec![0.0f32; r * d],
            mrm_grads: if use_mrm { Some(MrmV2Grads::new(d)) } else { None },
        }
    }

    pub fn zero(&mut self) {
        self.grad_norm1_gamma.fill(0.0f32);
        self.grad_wq.fill(0.0f32);
        self.grad_wk.fill(0.0f32);
        self.grad_wv.fill(0.0f32);
        self.grad_wo.fill(0.0f32);
        self.grad_norm2_gamma.fill(0.0f32);
        self.grad_w1.fill(0.0f32);
        self.grad_w1u.fill(0.0f32);
        self.grad_w2.fill(0.0f32);
        self.grad_adapter_u.fill(0.0f32);
        self.grad_adapter_v.fill(0.0f32);
        if let Some(ref mut mg) = self.mrm_grads { mg.zero(); }
    }

    pub fn add(&mut self, other: &TesseraStageGrads) {
        for (a, &b) in self.grad_norm1_gamma.iter_mut().zip(other.grad_norm1_gamma.iter()) { *a += b; }
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_wk.iter_mut().zip(other.grad_wk.iter()) { *a += b; }
        for (a, &b) in self.grad_wv.iter_mut().zip(other.grad_wv.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        for (a, &b) in self.grad_norm2_gamma.iter_mut().zip(other.grad_norm2_gamma.iter()) { *a += b; }
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

/// A Progressive Hierarchy Stage with 4-Head Attention + RoPE + Affine RMSNorm + SwiGLU + MRM-v2.
#[derive(Debug, Clone)]
pub struct TesseraStage {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub adapter_rank: usize,
    pub norm1_gamma: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub norm2_gamma: Vec<f32>,
    pub w1: Vec<f32>,
    pub w1u: Vec<f32>,
    pub w2: Vec<f32>,
    pub adapter_u: Vec<f32>,
    pub adapter_v: Vec<f32>,
    pub mrm: Option<MultiResMemoryV2>,
}

impl TesseraStage {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
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

        let norm1_gamma = vec![1.0f32; d_model];
        let wq = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wk = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wv = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wo = (0..d_model * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let norm2_gamma = vec![1.0f32; d_model];
        let w1 = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w1u = (0..d_ff * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w2 = (0..d_model * d_ff).map(|_| rng.gen_range(-scale_ff..scale_ff)).collect();

        let adapter_u = (0..d_model * adapter_rank).map(|_| rng.gen_range(-scale_r..scale_r)).collect();
        let adapter_v = vec![0.0f32; adapter_rank * d_model];

        let mrm = if use_mrm {
            Some(MultiResMemoryV2::new(d_model, k_fine, k_coarse, seed + 10))
        } else {
            None
        };

        Self {
            d_model,
            d_ff,
            n_heads,
            adapter_rank,
            norm1_gamma,
            wq,
            wk,
            wv,
            wo,
            norm2_gamma,
            w1,
            w1u,
            w2,
            adapter_u,
            adapter_v,
            mrm,
        }
    }

    pub fn param_count(&self) -> usize {
        let norms = 2 * self.d_model;
        let attn = 4 * (self.d_model * self.d_model);
        let dense = 2 * (self.d_ff * self.d_model) + (self.d_model * self.d_ff);
        let adapter = 2 * (self.d_model * self.adapter_rank);
        let mrm_p = self.mrm.as_ref().map(|m| m.param_count()).unwrap_or(0);
        norms + attn + dense + adapter + mrm_p
    }
}

#[derive(Debug, Clone)]
pub struct TesseraModelGrads {
    pub grad_embed: Vec<f32>,
    pub stage_grads: Vec<TesseraStageGrads>,
    pub grad_final_norm_gamma: Vec<f32>,
}

impl TesseraModelGrads {
    pub fn new(vocab_size: usize, d_model: usize, _max_seq: usize, stages: &[TesseraStage]) -> Self {
        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            stage_grads: stages.iter().map(|s| {
                TesseraStageGrads::new(d_model, s.d_ff, s.adapter_rank, s.mrm.is_some())
            }).collect(),
            grad_final_norm_gamma: vec![0.0f32; d_model],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embed.fill(0.0f32);
        for sg in &mut self.stage_grads { sg.zero(); }
        self.grad_final_norm_gamma.fill(0.0f32);
    }

    pub fn add(&mut self, other: &TesseraModelGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (sg, osg) in self.stage_grads.iter_mut().zip(other.stage_grads.iter()) { sg.add(osg); }
        for (a, &b) in self.grad_final_norm_gamma.iter_mut().zip(other.grad_final_norm_gamma.iter()) { *a += b; }
    }
}

/// Full TESSERA Architecture with 4-Head Attention, RoPE, Affine RMSNorm, and Tied Embeddings.
#[derive(Debug, Clone)]
pub struct TesseraModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub config: TesseraConfig,
    pub embeddings: Vec<f32>,
    pub stages: Vec<TesseraStage>,
    pub final_norm_gamma: Vec<f32>,
}

impl TesseraModel {
    pub fn new(vocab_size: usize, max_seq_len: usize, config: TesseraConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / config.d_model as f32).sqrt();

        let embeddings = (0..vocab_size * config.d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let final_norm_gamma = vec![1.0f32; config.d_model];

        let stages = (0..config.n_stages)
            .map(|p| {
                TesseraStage::new(
                    config.d_model,
                    config.d_ff,
                    config.n_heads,
                    config.adapter_rank,
                    config.use_mrm_v2 && p == config.n_stages - 1,
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
            stages,
            final_norm_gamma,
        }
    }

    pub fn parameter_metrics(&self) -> (usize, usize, usize, usize) {
        let embed = self.vocab_size * self.d_model;
        let final_norm = self.d_model;
        let stage_params: usize = self.stages.iter().map(|s| s.param_count()).sum();

        let total_params = embed + final_norm + stage_params;
        let active_params = total_params;
        let dram_bytes_per_token = if self.config.use_mrm_v2 {
            self.config.k_fine_slots * 64 + self.config.k_coarse_slots * 32
        } else {
            512
        };
        let resident_l3_bytes = total_params * 4;

        (total_params, active_params, dram_bytes_per_token, resident_l3_bytes)
    }

    /// Full forward-backward pass through TESSERA with Affine RMSNorm & RoPE.
    pub fn forward_backward_sequence(
        &mut self,
        x_seq: &[usize],
        y_seq: &[usize],
        grads: &mut TesseraModelGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;
        let n_heads = self.config.n_heads;
        let d_k = d / n_heads; // 32
        let scale_attn = 1.0f32 / (d_k as f32).sqrt();
        let scale_embed = (d as f32).sqrt();
        let eps = 1e-5f32;

        let d_ff = self.stages[0].d_ff;
        let r_adapt = self.stages[0].adapter_rank;

        // Reusable scratch buffers allocated ONCE per sequence call
        let mut buf_gate = vec![0.0f32; d_ff];
        let mut buf_up = vec![0.0f32; d_ff];
        let mut buf_ff = vec![0.0f32; d_ff];
        let mut buf_ff_out = vec![0.0f32; d];
        let mut buf_adapt_mid = vec![0.0f32; r_adapt];
        let mut buf_adapt_out = vec![0.0f32; d];
        let mut buf_proj_out = vec![0.0f32; d];
        let mut buf_d_ff = vec![0.0f32; d_ff];
        let mut buf_d_gate = vec![0.0f32; d_ff];
        let mut buf_d_up = vec![0.0f32; d_ff];
        let mut buf_d_adapt_mid = vec![0.0f32; r_adapt];
        let mut buf_d_hn_up = vec![0.0f32; d];
        let mut buf_d_hn_adapt = vec![0.0f32; d];
        let mut buf_tmp = vec![0.0f32; d];
        let mut buf_scores = vec![0.0f32; t_len];
        let mut buf_probs = vec![0.0f32; t_len];
        let mut buf_d_scores = vec![0.0f32; t_len];
        let mut buf_logits = vec![0.0f32; v];
        let mut buf_pred_probs = vec![0.0f32; v];
        let mut buf_pred_grad = vec![0.0f32; v];

        // 1. Initial Embedding with scale
        let mut h_curr = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let tok = x_seq[t];
            let embed = &self.embeddings[tok * d..(tok + 1) * d];
            for i in 0..d {
                h_curr[t * d + i] = embed[i] * scale_embed;
            }
        }

        // Cache for backpropagation
        let mut stage_h_in = Vec::with_capacity(self.stages.len());
        let mut stage_hnorm1 = Vec::with_capacity(self.stages.len());
        let mut stage_rms1 = Vec::with_capacity(self.stages.len());
        let mut stage_q_rope = Vec::with_capacity(self.stages.len());
        let mut stage_k_rope = Vec::with_capacity(self.stages.len());
        let mut stage_v = Vec::with_capacity(self.stages.len());
        let mut stage_attn_probs = Vec::with_capacity(self.stages.len());
        let mut stage_h_mid = Vec::with_capacity(self.stages.len());
        let mut stage_hnorm2 = Vec::with_capacity(self.stages.len());
        let mut stage_rms2 = Vec::with_capacity(self.stages.len());

        // 2. Progressive Folding Stages with Affine RMSNorm + RoPE
        for stage in &mut self.stages {
            let h_in = h_curr.clone();
            stage_h_in.push(h_in.clone());

            // A. Pre-LN Affine RMSNorm 1
            let mut h_norm1 = vec![0.0f32; t_len * d];
            let mut rms1 = vec![0.0f32; t_len];
            for t in 0..t_len {
                let xt = &h_in[t * d..(t + 1) * d];
                let out_t = &mut h_norm1[t * d..(t + 1) * d];
                rms1[t] = rms_norm_affine(xt, &stage.norm1_gamma, out_t, eps);
            }
            stage_hnorm1.push(h_norm1.clone());
            stage_rms1.push(rms1);

            // 4-Head Causal Self-Attention
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

                // Apply Rotary Position Embeddings (RoPE) to Q and K per head
                for h in 0..n_heads {
                    let h_offset = h * d_k;
                    apply_rope(&mut q_mat[t * d + h_offset..t * d + h_offset + d_k], t, d_k);
                    apply_rope(&mut k_mat[t * d + h_offset..t * d + h_offset + d_k], t, d_k);
                }
            }

            let mut attn_probs = vec![0.0f32; n_heads * t_len * t_len];
            let mut attn_out = vec![0.0f32; t_len * d];

            for h in 0..n_heads {
                let h_offset = h * d_k;
                for i in 0..t_len {
                    let qi = &q_mat[i * d + h_offset..i * d + h_offset + d_k];
                    let cur_scores = &mut buf_scores[..=i];
                    let cur_probs = &mut buf_probs[..=i];

                    for j in 0..=i {
                        let kj = &k_mat[j * d + h_offset..j * d + h_offset + d_k];
                        cur_scores[j] = dot(qi, kj) * scale_attn;
                    }
                    softmax(cur_scores, cur_probs);

                    for j in 0..=i {
                        attn_probs[h * (t_len * t_len) + i * t_len + j] = cur_probs[j];
                        let vj = &v_mat[j * d + h_offset..j * d + h_offset + d_k];
                        vec_add_scaled(&mut attn_out[i * d + h_offset..i * d + h_offset + d_k], vj, cur_probs[j]);
                    }
                }
            }

            let mut h_after_attn = h_in.clone();
            for t in 0..t_len {
                let ctx_t = &attn_out[t * d..(t + 1) * d];
                matvec(&wo_v, ctx_t, &mut buf_proj_out);
                let out_slice = &mut h_after_attn[t * d..(t + 1) * d];
                vec_add_scaled(out_slice, &buf_proj_out, 1.0);
            }

            stage_q_rope.push(q_mat);
            stage_k_rope.push(k_mat);
            stage_v.push(v_mat);
            stage_attn_probs.push(attn_probs);
            stage_h_mid.push(h_after_attn.clone());

            // B. Pre-LN Affine RMSNorm 2
            let mut h_norm2 = vec![0.0f32; t_len * d];
            let mut rms2 = vec![0.0f32; t_len];
            for t in 0..t_len {
                let ht = &h_after_attn[t * d..(t + 1) * d];
                let out_t = &mut h_norm2[t * d..(t + 1) * d];
                rms2[t] = rms_norm_affine(ht, &stage.norm2_gamma, out_t, eps);
            }
            stage_hnorm2.push(h_norm2.clone());
            stage_rms2.push(rms2);

            // SwiGLU + Adapter
            let w1_v = MatrixView::new(&stage.w1, stage.d_ff, d);
            let w1u_v = MatrixView::new(&stage.w1u, stage.d_ff, d);
            let w2_v = MatrixView::new(&stage.w2, d, stage.d_ff);
            let r = stage.adapter_rank;
            let v_view = MatrixView::new(&stage.adapter_v, r, d);
            let u_view = MatrixView::new(&stage.adapter_u, d, r);

            let mut h_stage_out = h_after_attn.clone();

            for t in 0..t_len {
                let ht = &h_norm2[t * d..(t + 1) * d];
                matvec(&w1_v, ht, &mut buf_gate);
                matvec(&w1u_v, ht, &mut buf_up);

                for i in 0..stage.d_ff {
                    let g = buf_gate[i];
                    let silu = g / (1.0 + (-g).exp());
                    buf_ff[i] = silu * buf_up[i];
                }
                matvec(&w2_v, &buf_ff, &mut buf_ff_out);

                matvec(&v_view, ht, &mut buf_adapt_mid);
                matvec(&u_view, &buf_adapt_mid, &mut buf_adapt_out);

                let out_slice = &mut h_stage_out[t * d..(t + 1) * d];
                vec_add_scaled(out_slice, &buf_ff_out, 1.0);
                vec_add_scaled(out_slice, &buf_adapt_out, 1.0);
            }

            // C. MRM-v2 Active Working Memory
            if let Some(ref mut mrm) = stage.mrm {
                let mut mrm_out = vec![0.0f32; t_len * d];
                mrm.forward_sequence(&h_stage_out, t_len, &mut mrm_out);
                h_stage_out = mrm_out;
            }

            h_curr = h_stage_out;
        }

        // 3. Final Affine RMSNorm + Tied Output Logits
        let mut h_final_norm = vec![0.0f32; t_len * d];
        let mut rms_final = vec![0.0f32; t_len];
        for t in 0..t_len {
            let ht = &h_curr[t * d..(t + 1) * d];
            let out_t = &mut h_final_norm[t * d..(t + 1) * d];
            rms_final[t] = rms_norm_affine(ht, &self.final_norm_gamma, out_t, eps);
        }

        let embed_view = MatrixView::new(&self.embeddings, v, d);
        let mut grad_embed_view = MatrixViewMut::new(&mut grads.grad_embed, v, d);
        let mut delta_head = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let ht = &h_final_norm[t * d..(t + 1) * d];

            matvec(&embed_view, ht, &mut buf_logits);
            let loss = cross_entropy_loss_and_grad(&buf_logits, y_seq[t], &mut buf_pred_probs, &mut buf_pred_grad);
            total_loss += loss;

            outer_product_accumulate(&buf_pred_grad, ht, 1.0, &mut grad_embed_view);
            let d_ht = &mut delta_head[t * d..(t + 1) * d];
            matvec_transposed(&embed_view, &buf_pred_grad, d_ht);
        }

        // Backprop through Final Affine RMSNorm
        let mut delta_upstream = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let ht = &h_curr[t * d..(t + 1) * d];
            let dh_norm = &delta_head[t * d..(t + 1) * d];
            let dh_in = &mut delta_upstream[t * d..(t + 1) * d];
            rms_norm_affine_backward(
                ht,
                &self.final_norm_gamma,
                rms_final[t],
                dh_norm,
                dh_in,
                &mut grads.grad_final_norm_gamma,
            );
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
            let q_mat = &stage_q_rope[s_idx];
            let k_mat = &stage_k_rope[s_idx];
            let v_mat = &stage_v[s_idx];
            let attn_probs = &stage_attn_probs[s_idx];

            // SwiGLU + Adapter Backward
            let w1_v = MatrixView::new(&stage.w1, stage.d_ff, d);
            let w1u_v = MatrixView::new(&stage.w1u, stage.d_ff, d);
            let w2_v = MatrixView::new(&stage.w2, d, stage.d_ff);
            let mut gw1 = MatrixViewMut::new(&mut s_grads.grad_w1, stage.d_ff, d);
            let mut gw1u = MatrixViewMut::new(&mut s_grads.grad_w1u, stage.d_ff, d);
            let mut gw2 = MatrixViewMut::new(&mut s_grads.grad_w2, d, stage.d_ff);
            let r = stage.adapter_rank;
            let v_view = MatrixView::new(&stage.adapter_v, r, d);
            let u_view = MatrixView::new(&stage.adapter_u, d, r);
            let mut gu = MatrixViewMut::new(&mut s_grads.grad_adapter_u, d, r);
            let mut gv = MatrixViewMut::new(&mut s_grads.grad_adapter_v, r, d);

            let mut delta_hnorm2 = vec![0.0f32; t_len * d];

            for t in 0..t_len {
                let dh = &delta_upstream[t * d..(t + 1) * d];
                let ht_norm = &h_norm2[t * d..(t + 1) * d];

                matvec(&w1_v, ht_norm, &mut buf_gate);
                matvec(&w1u_v, ht_norm, &mut buf_up);

                for i in 0..stage.d_ff {
                    let g = buf_gate[i];
                    buf_ff[i] = (g / (1.0 + (-g).exp())) * buf_up[i];
                }

                outer_product_accumulate(dh, &buf_ff, 1.0, &mut gw2);

                matvec_transposed(&w2_v, dh, &mut buf_d_ff);

                for i in 0..stage.d_ff {
                    let g = buf_gate[i];
                    let sig = 1.0 / (1.0 + (-g).exp());
                    let silu = g * sig;
                    let silu_grad = sig * (1.0 + g * (1.0 - sig));
                    buf_d_up[i] = buf_d_ff[i] * silu;
                    buf_d_gate[i] = buf_d_ff[i] * buf_up[i] * silu_grad;
                }

                outer_product_accumulate(&buf_d_gate, ht_norm, 1.0, &mut gw1);
                outer_product_accumulate(&buf_d_up, ht_norm, 1.0, &mut gw1u);

                let d_hn = &mut delta_hnorm2[t * d..(t + 1) * d];
                matvec_transposed(&w1_v, &buf_d_gate, d_hn);
                matvec_transposed(&w1u_v, &buf_d_up, &mut buf_d_hn_up);
                vec_add_scaled(d_hn, &buf_d_hn_up, 1.0);

                // Adapter backward
                matvec(&v_view, ht_norm, &mut buf_adapt_mid);
                outer_product_accumulate(dh, &buf_adapt_mid, 1.0, &mut gu);

                matvec_transposed(&u_view, dh, &mut buf_d_adapt_mid);
                outer_product_accumulate(&buf_d_adapt_mid, ht_norm, 1.0, &mut gv);

                matvec_transposed(&v_view, &buf_d_adapt_mid, &mut buf_d_hn_adapt);
                vec_add_scaled(d_hn, &buf_d_hn_adapt, 1.0);
            }

            // Backprop through Affine RMSNorm 2 + Residual
            let mut delta_mid = delta_upstream.clone();
            for t in 0..t_len {
                let ht_mid = &h_mid[t * d..(t + 1) * d];
                let d_hn = &delta_hnorm2[t * d..(t + 1) * d];
                let d_m = &mut delta_mid[t * d..(t + 1) * d];
                rms_norm_affine_backward(
                    ht_mid,
                    &stage.norm2_gamma,
                    rms2[t],
                    d_hn,
                    d_m,
                    &mut s_grads.grad_norm2_gamma,
                );
            }

            // 4-Head Attention Backward
            let wo_v = MatrixView::new(&stage.wo, d, d);
            let mut gwo = MatrixViewMut::new(&mut s_grads.grad_wo, d, d);
            let mut gwq = MatrixViewMut::new(&mut s_grads.grad_wq, d, d);
            let mut gwk = MatrixViewMut::new(&mut s_grads.grad_wk, d, d);
            let mut gwv = MatrixViewMut::new(&mut s_grads.grad_wv, d, d);

            let mut d_attn_out = vec![0.0f32; t_len * d];
            for t in 0..t_len {
                let dh = &delta_mid[t * d..(t + 1) * d];
                let mut ctx_t = vec![0.0f32; d];
                for h in 0..n_heads {
                    let h_offset = h * d_k;
                    for j in 0..=t {
                        let vj = &v_mat[j * d + h_offset..j * d + h_offset + d_k];
                        let p = attn_probs[h * (t_len * t_len) + t * t_len + j];
                        vec_add_scaled(&mut ctx_t[h_offset..h_offset + d_k], vj, p);
                    }
                }
                outer_product_accumulate(dh, &ctx_t, 1.0, &mut gwo);
                matvec_transposed(&wo_v, dh, &mut d_attn_out[t * d..(t + 1) * d]);
            }

            let mut dq_rope = vec![0.0f32; t_len * d];
            let mut dk_rope = vec![0.0f32; t_len * d];
            let mut dv_mat = vec![0.0f32; t_len * d];

            for h in 0..n_heads {
                let h_offset = h * d_k;
                for i in 0..t_len {
                    let d_out_i = &d_attn_out[i * d + h_offset..i * d + h_offset + d_k];
                    let qi = &q_mat[i * d + h_offset..i * d + h_offset + d_k];

                    let cur_d_scores = &mut buf_d_scores[..=i];
                    for j in 0..=i {
                        let p_ij = attn_probs[h * (t_len * t_len) + i * t_len + j];
                        let vj = &v_mat[j * d + h_offset..j * d + h_offset + d_k];
                        let dot_v = dot(d_out_i, vj);
                        cur_d_scores[j] = p_ij * dot_v;
                        vec_add_scaled(&mut dv_mat[j * d + h_offset..j * d + h_offset + d_k], d_out_i, p_ij);
                    }

                    let sum_dp: f32 = cur_d_scores.iter().sum();
                    for j in 0..=i {
                        let p_ij = attn_probs[h * (t_len * t_len) + i * t_len + j];
                        let d_score_j = (cur_d_scores[j] - p_ij * sum_dp) * scale_attn;
                        let kj = &k_mat[j * d + h_offset..j * d + h_offset + d_k];

                        vec_add_scaled(&mut dq_rope[i * d + h_offset..i * d + h_offset + d_k], kj, d_score_j);
                        vec_add_scaled(&mut dk_rope[j * d + h_offset..j * d + h_offset + d_k], qi, d_score_j);
                    }
                }
            }

            // Backprop through RoPE rotation for Q and K
            let mut dq_mat = vec![0.0f32; t_len * d];
            let mut dk_mat = vec![0.0f32; t_len * d];
            for t in 0..t_len {
                for h in 0..n_heads {
                    let h_offset = h * d_k;
                    apply_rope_backward(
                        &dq_rope[t * d + h_offset..t * d + h_offset + d_k],
                        &mut dq_mat[t * d + h_offset..t * d + h_offset + d_k],
                        t,
                        d_k,
                    );
                    apply_rope_backward(
                        &dk_rope[t * d + h_offset..t * d + h_offset + d_k],
                        &mut dk_mat[t * d + h_offset..t * d + h_offset + d_k],
                        t,
                        d_k,
                    );
                }
            }

            let mut delta_hnorm1 = vec![0.0f32; t_len * d];
            let wq_v = MatrixView::new(&stage.wq, d, d);
            let wk_v = MatrixView::new(&stage.wk, d, d);
            let wv_v = MatrixView::new(&stage.wv, d, d);

            for t in 0..t_len {
                let ht_norm = &h_norm1[t * d..(t + 1) * d];
                let dq_t = &dq_mat[t * d..(t + 1) * d];
                let dk_t = &dk_mat[t * d..(t + 1) * d];
                let dv_t = &dv_mat[t * d..(t + 1) * d];

                outer_product_accumulate(dq_t, ht_norm, 1.0, &mut gwq);
                outer_product_accumulate(dk_t, ht_norm, 1.0, &mut gwk);
                outer_product_accumulate(dv_t, ht_norm, 1.0, &mut gwv);

                let d_hn = &mut delta_hnorm1[t * d..(t + 1) * d];
                matvec_transposed(&wq_v, dq_t, &mut buf_tmp);
                vec_add_scaled(d_hn, &buf_tmp, 1.0);
                matvec_transposed(&wk_v, dk_t, &mut buf_tmp);
                vec_add_scaled(d_hn, &buf_tmp, 1.0);
                matvec_transposed(&wv_v, dv_t, &mut buf_tmp);
                vec_add_scaled(d_hn, &buf_tmp, 1.0);
            }

            // Backprop through Affine RMSNorm 1 + Residual
            let mut delta_in = delta_mid.clone();
            for t in 0..t_len {
                let ht_in = &h_stage_in[t * d..(t + 1) * d];
                let d_hn = &delta_hnorm1[t * d..(t + 1) * d];
                let d_in_t = &mut delta_in[t * d..(t + 1) * d];
                rms_norm_affine_backward(
                    ht_in,
                    &stage.norm1_gamma,
                    rms1[t],
                    d_hn,
                    d_in_t,
                    &mut s_grads.grad_norm1_gamma,
                );
            }

            delta_upstream = delta_in;
        }

        // 5. Backprop to Tied Input Embeddings
        for t in 0..t_len {
            let tok = x_seq[t];
            let dh = &delta_upstream[t * d..(t + 1) * d];
            let emb_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            for i in 0..d {
                emb_slice[i] += dh[i] * scale_embed;
            }
        }

        total_loss
    }
}
