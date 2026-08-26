//! Standard 4-Layer Causal Transformer Baseline with Full Serial Depth Backpropagation.

use axiom_core::activations::gelu;
use axiom_core::matvec::{matvec, matvec_transposed, outer_product_accumulate};
use axiom_core::softmax::{cross_entropy_loss_and_grad, softmax};
use axiom_core::tensor::{dot, MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Gradients for a single Transformer block.
#[derive(Debug, Clone)]
pub struct TransformerBlockGrads {
    pub grad_wq: Vec<f32>, // (d x d)
    pub grad_wk: Vec<f32>, // (d x d)
    pub grad_wv: Vec<f32>, // (d x d)
    pub grad_wo: Vec<f32>, // (d x d)
    pub grad_w1: Vec<f32>, // (d_ffn x d)
    pub grad_w2: Vec<f32>, // (d x d_ffn)
}

impl TransformerBlockGrads {
    pub fn new(d: usize, d_ffn: usize) -> Self {
        Self {
            grad_wq: vec![0.0f32; d * d],
            grad_wk: vec![0.0f32; d * d],
            grad_wv: vec![0.0f32; d * d],
            grad_wo: vec![0.0f32; d * d],
            grad_w1: vec![0.0f32; d_ffn * d],
            grad_w2: vec![0.0f32; d * d_ffn],
        }
    }

    pub fn zero(&mut self) {
        self.grad_wq.fill(0.0f32);
        self.grad_wk.fill(0.0f32);
        self.grad_wv.fill(0.0f32);
        self.grad_wo.fill(0.0f32);
        self.grad_w1.fill(0.0f32);
        self.grad_w2.fill(0.0f32);
    }

    pub fn add(&mut self, other: &TransformerBlockGrads) {
        for (a, &b) in self.grad_wq.iter_mut().zip(other.grad_wq.iter()) { *a += b; }
        for (a, &b) in self.grad_wk.iter_mut().zip(other.grad_wk.iter()) { *a += b; }
        for (a, &b) in self.grad_wv.iter_mut().zip(other.grad_wv.iter()) { *a += b; }
        for (a, &b) in self.grad_wo.iter_mut().zip(other.grad_wo.iter()) { *a += b; }
        for (a, &b) in self.grad_w1.iter_mut().zip(other.grad_w1.iter()) { *a += b; }
        for (a, &b) in self.grad_w2.iter_mut().zip(other.grad_w2.iter()) { *a += b; }
    }
}

/// Gradients for full Transformer Model.
#[derive(Debug, Clone)]
pub struct TransformerGrads {
    pub grad_embed: Vec<f32>,
    pub grad_pos_embed: Vec<f32>,
    pub block_grads: Vec<TransformerBlockGrads>,
    pub grad_head: Vec<f32>,
}

impl TransformerGrads {
    pub fn new(vocab_size: usize, d_model: usize, max_seq_len: usize, n_layers: usize, d_ffn: usize) -> Self {
        let mut block_grads = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            block_grads.push(TransformerBlockGrads::new(d_model, d_ffn));
        }

        Self {
            grad_embed: vec![0.0f32; vocab_size * d_model],
            grad_pos_embed: vec![0.0f32; max_seq_len * d_model],
            block_grads,
            grad_head: vec![0.0f32; vocab_size * d_model],
        }
    }

    pub fn zero(&mut self) {
        self.grad_embed.fill(0.0f32);
        self.grad_pos_embed.fill(0.0f32);
        for bg in &mut self.block_grads {
            bg.zero();
        }
        self.grad_head.fill(0.0f32);
    }

    pub fn add(&mut self, other: &TransformerGrads) {
        for (a, &b) in self.grad_embed.iter_mut().zip(other.grad_embed.iter()) { *a += b; }
        for (a, &b) in self.grad_pos_embed.iter_mut().zip(other.grad_pos_embed.iter()) { *a += b; }
        for (a_bg, b_bg) in self.block_grads.iter_mut().zip(other.block_grads.iter()) { a_bg.add(b_bg); }
        for (a, &b) in self.grad_head.iter_mut().zip(other.grad_head.iter()) { *a += b; }
    }
}

/// Single Transformer Layer Block (Self-Attention + FFN + Residuals).
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub d_model: usize,
    pub d_ffn: usize,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
}

impl TransformerBlock {
    pub fn new(d: usize, d_ffn: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d as f32).sqrt();
        let scale_ffn = (1.0f32 / d_ffn as f32).sqrt();

        let wq = (0..d * d).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wk = (0..d * d).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wv = (0..d * d).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let wo = (0..d * d).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w1 = (0..d_ffn * d).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let w2 = (0..d * d_ffn).map(|_| rng.gen_range(-scale_ffn..scale_ffn)).collect();

        Self { d_model: d, d_ffn, wq, wk, wv, wo, w1, w2 }
    }
}

/// Matched Standard 4-Layer Causal Transformer.
#[derive(Debug, Clone)]
pub struct TransformerModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub d_ffn: usize,
    pub max_seq_len: usize,
    pub embeddings: Vec<f32>,
    pub pos_embeddings: Vec<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub head: Vec<f32>,
}

impl TransformerModel {
    pub fn new(vocab_size: usize, d_model: usize, n_layers: usize, d_ffn: usize, max_seq_len: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_d = (1.0f32 / d_model as f32).sqrt();

        let embeddings = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();
        let pos_embeddings = (0..max_seq_len * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        let mut blocks = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            blocks.push(TransformerBlock::new(d_model, d_ffn, seed + 10 + (l as u64) * 100));
        }

        let head = (0..vocab_size * d_model).map(|_| rng.gen_range(-scale_d..scale_d)).collect();

        Self {
            vocab_size,
            d_model,
            n_layers,
            d_ffn,
            max_seq_len,
            embeddings,
            pos_embeddings,
            blocks,
            head,
        }
    }

    /// Full serial forward + serial backward pass over a sequence item.
    pub fn forward_backward_sequence(
        &self,
        x_seq: &[usize],
        y_seq: &[usize],
        grads: &mut TransformerGrads,
    ) -> f32 {
        let t_len = x_seq.len();
        let d = self.d_model;
        let v = self.vocab_size;
        let ffn = self.d_ffn;
        let n_layers = self.n_layers;

        // 1. Initial embedding
        let mut layer_h = vec![vec![0.0f32; t_len * d]; n_layers + 1];
        for t in 0..t_len {
            let tok = x_seq[t];
            let embed = &self.embeddings[tok * d..(tok + 1) * d];
            let pos_e = &self.pos_embeddings[t * d..(t + 1) * d];
            for i in 0..d {
                layer_h[0][t * d + i] = embed[i] + pos_e[i];
            }
        }

        // Cache for all layers
        let mut q_cache = vec![vec![0.0f32; t_len * d]; n_layers];
        let mut k_cache = vec![vec![0.0f32; t_len * d]; n_layers];
        let mut v_cache = vec![vec![0.0f32; t_len * d]; n_layers];
        let mut attn_weights_cache = vec![vec![0.0f32; t_len * t_len]; n_layers];
        let mut attn_out_cache = vec![vec![0.0f32; t_len * d]; n_layers];
        let mut mid_h_cache = vec![vec![0.0f32; t_len * d]; n_layers];
        let mut ffn_raw_cache = vec![vec![0.0f32; t_len * ffn]; n_layers];
        let mut ffn_act_cache = vec![vec![0.0f32; t_len * ffn]; n_layers];

        let scale_attn = 1.0f32 / (d as f32).sqrt();

        // 2. Serial Forward pass through all L layers
        for l in 0..n_layers {
            let block = &self.blocks[l];
            let wq_v = MatrixView::new(&block.wq, d, d);
            let wk_v = MatrixView::new(&block.wk, d, d);
            let wv_v = MatrixView::new(&block.wv, d, d);
            let wo_v = MatrixView::new(&block.wo, d, d);
            let w1_v = MatrixView::new(&block.w1, ffn, d);
            let w2_v = MatrixView::new(&block.w2, d, ffn);

            let in_h = &layer_h[l];

            // Compute Q, K, V
            for t in 0..t_len {
                let h_t = &in_h[t * d..(t + 1) * d];
                matvec(&wq_v, h_t, &mut q_cache[l][t * d..(t + 1) * d]);
                matvec(&wk_v, h_t, &mut k_cache[l][t * d..(t + 1) * d]);
                matvec(&wv_v, h_t, &mut v_cache[l][t * d..(t + 1) * d]);
            }

            // Causal Self-Attention
            for i in 0..t_len {
                let q_i = &q_cache[l][i * d..(i + 1) * d];
                let mut scores = vec![-1e9f32; t_len];

                for j in 0..=i {
                    let k_j = &k_cache[l][j * d..(j + 1) * d];
                    scores[j] = dot(q_i, k_j) * scale_attn;
                }

                // Stable softmax over 0..=i
                let mut probs = vec![0.0f32; t_len];
                softmax(&scores[..=i], &mut probs[..=i]);
                for j in 0..=i {
                    attn_weights_cache[l][i * t_len + j] = probs[j];
                    let v_j = &v_cache[l][j * d..(j + 1) * d];
                    let p_j = probs[j];
                    for idx in 0..d {
                        attn_out_cache[l][i * d + idx] += p_j * v_j[idx];
                    }
                }
            }

            // Attn Output projection + Residual: mid_h = in_h + W_o * attn_out
            for t in 0..t_len {
                let a_t = &attn_out_cache[l][t * d..(t + 1) * d];
                let in_t = &in_h[t * d..(t + 1) * d];
                let mut wo_out = vec![0.0f32; d];
                matvec(&wo_v, a_t, &mut wo_out);
                for idx in 0..d {
                    mid_h_cache[l][t * d + idx] = in_t[idx] + wo_out[idx];
                }
            }

            // FFN: out_h = mid_h + W2 * gelu(W1 * mid_h)
            for t in 0..t_len {
                let mid_t = &mid_h_cache[l][t * d..(t + 1) * d];
                let raw_f = &mut ffn_raw_cache[l][t * ffn..(t + 1) * ffn];
                let act_f = &mut ffn_act_cache[l][t * ffn..(t + 1) * ffn];

                matvec(&w1_v, mid_t, raw_f);
                gelu(raw_f, act_f);

                let mut ffn_out = vec![0.0f32; d];
                matvec(&w2_v, act_f, &mut ffn_out);

                let out_t = &mut layer_h[l + 1][t * d..(t + 1) * d];
                for idx in 0..d {
                    out_t[idx] = mid_t[idx] + ffn_out[idx];
                }
            }
        }

        // 3. Prediction Head & Loss
        let head_view = MatrixView::new(&self.head, v, d);
        let mut grad_head_view = MatrixViewMut::new(&mut grads.grad_head, v, d);
        let mut delta_h = vec![0.0f32; t_len * d];
        let mut total_loss = 0.0f32;

        for t in 0..t_len {
            let final_h = &layer_h[n_layers][t * d..(t + 1) * d];
            let mut logits = vec![0.0f32; v];
            let mut probs = vec![0.0f32; v];
            let mut pred_grad = vec![0.0f32; v];

            matvec(&head_view, final_h, &mut logits);
            let loss = cross_entropy_loss_and_grad(&logits, y_seq[t], &mut probs, &mut pred_grad);
            total_loss += loss;

            outer_product_accumulate(&pred_grad, final_h, 1.0, &mut grad_head_view);
            let d_ht = &mut delta_h[t * d..(t + 1) * d];
            matvec_transposed(&head_view, &pred_grad, d_ht);
        }

        // 4. Serial Backpropagation through all L layers (Full Depth Backward Pass)
        for l in (0..n_layers).rev() {
            let block = &self.blocks[l];
            let bg = &mut grads.block_grads[l];

            let wq_v = MatrixView::new(&block.wq, d, d);
            let wk_v = MatrixView::new(&block.wk, d, d);
            let wv_v = MatrixView::new(&block.wv, d, d);
            let wo_v = MatrixView::new(&block.wo, d, d);
            let w1_v = MatrixView::new(&block.w1, ffn, d);
            let w2_v = MatrixView::new(&block.w2, d, ffn);

            let mut grad_wq = MatrixViewMut::new(&mut bg.grad_wq, d, d);
            let mut grad_wk = MatrixViewMut::new(&mut bg.grad_wk, d, d);
            let mut grad_wv = MatrixViewMut::new(&mut bg.grad_wv, d, d);
            let mut grad_wo = MatrixViewMut::new(&mut bg.grad_wo, d, d);
            let mut grad_w1 = MatrixViewMut::new(&mut bg.grad_w1, ffn, d);
            let mut grad_w2 = MatrixViewMut::new(&mut bg.grad_w2, d, ffn);

            let mut delta_mid = vec![0.0f32; t_len * d];

            // Backward through FFN
            for t in 0..t_len {
                let dh = &delta_h[t * d..(t + 1) * d];
                let act_f = &ffn_act_cache[l][t * ffn..(t + 1) * ffn];
                let raw_f = &ffn_raw_cache[l][t * ffn..(t + 1) * ffn];
                let mid_t = &mid_h_cache[l][t * d..(t + 1) * d];

                outer_product_accumulate(dh, act_f, 1.0, &mut grad_w2);

                let mut d_act = vec![0.0f32; ffn];
                matvec_transposed(&w2_v, dh, &mut d_act);

                let mut d_raw = vec![0.0f32; ffn];
                for i in 0..ffn {
                    let c = 0.79788456f32;
                    let x = raw_f[i];
                    let tanh_out = (c * (x + 0.044715 * x.powi(3))).tanh();
                    let sech2 = 1.0 - tanh_out * tanh_out;
                    let grad_gelu = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x);
                    d_raw[i] = d_act[i] * grad_gelu;
                }

                outer_product_accumulate(&d_raw, mid_t, 1.0, &mut grad_w1);

                let mut d_mid_ffn = vec![0.0f32; d];
                matvec_transposed(&w1_v, &d_raw, &mut d_mid_ffn);

                // delta_mid = dh (residual) + d_mid_ffn
                let dm = &mut delta_mid[t * d..(t + 1) * d];
                for idx in 0..d {
                    dm[idx] = dh[idx] + d_mid_ffn[idx];
                }
            }

            // Backward through Attention
            let in_h = &layer_h[l];
            let mut delta_in = vec![0.0f32; t_len * d];

            for t in 0..t_len {
                let dm = &delta_mid[t * d..(t + 1) * d];
                let a_t = &attn_out_cache[l][t * d..(t + 1) * d];
                outer_product_accumulate(dm, a_t, 1.0, &mut grad_wo);

                let mut d_attn_out = vec![0.0f32; d];
                matvec_transposed(&wo_v, dm, &mut d_attn_out);

                // Residual connection: delta_in += delta_mid
                for idx in 0..d {
                    delta_in[t * d + idx] += dm[idx];
                }

                // Attn gradient into Q, K, V
                for j in 0..=t {
                    let p_j = attn_weights_cache[l][t * t_len + j];
                    let v_j = &v_cache[l][j * d..(j + 1) * d];

                    // d_V_j += p_j * d_attn_out
                    let mut d_vj = vec![0.0f32; d];
                    for idx in 0..d { d_vj[idx] = p_j * d_attn_out[idx]; }
                    outer_product_accumulate(&d_vj, &in_h[j * d..(j + 1) * d], 1.0, &mut grad_wv);

                    // d_Q_t and d_K_j
                    let dot_v = dot(&d_attn_out, v_j);
                    let d_score = p_j * (dot_v) * scale_attn;

                    let q_t = &q_cache[l][t * d..(t + 1) * d];
                    let k_j = &k_cache[l][j * d..(j + 1) * d];

                    let mut d_qt = vec![0.0f32; d];
                    let mut d_kj = vec![0.0f32; d];
                    for idx in 0..d {
                        d_qt[idx] = d_score * k_j[idx];
                        d_kj[idx] = d_score * q_t[idx];
                    }

                    outer_product_accumulate(&d_qt, &in_h[t * d..(t + 1) * d], 1.0, &mut grad_wq);
                    outer_product_accumulate(&d_kj, &in_h[j * d..(j + 1) * d], 1.0, &mut grad_wk);

                    let mut dq_in = vec![0.0f32; d];
                    let mut dk_in = vec![0.0f32; d];
                    let mut dv_in = vec![0.0f32; d];
                    matvec_transposed(&wq_v, &d_qt, &mut dq_in);
                    matvec_transposed(&wk_v, &d_kj, &mut dk_in);
                    matvec_transposed(&wv_v, &d_vj, &mut dv_in);

                    for idx in 0..d {
                        delta_in[t * d + idx] += dq_in[idx];
                        delta_in[j * d + idx] += dk_in[idx] + dv_in[idx];
                    }
                }
            }

            // Propagate delta_in to previous layer: delta_h = delta_in
            delta_h = delta_in;
        }

        // 5. Backprop into token and positional embeddings
        for t in 0..t_len {
            let tok = x_seq[t];
            let dh = &delta_h[t * d..(t + 1) * d];

            let embed_slice = &mut grads.grad_embed[tok * d..(tok + 1) * d];
            let pos_slice = &mut grads.grad_pos_embed[t * d..(t + 1) * d];
            for idx in 0..d {
                embed_slice[idx] += dh[idx];
                pos_slice[idx] += dh[idx];
            }
        }

        total_loss
    }
}
