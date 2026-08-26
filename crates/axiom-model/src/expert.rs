//! Feed-Forward Expert MLP for sparse MoE computation with analytical gradients.

use axiom_core::activations::{gelu, gelu_grad, gelu_in_place};
use axiom_core::matvec::{matvec, matvec_transposed, matvec_transposed_accumulate, outer_product_accumulate};
use axiom_core::tensor::{MatrixView, MatrixViewMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A single FFN expert: Up-projection (d -> d_ffn) + GELU + Down-projection (d_ffn -> d).
#[derive(Debug, Clone)]
pub struct ExpertMLP {
    pub d_model: usize,
    pub d_ffn: usize,
    pub w_up: Vec<f32>,     // (d_ffn x d_model)
    pub w_down: Vec<f32>,   // (d_model x d_ffn)
}

/// Gradients for a single ExpertMLP.
#[derive(Debug, Clone)]
pub struct ExpertGrads {
    pub grad_w_up: Vec<f32>,
    pub grad_w_down: Vec<f32>,
}

impl ExpertGrads {
    pub fn new(d_model: usize, d_ffn: usize) -> Self {
        Self {
            grad_w_up: vec![0.0f32; d_ffn * d_model],
            grad_w_down: vec![0.0f32; d_model * d_ffn],
        }
    }

    pub fn zero(&mut self) {
        self.grad_w_up.fill(0.0f32);
        self.grad_w_down.fill(0.0f32);
    }
}

impl ExpertMLP {
    pub fn new(d_model: usize, d_ffn: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_up = (2.0f32 / d_model as f32).sqrt();
        let scale_down = (2.0f32 / d_ffn as f32).sqrt();

        let w_up: Vec<f32> = (0..d_ffn * d_model)
            .map(|_| rng.gen_range(-scale_up..scale_up))
            .collect();
        let w_down: Vec<f32> = (0..d_model * d_ffn)
            .map(|_| rng.gen_range(-scale_down..scale_down))
            .collect();

        Self {
            d_model,
            d_ffn,
            w_up,
            w_down,
        }
    }

    /// Forward pass storing intermediate activations for backward pass.
    /// `raw_ffn` (pre-activation z) and `act_ffn` (post-activation a) must have len >= d_ffn.
    #[inline]
    pub fn forward_with_cache(
        &self,
        x: &[f32],
        raw_ffn: &mut [f32],
        act_ffn: &mut [f32],
        out: &mut [f32],
    ) {
        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(out.len(), self.d_model);
        debug_assert!(raw_ffn.len() >= self.d_ffn);
        debug_assert!(act_ffn.len() >= self.d_ffn);

        let up_view = MatrixView::new(&self.w_up, self.d_ffn, self.d_model);
        let down_view = MatrixView::new(&self.w_down, self.d_model, self.d_ffn);

        // 1. z = W_up * x
        matvec(&up_view, x, &mut raw_ffn[..self.d_ffn]);

        // 2. a = GELU(z)
        gelu(&raw_ffn[..self.d_ffn], &mut act_ffn[..self.d_ffn]);

        // 3. out = W_down * a
        matvec(&down_view, &act_ffn[..self.d_ffn], out);
    }

    /// Fast forward pass (inference only).
    #[inline]
    pub fn forward(&self, x: &[f32], scratch_ffn: &mut [f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(out.len(), self.d_model);
        debug_assert!(scratch_ffn.len() >= self.d_ffn);

        let up_view = MatrixView::new(&self.w_up, self.d_ffn, self.d_model);
        let down_view = MatrixView::new(&self.w_down, self.d_model, self.d_ffn);

        matvec(&up_view, x, &mut scratch_ffn[..self.d_ffn]);
        gelu_in_place(&mut scratch_ffn[..self.d_ffn]);
        matvec(&down_view, &scratch_ffn[..self.d_ffn], out);
    }

    /// Analytical backward pass: computes gradients w.r.t W_down, W_up, and input x.
    /// Accumulates gradients into `grads.grad_w_down` and `grads.grad_w_up`,
    /// and adds input gradients to `grad_x`.
    #[inline]
    pub fn backward(
        &self,
        x: &[f32],
        raw_ffn: &[f32],
        act_ffn: &[f32],
        grad_out: &[f32],
        scratch_delta_a: &mut [f32],
        scratch_delta_z: &mut [f32],
        grads: &mut ExpertGrads,
        grad_x: &mut [f32],
    ) {
        let down_view = MatrixView::new(&self.w_down, self.d_model, self.d_ffn);
        let up_view = MatrixView::new(&self.w_up, self.d_ffn, self.d_model);

        // 1. grad_w_down += grad_out ⊗ act_ffn
        let mut grad_down_view = MatrixViewMut::new(&mut grads.grad_w_down, self.d_model, self.d_ffn);
        outer_product_accumulate(grad_out, &act_ffn[..self.d_ffn], 1.0, &mut grad_down_view);

        // 2. delta_a = W_down^T * grad_out
        matvec_transposed(&down_view, grad_out, &mut scratch_delta_a[..self.d_ffn]);

        // 3. delta_z = delta_a * GELU'(raw_ffn)
        gelu_grad(
            &raw_ffn[..self.d_ffn],
            &scratch_delta_a[..self.d_ffn],
            &mut scratch_delta_z[..self.d_ffn],
        );

        // 4. grad_w_up += delta_z ⊗ x
        let mut grad_up_view = MatrixViewMut::new(&mut grads.grad_w_up, self.d_ffn, self.d_model);
        outer_product_accumulate(&scratch_delta_z[..self.d_ffn], x, 1.0, &mut grad_up_view);

        // 5. grad_x += W_up^T * delta_z
        matvec_transposed_accumulate(&up_view, &scratch_delta_z[..self.d_ffn], grad_x);
    }
}
