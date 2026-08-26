//! Activation functions and loss primitives for local layer training.

use std::f32::consts::PI;

/// Element-wise Sigmoid activation: 1 / (1 + exp(-x))
#[inline]
pub fn sigmoid(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = 1.0f32 / (1.0f32 + (-v).exp());
    }
}

/// Sigmoid backward gradient: grad_in = grad_out * sig * (1 - sig)
#[inline]
pub fn sigmoid_grad(sig_out: &[f32], grad_out: &[f32], grad_in: &mut [f32]) {
    debug_assert_eq!(sig_out.len(), grad_out.len());
    debug_assert_eq!(sig_out.len(), grad_in.len());
    for ((gi, &go), &s) in grad_in.iter_mut().zip(grad_out.iter()).zip(sig_out.iter()) {
        *gi = go * s * (1.0f32 - s);
    }
}

/// Element-wise GELU activation (tanh approximation):
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let sqrt_2_over_pi = (2.0f32 / PI).sqrt();
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        let v3 = v * v * v;
        let inner = sqrt_2_over_pi * (v + 0.044715f32 * v3);
        *o = 0.5f32 * v * (1.0f32 + inner.tanh());
    }
}

/// In-place GELU activation
#[inline]
pub fn gelu_in_place(buf: &mut [f32]) {
    let sqrt_2_over_pi = (2.0f32 / PI).sqrt();
    for v in buf.iter_mut() {
        let x = *v;
        let v3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715f32 * v3);
        *v = 0.5f32 * x * (1.0f32 + inner.tanh());
    }
}

/// Element-wise GELU backward gradient
#[inline]
pub fn gelu_grad(x: &[f32], grad_out: &[f32], grad_in: &mut [f32]) {
    debug_assert_eq!(x.len(), grad_out.len());
    debug_assert_eq!(x.len(), grad_in.len());
    let sqrt_2_over_pi = (2.0f32 / PI).sqrt();
    for ((gi, &go), &v) in grad_in.iter_mut().zip(grad_out.iter()).zip(x.iter()) {
        let v2 = v * v;
        let v3 = v2 * v;
        let inner = sqrt_2_over_pi * (v + 0.044715f32 * v3);
        let tanh_inner = inner.tanh();
        let sech2 = 1.0f32 - tanh_inner * tanh_inner;
        let d_inner = sqrt_2_over_pi * (1.0f32 + 3.0f32 * 0.044715f32 * v2);
        let cdf = 0.5f32 * (1.0f32 + tanh_inner);
        let pdf = 0.5f32 * v * sech2 * d_inner;
        *gi = go * (cdf + pdf);
    }
}

/// SwiGLU forward pass: out = (x1 * sigmoid(x1)) * x2
#[inline]
pub fn swiglu(x1: &[f32], x2: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x1.len(), x2.len());
    debug_assert_eq!(x1.len(), out.len());
    for (o, (&v1, &v2)) in out.iter_mut().zip(x1.iter().zip(x2.iter())) {
        let sig = 1.0f32 / (1.0f32 + (-v1).exp());
        *o = (v1 * sig) * v2;
    }
}

/// Mean Squared Error (MSE) loss and gradient for reconstruction check (§4.2):
/// L_recon = (1/d) * sum_i (pred_i - target_i)^2
/// grad_pred_i = (2/d) * (pred_i - target_i)
#[inline]
pub fn mse_loss_and_grad(pred: &[f32], target: &[f32], grad_pred: &mut [f32]) -> f32 {
    let d = pred.len();
    debug_assert_eq!(d, target.len());
    debug_assert_eq!(d, grad_pred.len());
    if d == 0 {
        return 0.0f32;
    }

    let inv_d = 1.0f32 / (d as f32);
    let factor = 2.0f32 * inv_d;
    let mut sum_sq = 0.0f32;

    for ((gp, &p), &t) in grad_pred.iter_mut().zip(pred.iter()).zip(target.iter()) {
        let diff = p - t;
        sum_sq += diff * diff;
        *gp = factor * diff;
    }

    sum_sq * inv_d
}

/// Squared L2 distance: ||a - b||^2
#[inline]
pub fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum_sq = 0.0f32;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let diff = va - vb;
        sum_sq += diff * diff;
    }
    sum_sq
}
