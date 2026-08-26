//! Numerically stable softmax, cross-entropy, and entropy kernels.

/// Compute numerically stable softmax in-place / into output slice.
/// out_probs: pre-allocated slice of same length as logits.
#[inline]
pub fn softmax(logits: &[f32], out_probs: &mut [f32]) {
    let n = logits.len();
    debug_assert_eq!(n, out_probs.len(), "softmax: length mismatch");
    if n == 0 {
        return;
    }

    // Step 1: Find maximum for numerical stability
    let mut max_val = logits[0];
    for &val in &logits[1..] {
        if val > max_val {
            max_val = val;
        }
    }

    // Step 2: Exponentiate and sum
    let mut sum = 0.0f32;
    for (out, &l) in out_probs.iter_mut().zip(logits.iter()) {
        let e = (l - max_val).exp();
        *out = e;
        sum += e;
    }

    // Step 3: Normalize
    let inv_sum = if sum > 0.0f32 { 1.0f32 / sum } else { 0.0f32 };
    for out in out_probs.iter_mut() {
        *out *= inv_sum;
    }
}

/// Numerically stable softmax with temperature scaling: p_i = exp(l_i / T) / sum(exp(l_j / T))
#[inline]
pub fn softmax_temperature(logits: &[f32], temperature: f32, out_probs: &mut [f32]) {
    let n = logits.len();
    debug_assert_eq!(n, out_probs.len(), "softmax_temperature: length mismatch");
    if n == 0 {
        return;
    }

    let inv_temp = 1.0f32 / temperature.max(1e-6);

    let mut max_val = logits[0] * inv_temp;
    for &val in &logits[1..] {
        let scaled = val * inv_temp;
        if scaled > max_val {
            max_val = scaled;
        }
    }

    let mut sum = 0.0f32;
    for (out, &l) in out_probs.iter_mut().zip(logits.iter()) {
        let e = (l * inv_temp - max_val).exp();
        *out = e;
        sum += e;
    }

    let inv_sum = if sum > 0.0f32 { 1.0f32 / sum } else { 0.0f32 };
    for out in out_probs.iter_mut() {
        *out *= inv_sum;
    }
}

/// Computes cross-entropy loss and gradient with respect to logits:
/// Loss = -log(softmax(logits)[target])
/// Grad = softmax(logits) - OneHot(target)
/// Returns the scalar loss value, and writes probabilities and gradients into pre-allocated slices.
#[inline]
pub fn cross_entropy_loss_and_grad(
    logits: &[f32],
    target: usize,
    out_probs: &mut [f32],
    out_grad: &mut [f32],
) -> f32 {
    let n = logits.len();
    debug_assert_eq!(n, out_probs.len());
    debug_assert_eq!(n, out_grad.len());
    debug_assert!(target < n, "cross_entropy: target index out of range");

    // 1. Softmax
    softmax(logits, out_probs);

    // 2. Loss = -ln(p_target)
    let p_target = out_probs[target].max(1e-12);
    let loss = -p_target.ln();

    // 3. Grad = p - 1_{target}
    out_grad.copy_from_slice(out_probs);
    out_grad[target] -= 1.0f32;

    loss
}

/// Compute Shannon entropy H(p) = -sum(p_i * ln(p_i))
/// Used for triggering copy buffer and expert load balancing.
#[inline]
pub fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-12 {
            h -= p * p.ln();
        }
    }
    h
}
