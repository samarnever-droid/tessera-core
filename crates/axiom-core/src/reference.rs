//! Naive, unoptimized reference implementations of all kernels for numerical parity testing.

/// Naive reference matrix-vector multiplication: y = W * x
pub fn ref_matvec(w_data: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w_data.len(), rows * cols);
    assert_eq!(x.len(), cols);
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f64;
        for c in 0..cols {
            sum += w_data[r * cols + c] as f64 * x[c] as f64;
        }
        y[r] = sum as f32;
    }
    y
}

/// Naive reference transposed matrix-vector multiplication: y = W^T * x
pub fn ref_matvec_transposed(w_data: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w_data.len(), rows * cols);
    assert_eq!(x.len(), rows);
    let mut y = vec![0.0f32; cols];
    for c in 0..cols {
        let mut sum = 0.0f64;
        for r in 0..rows {
            sum += w_data[r * cols + c] as f64 * x[r] as f64;
        }
        y[c] = sum as f32;
    }
    y
}

/// Naive reference top-k selection using full sorting.
pub fn ref_topk(scores: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut indexed: Vec<(f32, usize)> = scores.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

/// Naive reference softmax.
pub fn ref_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f64> = logits.iter().map(|&x| ((x - max_val) as f64).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&x| (x / sum) as f32).collect()
}

/// Naive reference Hebbian update: M = lambda * M + eta * (h ⊗ h^T)
pub fn ref_hebbian_update(m_data: &mut [f32], dim: usize, h: &[f32], decay: f32, lr: f32) {
    for i in 0..dim {
        for j in 0..dim {
            let idx = i * dim + j;
            m_data[idx] = decay * m_data[idx] + lr * h[i] * h[j];
        }
    }
}

/// Naive reference cross entropy loss.
pub fn ref_cross_entropy(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let probs = ref_softmax(logits);
    let loss = -probs[target].ln();
    let mut grad = probs.clone();
    grad[target] -= 1.0;
    (loss, grad)
}
