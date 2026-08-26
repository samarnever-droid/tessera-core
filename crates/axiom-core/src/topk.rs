//! Zero-allocation top-k selection kernels for sparse MoE expert routing.

/// Fast specialized top-2 selection from a slice of scores (e.g. E = 8 to 64 experts).
/// Returns array of `(score, index)` sorted descending by score.
/// Completely zero heap allocation.
#[inline]
pub fn top2(scores: &[f32]) -> [(f32, usize); 2] {
    let n = scores.len();
    debug_assert!(n >= 2, "top2 requires at least 2 elements");

    let mut first_val = f32::NEG_INFINITY;
    let mut first_idx = 0;
    let mut second_val = f32::NEG_INFINITY;
    let mut second_idx = 0;

    for (i, &val) in scores.iter().enumerate() {
        if val > first_val {
            second_val = first_val;
            second_idx = first_idx;
            first_val = val;
            first_idx = i;
        } else if val > second_val {
            second_val = val;
            second_idx = i;
        }
    }

    [(first_val, first_idx), (second_val, second_idx)]
}

/// Generic zero-allocation top-K selection for small K (e.g. K <= 8, E <= 64).
/// Returns array of `(score, index)` sorted descending.
#[inline]
pub fn topk<const K: usize>(scores: &[f32]) -> [(f32, usize); K] {
    debug_assert!(K > 0, "K must be > 0");
    debug_assert!(scores.len() >= K, "scores length must be >= K");

    let mut result = [(f32::NEG_INFINITY, 0usize); K];

    for (i, &val) in scores.iter().enumerate() {
        if val > result[K - 1].0 {
            // Find insertion index in sorted array of length K
            let mut insert_pos = K - 1;
            while insert_pos > 0 && val > result[insert_pos - 1].0 {
                insert_pos -= 1;
            }
            // Shift elements down
            let mut j = K - 1;
            while j > insert_pos {
                result[j] = result[j - 1];
                j -= 1;
            }
            result[insert_pos] = (val, i);
        }
    }

    result
}

/// Computes normalized router weights for the top-k selected experts.
/// Takes raw top-k scores and computes softmax over only those active k experts.
#[inline]
pub fn topk_softmax<const K: usize>(top_elements: &[(f32, usize); K], out_weights: &mut [f32; K]) {
    let mut max_val = f32::NEG_INFINITY;
    for &(val, _) in top_elements.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0f32;
    for (i, &(val, _)) in top_elements.iter().enumerate() {
        let exp_val = (val - max_val).exp();
        out_weights[i] = exp_val;
        sum += exp_val;
    }

    let inv_sum = if sum > 0.0f32 { 1.0f32 / sum } else { 0.0f32 };
    for w in out_weights.iter_mut() {
        *w *= inv_sum;
    }
}
