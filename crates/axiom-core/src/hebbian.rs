//! Hebbian fast associative memory update and recall kernels.
//! Supports fused exponential decay + rank-1 outer product updates,
//! associative recall (M * h), Frobenius normalization, and Oja's rule.

use crate::matvec::matvec;
use crate::tensor::{vec_add_scaled, vec_scale, MatrixView, MatrixViewMut};

/// Online Hebbian associative memory state of dimension (d x d).
#[derive(Debug, Clone)]
pub struct HebbianMemory {
    pub dim: usize,
    pub decay: f32,
    pub lr: f32,
    pub m: Vec<f32>,
}

impl HebbianMemory {
    /// Initialize a zeroed (d x d) Hebbian associative memory matrix.
    pub fn new(dim: usize, decay: f32, lr: f32) -> Self {
        Self {
            dim,
            decay,
            lr,
            m: vec![0.0f32; dim * dim],
        }
    }

    /// Reset memory matrix to all zeros.
    #[inline]
    pub fn clear(&mut self) {
        self.m.fill(0.0f32);
    }

    /// View as immutable MatrixView.
    #[inline]
    pub fn as_view(&self) -> MatrixView<'_> {
        MatrixView::new(&self.m, self.dim, self.dim)
    }

    /// View as mutable MatrixViewMut.
    #[inline]
    pub fn as_view_mut(&mut self) -> MatrixViewMut<'_> {
        MatrixViewMut::new(&mut self.m, self.dim, self.dim)
    }

    /// Associative Recall: m_t = M * h
    /// Reads associative context vector from memory using current hidden state.
    #[inline]
    pub fn recall(&self, h: &[f32], out_m: &mut [f32]) {
        let view = self.as_view();
        matvec(&view, h, out_m);
    }

    /// Online Hebbian Write (fused decay + rank-1 update):
    /// M = lambda * M + eta * (h ⊗ h^T)
    /// Completely zero dynamic allocation; single contiguous pass over matrix memory.
    #[inline]
    pub fn update(&mut self, h: &[f32]) {
        hebbian_fused_update(&mut self.m, self.dim, h, self.decay, self.lr);
    }

    /// Compute Frobenius norm: ||M||_F = sqrt(sum(M_ij^2))
    #[inline]
    pub fn frobenius_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for &val in &self.m {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }

    /// Normalize memory matrix by its Frobenius norm if it exceeds max_norm (§9.3 mitigation).
    #[inline]
    pub fn clip_frobenius(&mut self, max_norm: f32) {
        let norm = self.frobenius_norm();
        if norm > max_norm && norm > 1e-8 {
            let scale = max_norm / norm;
            vec_scale(&mut self.m, scale);
        }
    }

    /// Online Oja's rule update (§9.3 mitigation):
    /// ΔM = η * (h - M*h) * h^T
    /// Prevents unbounded growth of eigenvalues while learning principal components.
    #[inline]
    pub fn update_oja(&mut self, h: &[f32], scratch_recall: &mut [f32]) {
        debug_assert_eq!(h.len(), self.dim);
        debug_assert_eq!(scratch_recall.len(), self.dim);

        // 1. recall = M * h
        self.recall(h, scratch_recall);

        // 2. error = h - recall
        for (err, &hi) in scratch_recall.iter_mut().zip(h.iter()) {
            *err = hi - *err;
        }

        // 3. M += lr * (error ⊗ h^T)
        for r in 0..self.dim {
            let scale = self.lr * scratch_recall[r];
            if scale != 0.0f32 {
                let start = r * self.dim;
                let row = &mut self.m[start..start + self.dim];
                vec_add_scaled(row, h, scale);
            }
        }
    }
}

/// Fused Hebbian update kernel: M_ij = lambda * M_ij + eta * h_i * h_j
#[inline]
pub fn hebbian_fused_update(m_data: &mut [f32], dim: usize, h: &[f32], decay: f32, lr: f32) {
    debug_assert_eq!(h.len(), dim);
    debug_assert_eq!(m_data.len(), dim * dim);

    for r in 0..dim {
        let hr_scale = lr * h[r];
        let start = r * dim;
        let row = &mut m_data[start..start + dim];

        // Process row with 8-way unrolling
        let mut chunks_row = row.chunks_exact_mut(8);
        let chunks_h = h.chunks_exact(8);
        let rem_h = chunks_h.remainder();

        for (crow, ch) in chunks_row.by_ref().zip(chunks_h) {
            crow[0] = decay * crow[0] + hr_scale * ch[0];
            crow[1] = decay * crow[1] + hr_scale * ch[1];
            crow[2] = decay * crow[2] + hr_scale * ch[2];
            crow[3] = decay * crow[3] + hr_scale * ch[3];
            crow[4] = decay * crow[4] + hr_scale * ch[4];
            crow[5] = decay * crow[5] + hr_scale * ch[5];
            crow[6] = decay * crow[6] + hr_scale * ch[6];
            crow[7] = decay * crow[7] + hr_scale * ch[7];
        }

        let rem_row = chunks_row.into_remainder();
        for (elem, &hv) in rem_row.iter_mut().zip(rem_h.iter()) {
            *elem = decay * *elem + hr_scale * hv;
        }
    }
}
