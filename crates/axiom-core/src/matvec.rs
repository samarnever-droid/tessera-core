//! High-performance matrix-vector multiplication kernels for CPU inference and training.
//! Designed for zero dynamic allocation and auto-vectorization across d = 512, 1024, 2048.

use crate::tensor::{dot, vec_add_scaled, MatrixView, MatrixViewMut};

/// Matrix-vector multiplication: y = W * x
/// W: (rows x cols)
/// x: (cols)
/// y: (rows) - pre-allocated destination
#[inline]
pub fn matvec(w: &MatrixView, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.cols, x.len(), "matvec: W cols != x len");
    debug_assert_eq!(w.rows, y.len(), "matvec: W rows != y len");

    for r in 0..w.rows {
        let row_slice = w.row(r);
        y[r] = dot(row_slice, x);
    }
}

/// Matrix-vector multiplication with accumulation: y += W * x
/// W: (rows x cols)
/// x: (cols)
/// y: (rows) - pre-allocated destination
#[inline]
pub fn matvec_accumulate(w: &MatrixView, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.cols, x.len(), "matvec_accumulate: W cols != x len");
    debug_assert_eq!(w.rows, y.len(), "matvec_accumulate: W rows != y len");

    for r in 0..w.rows {
        let row_slice = w.row(r);
        y[r] += dot(row_slice, x);
    }
}

/// Transposed matrix-vector multiplication: y = W^T * x
/// W: (rows x cols)
/// x: (rows)
/// y: (cols) - pre-allocated destination
#[inline]
pub fn matvec_transposed(w: &MatrixView, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.rows, x.len(), "matvec_transposed: W rows != x len");
    debug_assert_eq!(w.cols, y.len(), "matvec_transposed: W cols != y len");

    y.fill(0.0f32);
    matvec_transposed_accumulate(w, x, y);
}

/// Transposed matrix-vector multiplication with accumulation: y += W^T * x
/// W: (rows x cols)
/// x: (rows)
/// y: (cols) - pre-allocated destination
#[inline]
pub fn matvec_transposed_accumulate(w: &MatrixView, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.rows, x.len(), "matvec_transposed_accumulate: W rows != x len");
    debug_assert_eq!(w.cols, y.len(), "matvec_transposed_accumulate: W cols != y len");

    for r in 0..w.rows {
        let xr = x[r];
        if xr != 0.0f32 {
            let row_slice = w.row(r);
            vec_add_scaled(y, row_slice, xr);
        }
    }
}

/// Outer product accumulation: W += alpha * (u ⊗ v)
/// W: (m x n)
/// u: (m)
/// v: (n)
#[inline]
pub fn outer_product_accumulate(
    u: &[f32],
    v: &[f32],
    alpha: f32,
    w: &mut MatrixViewMut,
) {
    debug_assert_eq!(u.len(), w.rows, "outer_product: u len != W rows");
    debug_assert_eq!(v.len(), w.cols, "outer_product: v len != W cols");

    for r in 0..w.rows {
        let scale = alpha * u[r];
        if scale != 0.0f32 {
            let row = w.row_mut(r);
            vec_add_scaled(row, v, scale);
        }
    }
}
