//! Tensor slice views and basic vector primitives with zero dynamic allocation.

/// Immutable 2D matrix view over a contiguous slice in row-major layout.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [f32],
}

impl<'a> MatrixView<'a> {
    #[inline(always)]
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols, "MatrixView length mismatch");
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        debug_assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn row(&self, r: usize) -> &'a [f32] {
        debug_assert!(r < self.rows);
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }
}

/// Mutable 2D matrix view over a contiguous slice in row-major layout.
#[derive(Debug)]
pub struct MatrixViewMut<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a mut [f32],
}

impl<'a> MatrixViewMut<'a> {
    #[inline(always)]
    pub fn new(data: &'a mut [f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols, "MatrixViewMut length mismatch");
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        debug_assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        debug_assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c] = val;
    }

    #[inline(always)]
    pub fn row(&self, r: usize) -> &[f32] {
        debug_assert!(r < self.rows);
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        debug_assert!(r < self.rows);
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

/// Compute inner product (dot product) of two vectors: a · b
/// Optimized with 8-way unrolling to encourage AVX2/AVX-512 FMA auto-vectorization.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert_eq!(n, b.len(), "dot product vector lengths must match");

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        sum0 += ca[0] * cb[0];
        sum1 += ca[1] * cb[1];
        sum2 += ca[2] * cb[2];
        sum3 += ca[3] * cb[3];
        sum4 += ca[4] * cb[4];
        sum5 += ca[5] * cb[5];
        sum6 += ca[6] * cb[6];
        sum7 += ca[7] * cb[7];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    for (va, vb) in rem_a.iter().zip(rem_b.iter()) {
        total += va * vb;
    }
    total
}

/// In-place scaled vector addition: dst += alpha * src
#[inline]
pub fn vec_add_scaled(dst: &mut [f32], src: &[f32], alpha: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let mut chunks_d = dst.chunks_exact_mut(8);
    let chunks_s = src.chunks_exact(8);
    let rem_s = chunks_s.remainder();

    for (cd, cs) in chunks_d.by_ref().zip(chunks_s) {
        cd[0] += alpha * cs[0];
        cd[1] += alpha * cs[1];
        cd[2] += alpha * cs[2];
        cd[3] += alpha * cs[3];
        cd[4] += alpha * cs[4];
        cd[5] += alpha * cs[5];
        cd[6] += alpha * cs[6];
        cd[7] += alpha * cs[7];
    }

    let rem_d = chunks_d.into_remainder();
    for (d, s) in rem_d.iter_mut().zip(rem_s.iter()) {
        *d += alpha * s;
    }
}

/// In-place vector scaling: dst *= alpha
#[inline]
pub fn vec_scale(dst: &mut [f32], alpha: f32) {
    let mut chunks_d = dst.chunks_exact_mut(8);
    for cd in chunks_d.by_ref() {
        cd[0] *= alpha;
        cd[1] *= alpha;
        cd[2] *= alpha;
        cd[3] *= alpha;
        cd[4] *= alpha;
        cd[5] *= alpha;
        cd[6] *= alpha;
        cd[7] *= alpha;
    }
    let rem = chunks_d.into_remainder();
    for d in rem.iter_mut() {
        *d *= alpha;
    }
}

/// In-place vector zeroing
#[inline(always)]
pub fn vec_zero(dst: &mut [f32]) {
    dst.fill(0.0f32);
}

/// In-place vector copy: dst = src
#[inline(always)]
pub fn vec_copy(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}
