//! Vector SIMD Distance Acceleration (Cosine, Euclidean L2, Dot Product).

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return dot_product_avx2_fma(&a[..len], &b[..len]);
            }
        }
    }
    dot_product_scalar(&a[..len], &b[..len])
}

#[inline]
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return euclidean_distance_sq_avx2(&a[..len], &b[..len]);
            }
        }
    }
    euclidean_distance_sq_scalar(&a[..len], &b[..len])
}

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return cosine_similarity_avx2(&a[..len], &b[..len]);
            }
        }
    }
    cosine_similarity_scalar(&a[..len], &b[..len])
}

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        sum += a[idx] * b[idx]
            + a[idx + 1] * b[idx + 1]
            + a[idx + 2] * b[idx + 2]
            + a[idx + 3] * b[idx + 3];
    }
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline(always)]
fn euclidean_distance_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for i in (chunks * 4)..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[inline(always)]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks32 = len / 32;
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    for _ in 0..chunks32 {
        let va0 = _mm256_loadu_ps(ptr_a);
        let vb0 = _mm256_loadu_ps(ptr_b);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(ptr_a.add(8));
        let vb1 = _mm256_loadu_ps(ptr_b.add(8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        let va2 = _mm256_loadu_ps(ptr_a.add(16));
        let vb2 = _mm256_loadu_ps(ptr_b.add(16));
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

        let va3 = _mm256_loadu_ps(ptr_a.add(24));
        let vb3 = _mm256_loadu_ps(ptr_b.add(24));
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);

        ptr_a = ptr_a.add(32);
        ptr_b = ptr_b.add(32);
    }

    let chunks8 = (len % 32) / 8;
    for _ in 0..chunks8 {
        let va = _mm256_loadu_ps(ptr_a);
        let vb = _mm256_loadu_ps(ptr_b);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        ptr_a = ptr_a.add(8);
        ptr_b = ptr_b.add(8);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    let hsum = hsum_avx2(acc0);
    let rem_start = (len / 8) * 8;
    let mut rem_sum = 0.0f32;
    for i in rem_start..len {
        rem_sum += a[i] * b[i];
    }
    hsum + rem_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn euclidean_distance_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks32 = len / 32;
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    for _ in 0..chunks32 {
        let diff0 = _mm256_sub_ps(_mm256_loadu_ps(ptr_a), _mm256_loadu_ps(ptr_b));
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        let diff1 = _mm256_sub_ps(_mm256_loadu_ps(ptr_a.add(8)), _mm256_loadu_ps(ptr_b.add(8)));
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        let diff2 = _mm256_sub_ps(_mm256_loadu_ps(ptr_a.add(16)), _mm256_loadu_ps(ptr_b.add(16)));
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        let diff3 = _mm256_sub_ps(_mm256_loadu_ps(ptr_a.add(24)), _mm256_loadu_ps(ptr_b.add(24)));
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

        ptr_a = ptr_a.add(32);
        ptr_b = ptr_b.add(32);
    }

    let chunks8 = (len % 32) / 8;
    for _ in 0..chunks8 {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(ptr_a), _mm256_loadu_ps(ptr_b));
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
        ptr_a = ptr_a.add(8);
        ptr_b = ptr_b.add(8);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    let hsum = hsum_avx2(acc0);
    let rem_start = (len / 8) * 8;
    let mut rem_sum = 0.0f32;
    for i in rem_start..len {
        let d = a[i] - b[i];
        rem_sum += d * d;
    }
    hsum + rem_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut dot_acc0 = _mm256_setzero_ps();
    let mut dot_acc1 = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    let chunks16 = len / 16;
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    for _ in 0..chunks16 {
        let va0 = _mm256_loadu_ps(ptr_a);
        let vb0 = _mm256_loadu_ps(ptr_b);
        dot_acc0 = _mm256_fmadd_ps(va0, vb0, dot_acc0);
        norm_a_acc = _mm256_fmadd_ps(va0, va0, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb0, vb0, norm_b_acc);

        let va1 = _mm256_loadu_ps(ptr_a.add(8));
        let vb1 = _mm256_loadu_ps(ptr_b.add(8));
        dot_acc1 = _mm256_fmadd_ps(va1, vb1, dot_acc1);
        norm_a_acc = _mm256_fmadd_ps(va1, va1, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb1, vb1, norm_b_acc);

        ptr_a = ptr_a.add(16);
        ptr_b = ptr_b.add(16);
    }

    let chunks8 = (len % 16) / 8;
    for _ in 0..chunks8 {
        let va = _mm256_loadu_ps(ptr_a);
        let vb = _mm256_loadu_ps(ptr_b);
        dot_acc0 = _mm256_fmadd_ps(va, vb, dot_acc0);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
        ptr_a = ptr_a.add(8);
        ptr_b = ptr_b.add(8);
    }

    dot_acc0 = _mm256_add_ps(dot_acc0, dot_acc1);
    let mut dot = hsum_avx2(dot_acc0);
    let mut norm_a = hsum_avx2(norm_a_acc);
    let mut norm_b = hsum_avx2(norm_b_acc);

    let rem_start = (len / 8) * 8;
    for i in rem_start..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let v128 = _mm_add_ps(vlow, vhigh);
    let shuf = _mm_movehdup_ps(v128);
    let sums = _mm_add_ps(v128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(total)
}
