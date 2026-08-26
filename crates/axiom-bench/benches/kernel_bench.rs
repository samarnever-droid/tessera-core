use criterion::{black_box, criterion_group, criterion_main, Criterion};
use axiom_core::matvec::matvec;
use axiom_core::tensor::MatrixView;
use axiom_core::hebbian::hebbian_fused_update;
use axiom_core::topk::top2;
use axiom_core::softmax::softmax;

fn benchmark_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("axiom_kernels");

    // Matvec 512x512
    let dim = 512;
    let w_data = vec![0.01f32; dim * dim];
    let x = vec![0.01f32; dim];
    let mut y = vec![0.0f32; dim];
    let view = MatrixView::new(&w_data, dim, dim);

    group.bench_function("matvec_512x512", |b| {
        b.iter(|| {
            matvec(&view, black_box(&x), black_box(&mut y));
        })
    });

    // Top-2 (E=16)
    let scores = vec![0.5f32; 16];
    group.bench_function("top2_E16", |b| {
        b.iter(|| {
            top2(black_box(&scores))
        })
    });

    // Softmax (V=256)
    let logits = vec![1.0f32; 256];
    let mut probs = vec![0.0f32; 256];
    group.bench_function("softmax_V256", |b| {
        b.iter(|| {
            softmax(black_box(&logits), black_box(&mut probs));
        })
    });

    // Hebbian Update 512
    let mut m_data = vec![0.0f32; dim * dim];
    let h = vec![0.5f32; dim];
    group.bench_function("hebbian_update_512", |b| {
        b.iter(|| {
            hebbian_fused_update(black_box(&mut m_data), dim, black_box(&h), 0.999, 1e-4);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_kernels);
criterion_main!(benches);
