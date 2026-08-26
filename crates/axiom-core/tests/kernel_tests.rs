use axiom_core::activations::*;
use axiom_core::buffer::*;
use axiom_core::hebbian::*;
use axiom_core::matvec::*;
use axiom_core::reference::*;
use axiom_core::softmax::*;
use axiom_core::tensor::*;
use axiom_core::topk::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const EPSILON: f32 = 1e-4;

#[test]
fn test_matvec_dimensions() {
    let mut rng = StdRng::seed_from_u64(42);
    let dims = [1, 7, 8, 15, 16, 64, 512, 1024, 2048];

    for &rows in &dims {
        for &cols in &[8, 16, 64, 512] {
            let w_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut y = vec![0.0f32; rows];

            let view = MatrixView::new(&w_data, rows, cols);
            matvec(&view, &x, &mut y);

            let ref_y = ref_matvec(&w_data, rows, cols, &x);
            for r in 0..rows {
                let diff = (y[r] - ref_y[r]).abs();
                assert!(
                    diff < EPSILON * (cols as f32).sqrt(),
                    "matvec mismatch at rows={}, cols={}, r={}: got {}, expected {}",
                    rows, cols, r, y[r], ref_y[r]
                );
            }
        }
    }
}

#[test]
fn test_matvec_transposed() {
    let mut rng = StdRng::seed_from_u64(1337);
    let rows = 128;
    let cols = 256;

    let w_data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x: Vec<f32> = (0..rows).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut y = vec![0.0f32; cols];

    let view = MatrixView::new(&w_data, rows, cols);
    matvec_transposed(&view, &x, &mut y);

    let ref_y = ref_matvec_transposed(&w_data, rows, cols, &x);
    for c in 0..cols {
        let diff = (y[c] - ref_y[c]).abs();
        assert!(
            diff < EPSILON * (rows as f32).sqrt(),
            "matvec_transposed mismatch at c={}: got {}, expected {}",
            c, y[c], ref_y[c]
        );
    }
}

#[test]
fn test_outer_product_accumulate() {
    let mut rng = StdRng::seed_from_u64(999);
    let rows = 64;
    let cols = 128;
    let u: Vec<f32> = (0..rows).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let alpha = 0.5f32;

    let mut w = vec![0.0f32; rows * cols];
    let mut w_view = MatrixViewMut::new(&mut w, rows, cols);
    outer_product_accumulate(&u, &v, alpha, &mut w_view);

    for r in 0..rows {
        for c in 0..cols {
            let expected = alpha * u[r] * v[c];
            let actual = w_view.get(r, c);
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn test_top2_selection() {
    let mut rng = StdRng::seed_from_u64(777);
    for &e in &[2, 4, 8, 16, 32, 64] {
        let scores: Vec<f32> = (0..e).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let top = top2(&scores);
        let ref_top = ref_topk(&scores, 2);

        assert_eq!(top[0].1, ref_top[0].1, "Top-1 index mismatch for E={}", e);
        assert_eq!(top[1].1, ref_top[1].1, "Top-2 index mismatch for E={}", e);
        assert!((top[0].0 - ref_top[0].0).abs() < 1e-6);
        assert!((top[1].0 - ref_top[1].0).abs() < 1e-6);
    }
}

#[test]
fn test_topk_generic() {
    let mut rng = StdRng::seed_from_u64(888);
    let scores: Vec<f32> = (0..64).map(|_| rng.gen_range(-50.0..50.0)).collect();

    let top4: [(f32, usize); 4] = topk::<4>(&scores);
    let ref_top4 = ref_topk(&scores, 4);

    for i in 0..4 {
        assert_eq!(top4[i].1, ref_top4[i].1);
        assert!((top4[i].0 - ref_top4[i].0).abs() < 1e-6);
    }
}

#[test]
fn test_softmax_numerical_stability() {
    // Test normal logits
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut probs = vec![0.0; 5];
    softmax(&logits, &mut probs);
    let ref_p = ref_softmax(&logits);
    for (p, rp) in probs.iter().zip(ref_p.iter()) {
        assert!((p - rp).abs() < 1e-6);
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Test extreme logits (overflow prevention: values > 10,000)
    let extreme_logits = vec![10000.0, 10002.0, 9998.0, 10005.0];
    let mut extreme_probs = vec![0.0; 4];
    softmax(&extreme_logits, &mut extreme_probs);
    for &p in &extreme_probs {
        assert!(!p.is_nan());
        assert!(!p.is_infinite());
        assert!(p >= 0.0 && p <= 1.0);
    }
    let ext_sum: f32 = extreme_probs.iter().sum();
    assert!((ext_sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_loss_and_grad() {
    let logits = vec![2.0, 1.0, 0.1, -1.5];
    let target = 0;
    let mut probs = vec![0.0; 4];
    let mut grad = vec![0.0; 4];

    let loss = cross_entropy_loss_and_grad(&logits, target, &mut probs, &mut grad);
    let (ref_loss, ref_grad) = ref_cross_entropy(&logits, target);

    assert!((loss - ref_loss).abs() < 1e-6);
    for (g, rg) in grad.iter().zip(ref_grad.iter()) {
        assert!((g - rg).abs() < 1e-6);
    }
}

#[test]
fn test_circular_token_buffer() {
    let capacity = 4;
    let mut buf = CircularTokenBuffer::new(capacity);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());

    buf.push(10);
    buf.push(20);
    assert_eq!(buf.len(), 2);
    let items: Vec<u32> = buf.iter().collect();
    assert_eq!(items, vec![10, 20]);

    buf.push(30);
    buf.push(40);
    assert_eq!(buf.len(), 4);
    let items: Vec<u32> = buf.iter().collect();
    assert_eq!(items, vec![10, 20, 30, 40]);

    // Push past capacity - FIFO overwriting oldest
    buf.push(50);
    assert_eq!(buf.len(), 4);
    let items: Vec<u32> = buf.iter().collect();
    assert_eq!(items, vec![20, 30, 40, 50]);

    buf.push(60);
    let items: Vec<u32> = buf.iter().collect();
    assert_eq!(items, vec![30, 40, 50, 60]);

    buf.clear();
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn test_buffer_attention() {
    let capacity = 8;
    let mut buf = CircularTokenBuffer::new(capacity);
    buf.push(1);
    buf.push(3);
    buf.push(1); // Duplicate token in buffer

    let vocab_size = 5;
    let embed_dim = 4;
    let embed_table = vec![
        0.1, 0.2, 0.3, 0.4, // token 0
        1.0, 0.0, 0.0, 0.0, // token 1
        0.0, 1.0, 0.0, 0.0, // token 2
        0.0, 0.0, 1.0, 0.0, // token 3
        0.0, 0.0, 0.0, 1.0, // token 4
    ];

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let mut scratch_scores = vec![0.0; capacity];
    let mut scratch_probs = vec![0.0; capacity];
    let mut out_vocab_dist = vec![0.2; vocab_size]; // Uniform initial dist

    buffer_attention(
        &buf,
        &query,
        &embed_table,
        vocab_size,
        embed_dim,
        &mut scratch_scores,
        &mut scratch_probs,
        &mut out_vocab_dist,
        0.5,
    );

    let sum: f32 = out_vocab_dist.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Interpolated distribution must sum to 1.0, got {}", sum);
    assert!(out_vocab_dist[1] > out_vocab_dist[2], "Token 1 should receive higher copy probability");
}

#[test]
fn test_hebbian_memory_update_and_recall() {
    let dim = 16;
    let decay = 0.99f32;
    let lr = 0.01f32;
    let mut mem = HebbianMemory::new(dim, decay, lr);

    let h = vec![1.0f32; dim];
    let mut ref_m = vec![0.0f32; dim * dim];

    // Single step
    mem.update(&h);
    ref_hebbian_update(&mut ref_m, dim, &h, decay, lr);

    for (a, &b) in mem.m.iter().zip(ref_m.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // Multi-step stability test (1,000 steps)
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..1000 {
        let rand_h: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
        mem.update(&rand_h);
        ref_hebbian_update(&mut ref_m, dim, &rand_h, decay, lr);
    }

    for (a, &b) in mem.m.iter().zip(ref_m.iter()) {
        assert!((a - b).abs() < 1e-4);
    }

    // Recall test
    let mut recall_out = vec![0.0f32; dim];
    mem.recall(&h, &mut recall_out);
    let ref_recall = ref_matvec(&mem.m, dim, dim, &h);
    for (r, ref_r) in recall_out.iter().zip(ref_recall.iter()) {
        assert!((r - ref_r).abs() < 1e-4);
    }
}

#[test]
fn test_hebbian_frobenius_and_oja() {
    let dim = 8;
    let mut mem = HebbianMemory::new(dim, 1.0, 0.1);
    let h = vec![2.0f32; dim];

    for _ in 0..10 {
        mem.update(&h);
    }

    let norm = mem.frobenius_norm();
    assert!(norm > 1.0);

    mem.clip_frobenius(5.0);
    assert!((mem.frobenius_norm() - 5.0).abs() < 1e-5);

    let mut scratch = vec![0.0f32; dim];
    mem.update_oja(&h, &mut scratch);
    for &v in &mem.m {
        assert!(!v.is_nan());
        assert!(!v.is_infinite());
    }
}

#[test]
fn test_activations_and_mse() {
    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut sig = vec![0.0; 5];
    sigmoid(&x, &mut sig);
    for (&s, &v) in sig.iter().zip(x.iter()) {
        let expected = 1.0 / (1.0 + (-v).exp());
        assert!((s - expected).abs() < 1e-6);
    }

    let target = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let mut grad_mse = vec![0.0; 5];
    let loss = mse_loss_and_grad(&sig, &target, &mut grad_mse);
    assert!(loss >= 0.0);
    for (&g, (&s, &t)) in grad_mse.iter().zip(sig.iter().zip(target.iter())) {
        let expected_g = (2.0 / 5.0) * (s - t);
        assert!((g - expected_g).abs() < 1e-6);
    }
}
