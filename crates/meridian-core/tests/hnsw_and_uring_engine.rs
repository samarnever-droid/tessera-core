//! Comprehensive Verification Test Suite for io_uring Kernel-Bypass and HNSW Vector Search Engine.

use meridian_core::*;

#[test]
fn test_io_uring_ring_batching_and_completions() {
    let engine = UringEngine::new(1024);

    // Submit 100 mixed SQEs
    for i in 0..100 {
        let opcode = match i % 4 {
            0 => UringOpcode::Accept,
            1 => UringOpcode::Recv,
            2 => UringOpcode::Send,
            _ => UringOpcode::SendZc,
        };

        engine.submit_sqe(Sqe {
            user_data: 1000 + i,
            opcode,
            fd: 42,
            buf_index: (i % 8) as u16,
            len: 128,
            payload: vec![0xAA; 64],
        });
    }

    assert_eq!(engine.pending_sq_count(), 100);
    assert_eq!(engine.pending_cq_count(), 0);

    // Kernel polled process batch
    let processed = engine.poll_and_process();
    assert_eq!(processed, 100);
    assert_eq!(engine.pending_sq_count(), 0);
    assert_eq!(engine.pending_cq_count(), 100);

    // Userspace reaps completions
    let cqes = engine.reap_completions(100);
    assert_eq!(cqes.len(), 100);
    assert_eq!(cqes[0].user_data, 1000);
    assert_eq!(cqes[99].user_data, 1099);
}

#[test]
fn test_io_uring_wal_zero_copy_batch_commit() {
    let engine = UringEngine::new(512);

    // Submit 50 asynchronous WAL records
    for lsn in 1..=50 {
        let wal_payload = format!("LSN:{}:INSERT INTO orders VALUES ({})", lsn, lsn * 10);
        engine.submit_wal_record(lsn, 3, wal_payload.as_bytes());
    }

    assert_eq!(engine.pending_sq_count(), 50);

    // Kernel SQPOLL background batch flush
    let processed = engine.poll_and_process();
    assert_eq!(processed, 50);
    assert_eq!(engine.pending_sq_count(), 0);
    assert_eq!(engine.max_persisted_lsn.load(std::sync::atomic::Ordering::SeqCst), 50);

    // Verify CQ completions
    let cqes = engine.reap_completions(50);
    assert_eq!(cqes.len(), 50);
    assert!(cqes[0].res > 0, "WAL payload bytes must be positive");
}

#[test]
fn test_simd_distance_calculations() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];

    // Cosine similarity
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

    // Euclidean distance squared
    assert!((euclidean_distance_sq(&a, &b) - 0.0).abs() < 1e-6);
    assert!((euclidean_distance_sq(&a, &c) - 2.0).abs() < 1e-6);
}

#[test]
fn test_hnsw_multi_layer_vector_graph_search() {
    let mut hnsw = HnswIndex::new(8, 32);

    // Insert 50 structured vectors
    for i in 1..=50 {
        let v = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (50 - i) as f32];
        hnsw.insert(i as u64, v);
    }

    assert_eq!(hnsw.count(), 50);

    // Query for vector closest to ID 25
    let query = vec![25.0, 50.0, 75.0, 25.0];
    let results = hnsw.search(&query, 3, 16);

    assert!(!results.is_empty());
    // ID 25 must be the #1 closest nearest neighbor with cosine similarity ≈ 1.0
    assert_eq!(results[0].0, 25);
    assert!((results[0].1 - 1.0).abs() < 1e-5);
}
