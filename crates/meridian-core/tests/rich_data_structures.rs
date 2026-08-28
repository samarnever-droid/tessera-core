//! Comprehensive Verification Test Suite for Rich Data Structures (Phases 13-17).

use meridian_core::*;

#[test]
fn test_hyperloglog_cardinality_and_merge() {
    let hll1 = HyperLogLog::new();
    let hll2 = HyperLogLog::new();

    // Insert 10,000 distinct items into hll1 (hashes 1..=10000)
    for i in 1..=10_000u64 {
        let hash = crate::hash::hash_key(&i.to_le_bytes());
        hll1.add(hash);
    }

    let est1 = hll1.count();
    // With m=16384, error is bounded within ~3%
    assert!(est1 >= 9_500 && est1 <= 10_500, "HLL1 estimate {} out of expected range", est1);

    // Insert overlapping + new items into hll2 (hashes 5000..=15000)
    for i in 5_000..=15_000u64 {
        let hash = crate::hash::hash_key(&i.to_le_bytes());
        hll2.add(hash);
    }

    // Merge hll2 into hll1 -> total distinct should be ~15,000
    hll1.merge(&hll2);
    let est_merged = hll1.count();
    assert!(est_merged >= 14_000 && est_merged <= 16_000, "Merged HLL estimate {} out of range", est_merged);

    // Serialization roundtrip
    let bytes = hll1.to_bytes();
    let restored_hll = HyperLogLog::from_bytes(&bytes);
    assert_eq!(restored_hll.count(), est_merged);
}

#[test]
fn test_count_min_sketch_and_cuckoo_filter() {
    let cms = CountMinSketch::default_sketch();
    let key1 = 0xDEADBEEFu64;
    let key2 = 0xCAFEBABEu64;

    for _ in 0..42 {
        cms.increment(key1);
    }
    for _ in 0..7 {
        cms.increment(key2);
    }

    assert!(cms.estimate(key1) >= 42);
    assert!(cms.estimate(key2) >= 7);
    assert_eq!(cms.estimate(0x12345678), 0);

    let mut cuckoo = CuckooFilter::new(1024);
    for i in 0..100u64 {
        assert!(cuckoo.insert(i));
    }
    for i in 0..100u64 {
        assert!(cuckoo.contains(i));
    }
    assert!(!cuckoo.contains(99999));
}

#[test]
fn test_crdt_pn_counter_and_lww_set_convergence() {
    let mut cluster_a = PnCounter::new();
    let mut cluster_b = PnCounter::new();

    // Cluster A increments 100 on region 1, decrements 20 on region 1
    cluster_a.increment(1, 100);
    cluster_a.decrement(1, 20);

    // Cluster B increments 50 on region 2, decrements 10 on region 2
    cluster_b.increment(2, 50);
    cluster_b.decrement(2, 10);

    assert_eq!(cluster_a.value(), 80);
    assert_eq!(cluster_b.value(), 40);

    // Replicate / Gossip merge A -> B and B -> A
    let mut merged = cluster_a.clone();
    merged.merge(&cluster_b);
    assert_eq!(merged.value(), 120); // (100 - 20) + (50 - 10) = 120

    // LWW-Set Test
    let mut set_a = LwwSet::new();
    let mut set_b = LwwSet::new();

    // A adds item at t=100
    set_a.add(b"item:omega".to_vec(), 100);
    // B removes item at t=105 (newer timestamp wins)
    set_b.remove(b"item:omega".to_vec(), 105);

    set_a.merge(&set_b);
    assert!(!set_a.contains(b"item:omega"));

    // A adds item again with newer timestamp t=110 -> becomes visible again!
    set_a.add(b"item:omega".to_vec(), 110);
    assert!(set_a.contains(b"item:omega"));
}

#[test]
fn test_json_tape_in_place_path_mutations() {
    let mut doc = JsonValue::Object(std::collections::BTreeMap::new());
    doc.set_path("user.id", JsonValue::Int(42));
    doc.set_path("user.name", JsonValue::Str("Johan".to_string()));
    doc.set_path("user.metrics.score", JsonValue::Float(99.5));
    doc.set_path("user.roles[0]", JsonValue::Str("ADMIN".to_string()));

    assert_eq!(doc.get_path("user.id"), Some(&JsonValue::Int(42)));
    assert_eq!(doc.get_path("user.name"), Some(&JsonValue::Str("Johan".to_string())));
    assert_eq!(doc.get_path("user.metrics.score"), Some(&JsonValue::Float(99.5)));

    // In-place mutate score to 100.0
    doc.set_path("user.metrics.score", JsonValue::Float(100.0));
    assert_eq!(doc.get_path("user.metrics.score"), Some(&JsonValue::Float(100.0)));

    let json_str = doc.to_json_string();
    assert!(json_str.contains("100"));
}

#[test]
fn test_zset_ranking_and_range_slicing() {
    let mut zset = ZSet::new();
    zset.add(100.0, b"player1".to_vec());
    zset.add(250.0, b"player2".to_vec());
    zset.add(175.0, b"player3".to_vec());
    zset.add(50.0,  b"player4".to_vec());

    assert_eq!(zset.len(), 4);
    assert_eq!(zset.score(b"player3"), Some(175.0));

    // Ranks (0-indexed ascending score: player4 (50), player1 (100), player3 (175), player2 (250))
    assert_eq!(zset.rank(b"player4"), Some(0));
    assert_eq!(zset.rank(b"player1"), Some(1));
    assert_eq!(zset.rank(b"player3"), Some(2));
    assert_eq!(zset.rank(b"player2"), Some(3));

    // Range by score [100.0, 200.0] -> player1, player3
    let score_range = zset.range_by_score(100.0, 200.0);
    assert_eq!(score_range.len(), 2);
    assert_eq!(score_range[0].member, b"player1");
    assert_eq!(score_range[1].member, b"player3");

    // Remove player1
    assert!(zset.rem(b"player1"));
    assert_eq!(zset.len(), 3);
    assert_eq!(zset.rank(b"player3"), Some(1)); // player3 moves up to rank 1
}

#[test]
fn test_quantized_vector_embeddings_and_search() {
    let mut index = VectorIndex::new();

    // Add 3 vectors in 4D space
    index.add(1, &[1.0, 0.0, 0.0, 0.0]); // Vector 1: x-axis
    index.add(2, &[0.0, 1.0, 0.0, 0.0]); // Vector 2: y-axis
    index.add(3, &[0.9, 0.1, 0.0, 0.0]); // Vector 3: near x-axis

    // Search for query vector [0.95, 0.05, 0.0, 0.0]
    let query = [0.95, 0.05, 0.0, 0.0];
    let top_results = index.search_top_k(&query, 2);

    assert_eq!(top_results.len(), 2);
    // Nearest should be vector 1 and vector 3
    assert!(top_results[0].0 == 1 || top_results[0].0 == 3);
    assert!(top_results[0].1 > 0.95); // High cosine similarity
}

#[test]
fn test_engine_delta_rich_pipeline() {
    let raw = vec![0u8; 8];

    // 1. In-place HLL Delta
    let hll_delta = DeltaOp::HllAdd { element_hash: 424242 };
    let out1 = apply_delta(&raw, &hll_delta);
    assert_eq!(out1.len(), HLL_REGISTERS);

    // 2. In-place PN-Counter Delta
    let pn_delta = DeltaOp::CrdtPnInc { cluster_id: 1, amount: 500 };
    let out2 = apply_delta(&raw, &pn_delta);
    let val = i64::from_le_bytes(out2[0..8].try_into().unwrap());
    assert_eq!(val, 500);
}
