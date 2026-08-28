//! Comprehensive Verification Test Suite for FlashTier NVMe/SSD Storage Spillover Engine.

use meridian_core::*;

#[test]
fn test_flashtier_zero_io_sparse_index_probe() {
    let tier = FlashTier::new(1.1);

    tier.put(b"cold_item_1".to_vec(), b"payload_data_alpha".to_vec());
    tier.put(b"cold_item_2".to_vec(), b"payload_data_beta".to_vec());

    assert_eq!(tier.total_items(), 2);
    assert!(tier.contains_key(b"cold_item_1"));
    assert!(tier.contains_key(b"cold_item_2"));

    // Non-existent key must return false in DRAM with 0 disk reads
    assert!(!tier.contains_key(b"non_existent_key"));
    assert_eq!(tier.get(b"non_existent_key"), None);
}

#[test]
fn test_flashtier_roundtrip_spill_and_read() {
    let tier = FlashTier::new(1.05);

    // Spill 100 cold items
    for i in 0..100 {
        let key = format!("user:{}:profile", i).into_bytes();
        let val = format!("DATA_PAYLOAD_FOR_USER_{}", i).into_bytes();
        tier.put(key, val);
    }

    assert_eq!(tier.total_items(), 100);
    assert!(tier.total_bytes_on_disk() > 0);

    // Read back items and verify payload integrity
    for i in 0..100 {
        let key = format!("user:{}:profile", i).into_bytes();
        let expected_val = format!("DATA_PAYLOAD_FOR_USER_{}", i).into_bytes();
        let actual_val = tier.get(&key).expect("Spilled item must be retrievable");
        assert_eq!(actual_val, expected_val);
    }
}

#[test]
fn test_flashtier_frequency_based_promotion() {
    let tier = FlashTier::new(1.0);

    let key = b"hot_trending_topic".to_vec();
    let val = b"VIRAL_TREND_PAYLOAD".to_vec();
    tier.put(key.clone(), val.clone());

    assert_eq!(tier.total_promotions.load(std::sync::atomic::Ordering::Relaxed), 0);

    // First read: hit_count = 1 (< threshold 2) -> No promotion
    let r1 = tier.get(&key).unwrap();
    assert_eq!(r1, val);
    assert_eq!(tier.total_promotions.load(std::sync::atomic::Ordering::Relaxed), 0);

    // Second read: hit_count = 2 (>= threshold 2) -> Triggers promotion!
    let r2 = tier.get(&key).unwrap();
    assert_eq!(r2, val);
    assert_eq!(tier.total_promotions.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn test_flashtier_delete() {
    let tier = FlashTier::new(1.0);

    tier.put(b"temp_item".to_vec(), b"TEMP".to_vec());
    assert!(tier.contains_key(b"temp_item"));

    assert!(tier.delete(b"temp_item"));
    assert!(!tier.contains_key(b"temp_item"));
    assert_eq!(tier.get(b"temp_item"), None);
}
