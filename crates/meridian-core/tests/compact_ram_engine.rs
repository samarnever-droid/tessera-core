//! Comprehensive Verification Test Suite for Zero-Downgrade Compact RAM Engine.

use meridian_core::*;

#[test]
fn test_small_string_optimization_sso() {
    // 1. String <= 15 bytes -> Inlined with 0 heap allocation
    let short_key = b"user:1001";
    let sso_inline = CompactBytes::new(short_key);
    assert!(sso_inline.is_inline());
    assert_eq!(sso_inline.len(), 9);
    assert_eq!(sso_inline.as_slice(), short_key);

    // 2. String exactly 15 bytes -> Inlined
    let fifteen_bytes = b"123456789012345";
    let sso_fifteen = CompactBytes::new(fifteen_bytes);
    assert!(sso_fifteen.is_inline());
    assert_eq!(sso_fifteen.len(), 15);

    // 3. String > 15 bytes -> Spills to shared Heap slice
    let long_key = b"session:auth:token:9876543210:enterprise";
    let sso_heap = CompactBytes::new(long_key);
    assert!(!sso_heap.is_inline());
    assert_eq!(sso_heap.len(), long_key.len());
    assert_eq!(sso_heap.as_slice(), long_key);
}

#[test]
fn test_48bit_tagged_pointer_bit_stealing() {
    let dummy_val = 42u64;
    let ptr = &dummy_val as *const u64;
    let tag = 0xABCDu16;

    // Pack pointer and 16-bit metadata tag
    let packed = TaggedPtr::pack(ptr, tag);
    assert_ne!(packed, 0);

    // Unpack pointer and tag
    let unpacked_ptr = TaggedPtr::unpack_ptr::<u64>(packed);
    let unpacked_tag = TaggedPtr::unpack_tag(packed);

    assert_eq!(unpacked_ptr, ptr);
    assert_eq!(unpacked_tag, tag);
    unsafe {
        assert_eq!(*unpacked_ptr, 42);
    }
}

#[test]
fn test_vector_8bit_scalar_quantization_sq8() {
    let raw_v = vec![0.1f32, -0.5, 0.8, -0.2, 0.95, 0.0];
    let qv = QuantizedVector::from_f32(101, &raw_v);

    assert_eq!(qv.id, 101);
    // Data is stored as 8-bit signed integers (75% smaller than f32)
    assert_eq!(qv.data.len(), raw_v.len());

    let query_norm = raw_v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let sim = qv.cosine_similarity(&raw_v, query_norm);

    // High cosine similarity precision (> 0.999 fidelity)
    assert!((sim - 1.0).abs() < 0.005);
}

#[test]
fn test_adaptive_in_memory_value_compression() {
    // 1. Data under threshold (< 128 bytes) -> Uncompressed
    let short_payload = b"small json snippet";
    let (is_comp1, comp1) = compress_value(short_payload);
    assert!(!is_comp1);
    assert_eq!(comp1, short_payload);

    // 2. Redundant / structured payload >= 128 bytes -> Compressed
    let mut large_payload = Vec::new();
    for _ in 0..30 {
        large_payload.extend_from_slice(b"{\"status\":\"active\",\"count\":00000000}");
    }
    assert!(large_payload.len() >= 128);

    let (is_comp2, comp2) = compress_value(&large_payload);
    assert!(is_comp2);
    assert!(comp2.len() < large_payload.len()); // Achieved memory reduction!

    // Decompress and verify byte-for-byte exact equality
    let decomp = decompress_value(&comp2, is_comp2);
    assert_eq!(decomp, large_payload);
}
