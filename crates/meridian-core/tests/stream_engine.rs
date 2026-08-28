//! Comprehensive Verification Test Suite for Streaming & Pub/Sub Engine.

use meridian_core::*;
use std::time::Instant;

#[test]
fn test_stream_monotonic_ids_and_trimming() {
    let mut stream = Stream::new(5); // MAXLEN = 5

    let id1 = stream.add(vec![("action".to_string(), "login".to_string())]);
    let id2 = stream.add(vec![("action".to_string(), "view".to_string())]);
    let id3 = stream.add(vec![("action".to_string(), "buy".to_string())]);

    assert!(id2 > id1, "Stream IDs must be strictly increasing: {:?} > {:?}", id2, id1);
    assert!(id3 > id2, "Stream IDs must be strictly increasing: {:?} > {:?}", id3, id2);
    assert_eq!(stream.len(), 3);

    // Range query [id1, id2] -> exactly 2 entries
    let slice = stream.range(Some(id1), Some(id2));
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0].id, id1);
    assert_eq!(slice[1].id, id2);

    // Insert 4 more entries -> total 7 added, but bounded by MAXLEN 5
    for i in 4..=7 {
        stream.add(vec![("action".to_string(), format!("event_{}", i))]);
    }

    assert_eq!(stream.len(), 5); // Trimmed to 5!
}

#[test]
fn test_stream_scale_o1_sliding_window_benchmark() {
    let mut stream = Stream::new(10_000); // MAXLEN = 10,000
    let count = 200_000;

    let start = Instant::now();
    for i in 0..count {
        stream.add(vec![("event".to_string(), format!("data_{}", i))]);
    }
    let elapsed = start.elapsed();

    assert_eq!(stream.len(), 10_000);

    let nanos_per_op = elapsed.as_nanos() / count as u128;
    println!("200,000 stream appends completed in {:?} ({} ns/op)", elapsed, nanos_per_op);
    // With O(1) VecDeque ring buffer, latency must remain sub-microsecond (< 1,000 ns/op)
    assert!(nanos_per_op < 1_500, "Expected sub-microsecond O(1) ingestion, got {} ns/op", nanos_per_op);
}

#[test]
fn test_consumer_group_pel_ack_and_claim() {
    let mut stream = Stream::new(100);

    let id1 = stream.add(vec![("task".to_string(), "resize_img".to_string())]);
    let id2 = stream.add(vec![("task".to_string(), "send_email".to_string())]);
    let _id3 = stream.add(vec![("task".to_string(), "index_db".to_string())]);

    let mut group = ConsumerGroup::new("workers", StreamId::new(0, 0));

    // Consumer 1 reads 2 messages
    let batch1 = group.read_group("worker_1", 2, &stream);
    assert_eq!(batch1.len(), 2);
    assert_eq!(batch1[0].id, id1);
    assert_eq!(batch1[1].id, id2);
    assert_eq!(group.pending_count(), 2); // 2 in-flight in PEL

    // Consumer 1 ACKs message 1
    assert_eq!(group.ack(&[id1]), 1);
    assert_eq!(group.pending_count(), 1); // 1 remaining in PEL (id2)

    // Consumer 2 claims timed-out message 2 (with 0ms idle time threshold)
    let claimed = group.claim(&[id2], "worker_2", 0);
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0], id2);

    // Verify ownership was transferred to worker_2
    let pending_entry = group.pel.get(&id2).unwrap();
    assert_eq!(pending_entry.consumer, "worker_2");
    assert_eq!(pending_entry.delivery_count, 2);

    // Worker 2 ACKs message 2
    assert_eq!(group.ack(&[id2]), 1);
    assert_eq!(group.pending_count(), 0);
}

#[test]
fn test_hierarchical_wildcard_pubsub() {
    let bus = PubSubBus::new();

    // Subscriber 101: exact topic "finance.orders.ny"
    bus.subscribe("finance.orders.ny", 101);

    // Subscriber 202: pattern "finance.orders.*"
    bus.psubscribe("finance.orders.*", 202);

    // Subscriber 303: global wildcard "finance.*"
    bus.psubscribe("finance.*", 303);

    // Subscriber 404: non-matching topic "iot.sensors"
    bus.subscribe("iot.sensors", 404);

    // 1. Publish to "finance.orders.ny" -> matches 101, 202, 303
    let matched1 = bus.publish("finance.orders.ny", b"BUY 100 AAPL");
    assert_eq!(matched1, vec![101, 202, 303]);

    // 2. Publish to "finance.orders.london" -> matches 202, 303
    let matched2 = bus.publish("finance.orders.london", b"BUY 50 BP");
    assert_eq!(matched2, vec![202, 303]);

    // 3. Publish to "iot.sensors" -> matches only 404
    let matched3 = bus.publish("iot.sensors", b"TEMP 22.5C");
    assert_eq!(matched3, vec![404]);

    // 4. Unsubscribe 101 and publish again
    bus.unsubscribe("finance.orders.ny", 101);
    let matched4 = bus.publish("finance.orders.ny", b"SELL 10 AAPL");
    assert_eq!(matched4, vec![202, 303]);
}
