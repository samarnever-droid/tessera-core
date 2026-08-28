//! Concurrency stress for the seqlock read path: readers must never observe a
//! torn value while writers churn the same key, and per-thread key spaces
//! must stay consistent under concurrent load.

use meridian_core::{Engine, EngineOptions};
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

#[test]
fn seqlock_stress_no_torn_reads() {
    let e = Arc::new(Engine::new(EngineOptions {
        total_entries: 1 << 14,
        ..Default::default()
    }));
    let a = vec![b'a'; 64];
    let b = vec![b'b'; 64];
    // Seed the key first: after this point a read of a live key must never
    // miss — the bounded fallback guarantees it. (Regression net for the
    // torn-read-skip bug: skipping to the next probe bucket on a torn match
    // read instead of retrying produced exactly that false miss.)
    e.set(b"hot", &a);
    let mut handles = Vec::new();

    for w in 0..2u64 {
        let e = e.clone();
        let (a, b) = (a.clone(), b.clone());
        handles.push(thread::spawn(move || {
            for i in 0..20_000u64 {
                let v: &[u8] = if (i + w) % 2 == 0 { &a } else { &b };
                e.set(b"hot", v);
            }
        }));
    }
    for _ in 0..4 {
        let e = e.clone();
        handles.push(thread::spawn(move || {
            let mut seen: HashSet<Vec<u8>> = HashSet::new();
            for _ in 0..100_000usize {
                let v = e.get(b"hot").expect("false miss on a live key");
                assert_eq!(v.len(), 64);
                assert!(v.iter().all(|&c| c == v[0]), "torn value: {:?}", &v[..8]);
                seen.insert(v);
            }
            assert!(seen.len() <= 2, "observed more than the two written values");
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn disjoint_keys_stay_consistent() {
    let e = Arc::new(Engine::new(EngineOptions {
        total_entries: 1 << 16,
        ..Default::default()
    }));
    let mut handles = Vec::new();
    for t in 0..8u64 {
        let e = e.clone();
        handles.push(thread::spawn(move || {
            for i in 0..500u64 {
                let key = format!("t{t}:k{i}");
                let val = (t * 1000 + i).to_le_bytes();
                e.set(key.as_bytes(), &val);
            }
            for i in 0..500u64 {
                let key = format!("t{t}:k{i}");
                let expect = (t * 1000 + i).to_le_bytes();
                assert_eq!(e.get(key.as_bytes()), Some(expect.to_vec()), "key {key}");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(e.item_count(), 8 * 500);
}
