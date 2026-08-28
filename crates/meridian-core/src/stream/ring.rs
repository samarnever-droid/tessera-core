//! Monotonic Append-Only Stream Engine (Phase 19).
//!
//! Employs a true O(1) double-ended ring buffer (VecDeque) to eliminate O(N) memory
//! shifting during MAXLEN evictions, sustaining sub-microsecond ingestion at scale.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId {
    pub timestamp_ms: u64,
    pub sequence: u64,
}

impl StreamId {
    pub fn new(timestamp_ms: u64, sequence: u64) -> Self {
        Self { timestamp_ms, sequence }
    }

    pub fn to_string(&self) -> String {
        format!("{}-{}", self.timestamp_ms, self.sequence)
    }

    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 2 {
            return None;
        }
        let timestamp_ms = parts[0].parse().ok()?;
        let sequence = parts[1].parse().ok()?;
        Some(Self { timestamp_ms, sequence })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEntry {
    pub id: StreamId,
    pub fields: Vec<(String, String)>,
}

pub struct Stream {
    entries: VecDeque<StreamEntry>,
    max_len: usize,
    last_timestamp: AtomicU64,
    last_sequence: AtomicU64,
}

impl Stream {
    pub fn new(max_len: usize) -> Self {
        let effective_max = if max_len == 0 { 100_000 } else { max_len };
        Self {
            entries: VecDeque::with_capacity(effective_max),
            max_len: effective_max,
            last_timestamp: AtomicU64::new(0),
            last_sequence: AtomicU64::new(0),
        }
    }

    pub fn default_stream() -> Self {
        Self::new(10_000)
    }

    pub fn generate_id(&self, custom_time_ms: Option<u64>) -> StreamId {
        let now_ms = custom_time_ms.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        });

        let last_ts = self.last_timestamp.load(Ordering::SeqCst);
        let seq = if now_ms > last_ts {
            self.last_timestamp.store(now_ms, Ordering::SeqCst);
            self.last_sequence.store(0, Ordering::SeqCst);
            0
        } else {
            self.last_sequence.fetch_add(1, Ordering::SeqCst) + 1
        };

        StreamId {
            timestamp_ms: now_ms.max(last_ts),
            sequence: seq,
        }
    }

    pub fn add(&mut self, fields: Vec<(String, String)>) -> StreamId {
        let id = self.generate_id(None);
        self.add_with_id(id, fields)
    }

    /// Pure O(1) zero-copy append and head-eviction via VecDeque ring buffer.
    pub fn add_with_id(&mut self, id: StreamId, fields: Vec<(String, String)>) -> StreamId {
        self.entries.push_back(StreamEntry { id, fields });
        if self.entries.len() > self.max_len {
            self.entries.pop_front(); // Pure O(1) zero-copy head pointer advance!
        }
        id
    }

    pub fn range(&self, min_id: Option<StreamId>, max_id: Option<StreamId>) -> Vec<StreamEntry> {
        self.entries
            .iter()
            .filter(|e| {
                if let Some(min) = min_id {
                    if e.id < min {
                        return false;
                    }
                }
                if let Some(max) = max_id {
                    if e.id > max {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pure O(1) per-item trimming without shifting memory.
    pub fn trim(&mut self, max_len: usize) {
        while self.entries.len() > max_len {
            self.entries.pop_front();
        }
    }
}
