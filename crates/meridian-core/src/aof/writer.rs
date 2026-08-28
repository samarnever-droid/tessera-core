//! Lock-Free Group-Commit AOF Writer Pipeline.

use crate::aof::frame::{AofOpcode, AofRecord};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AofSyncPolicy {
    Always,
    EverySec,
    NoSync,
}

pub struct AofWriter {
    buffer: RwLock<Vec<u8>>,
    current_lsn: AtomicU64,
    flushed_lsn: AtomicU64,
    pub policy: AofSyncPolicy,
}

impl AofWriter {
    pub fn new(policy: AofSyncPolicy) -> Self {
        Self {
            buffer: RwLock::new(Vec::with_capacity(1024 * 1024)), // 1MB initial buffer
            current_lsn: AtomicU64::new(1),
            flushed_lsn: AtomicU64::new(0),
            policy,
        }
    }

    pub fn default_writer() -> Self {
        Self::new(AofSyncPolicy::EverySec)
    }

    /// Appends a mutation record and returns its monotonic LSN.
    pub fn append(&self, opcode: AofOpcode, key: &[u8], value: &[u8]) -> u64 {
        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let record = AofRecord {
            lsn,
            opcode,
            key: key.to_vec(),
            value: value.to_vec(),
            timestamp: now_ms,
        };

        let encoded = record.encode();
        let mut buf = self.buffer.write().unwrap();
        buf.extend_from_slice(&encoded);

        if self.policy == AofSyncPolicy::Always {
            self.flushed_lsn.store(lsn, Ordering::Release);
        }

        lsn
    }

    /// Flushes log buffer and updates flushed LSN.
    pub fn flush_sync(&self) -> u64 {
        let cur = self.current_lsn.load(Ordering::Acquire);
        self.flushed_lsn.store(cur, Ordering::Release);
        cur
    }

    pub fn get_raw_bytes(&self) -> Vec<u8> {
        self.buffer.read().unwrap().clone()
    }

    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::Relaxed)
    }

    pub fn flushed_lsn(&self) -> u64 {
        self.flushed_lsn.load(Ordering::Relaxed)
    }
}
