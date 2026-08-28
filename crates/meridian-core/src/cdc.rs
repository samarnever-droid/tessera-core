//! CDC (Phase 6): Change-Data-Capture Ingestion, Watermark Tracking & Degradation Ladder.
//!
//! Processes continuous commit streams from databases (e.g. SamarDB WAL),
//! tracks LSN progression, computes lag, and governs cache degradation.

use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CdcOp {
    Insert,
    Update,
    Delete,
    Truncate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdcRecord {
    pub lsn: u64,
    pub source: String,
    pub table: String,
    pub key_id: u64,
    pub op: CdcOp,
    pub before_bytes: Option<Vec<u8>>,
    pub after_bytes: Option<Vec<u8>>,
    pub timestamp_ms: u64,
}

/// 5-Level Degradation Ladder governing cache consistency under lag or fault.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DegradationLevel {
    /// Zero lag, full deterministic exact freshness
    ExactLive = 0,
    /// Measurable lag, bounded by max(lag, ttl)
    ExactDegraded = 1,
    /// Coarse provenance invalidation
    ProvenanceOnly = 2,
    /// Best effort approximate caching
    BestEffort = 3,
    /// Cache bypassed, direct origin queries
    Bypass = 4,
}

/// Monotonic per-source LSN watermark tracker.
pub struct WatermarkTracker {
    watermarks: RwLock<HashMap<String, u64>>,
    last_update_ts: RwLock<HashMap<String, u64>>,
}

impl WatermarkTracker {
    pub fn new() -> Self {
        Self {
            watermarks: RwLock::new(HashMap::new()),
            last_update_ts: RwLock::new(HashMap::new()),
        }
    }

    pub fn advance_lsn(&self, source: &str, lsn: u64, now_ms: u64) -> bool {
        let mut wm = self.watermarks.write().unwrap();
        let mut ts = self.last_update_ts.write().unwrap();
        
        let cur = wm.entry(source.to_string()).or_insert(0);
        if lsn > *cur {
            *cur = lsn;
            ts.insert(source.to_string(), now_ms);
            true
        } else {
            false // Replay / out of order
        }
    }

    pub fn get_watermark(&self, source: &str) -> u64 {
        self.watermarks.read().unwrap().get(source).copied().unwrap_or(0)
    }

    pub fn compute_lag_ms(&self, source: &str, now_ms: u64) -> u64 {
        let ts = self.last_update_ts.read().unwrap();
        if let Some(&last_ms) = ts.get(source) {
            now_ms.saturating_sub(last_ms)
        } else {
            0
        }
    }
}

/// Origin Token Bucket (Hard Floor Rate Limiter).
pub struct OriginTokenBucket {
    pub capacity: u64,
    tokens: std::sync::atomic::AtomicU64,
    refill_rate_per_sec: f64,
    last_refill_ms: std::sync::atomic::AtomicU64,
}

impl OriginTokenBucket {
    pub fn new(capacity: usize, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity: capacity as u64,
            tokens: std::sync::atomic::AtomicU64::new(capacity as u64),
            refill_rate_per_sec,
            last_refill_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn try_acquire(&self, count: u64, now_ms: u64) -> bool {
        let last = self.last_refill_ms.load(std::sync::atomic::Ordering::Relaxed);
        if last == 0 {
            self.last_refill_ms.store(now_ms, std::sync::atomic::Ordering::Relaxed);
        } else {
            let elapsed_sec = (now_ms.saturating_sub(last) as f64) / 1000.0;
            if elapsed_sec > 0.05 {
                let add_tokens = (elapsed_sec * self.refill_rate_per_sec) as u64;
                if add_tokens > 0 {
                    let cur = self.tokens.load(std::sync::atomic::Ordering::Relaxed);
                    let new_tokens = (cur + add_tokens).min(self.capacity);
                    self.tokens.store(new_tokens, std::sync::atomic::Ordering::Relaxed);
                    self.last_refill_ms.store(now_ms, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        let cur = self.tokens.load(std::sync::atomic::Ordering::Relaxed);
        if cur >= count {
            self.tokens.fetch_sub(count, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}
