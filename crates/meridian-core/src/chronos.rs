//! CHRONOS (Phase 8): Snapshot Isolation & Cross-Entry Consistency.
//!
//! Chains commit-LSN-tagged versions on the vers[] plane, allowing readers
//! to pin a watermark W and read mutually consistent states across multiple keys.

use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionRecord {
    pub valid_from_lsn: u64,
    pub payload: Vec<u8>,
}

/// CHRONOS Version Store & Multi-Version Snapshot Plane.
pub struct ChronosStore {
    versions: RwLock<HashMap<Vec<u8>, Vec<VersionRecord>>>,
    active_snapshots: RwLock<HashMap<u64, u64>>, // snapshot_id -> pinned_watermark_lsn
    next_snapshot_id: std::sync::atomic::AtomicU64,
}

impl ChronosStore {
    pub fn new() -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            active_snapshots: RwLock::new(HashMap::new()),
            next_snapshot_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Append a new version for a key stamped with the commit LSN.
    pub fn append_version(&self, key: &[u8], lsn: u64, payload: Vec<u8>) {
        let mut map = self.versions.write().unwrap();
        let list = map.entry(key.to_vec()).or_default();
        list.push(VersionRecord {
            valid_from_lsn: lsn,
            payload,
        });
        // Sort descending by LSN for fast reverse lookup
        list.sort_by(|a, b| b.valid_from_lsn.cmp(&a.valid_from_lsn));
    }

    /// Open a snapshot pinned at a specific watermark LSN.
    pub fn open_snapshot(&self, watermark_lsn: u64) -> u64 {
        let sid = self.next_snapshot_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.active_snapshots.write().unwrap().insert(sid, watermark_lsn);
        sid
    }

    /// Read a key through an open snapshot at its pinned watermark.
    pub fn read_snapshot(&self, snapshot_id: u64, key: &[u8]) -> Option<Vec<u8>> {
        let wm = {
            let snaps = self.active_snapshots.read().unwrap();
            *snaps.get(&snapshot_id)?
        };

        let map = self.versions.read().unwrap();
        if let Some(list) = map.get(key) {
            // Find newest version where valid_from_lsn <= wm
            for v in list {
                if v.valid_from_lsn <= wm {
                    return Some(v.payload.clone());
                }
            }
        }
        None
    }

    /// Close and release a snapshot.
    pub fn close_snapshot(&self, snapshot_id: u64) {
        self.active_snapshots.write().unwrap().remove(&snapshot_id);
    }
}
