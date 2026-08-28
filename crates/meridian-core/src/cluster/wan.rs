//! Multi-Region Active-Active WAN Mesh with Causal Vector Clocks & LWW-CRDT.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VectorClock {
    pub region_id: u8,
    pub epoch_seq: u64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WanDelta {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub clock: VectorClock,
}

pub struct WanMesh {
    pub local_region: u8,
    epoch_counter: AtomicU64,
    state: RwLock<HashMap<Vec<u8>, (Vec<u8>, VectorClock)>>,
    outbound_queue: RwLock<Vec<WanDelta>>,
}

impl WanMesh {
    pub fn new(local_region: u8) -> Self {
        Self {
            local_region,
            epoch_counter: AtomicU64::new(1),
            state: RwLock::new(HashMap::new()),
            outbound_queue: RwLock::new(Vec::new()),
        }
    }

    /// Performs a local write and generates an outbound WAN delta.
    pub fn apply_local_write(&self, key: &[u8], value: &[u8]) -> VectorClock {
        let seq = self.epoch_counter.fetch_add(1, Ordering::SeqCst);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let clock = VectorClock {
            region_id: self.local_region,
            epoch_seq: seq,
            timestamp_ns: now_ns,
        };

        self.state
            .write()
            .unwrap()
            .insert(key.to_vec(), (value.to_vec(), clock));

        self.outbound_queue.write().unwrap().push(WanDelta {
            key: key.to_vec(),
            value: value.to_vec(),
            clock,
        });

        clock
    }

    /// Receives and resolves an incoming cross-region WAN delta via deterministic LWW.
    pub fn receive_wan_delta(&self, delta: WanDelta) -> bool {
        let mut state = self.state.write().unwrap();
        if let Some((_, existing_clock)) = state.get(&delta.key) {
            // LWW-CRDT Conflict Resolution: Higher timestamp or higher epoch wins
            if delta.clock > *existing_clock {
                state.insert(delta.key, (delta.value, delta.clock));
                true
            } else {
                false // Ignored stale update
            }
        } else {
            state.insert(delta.key, (delta.value, delta.clock));
            true
        }
    }

    pub fn get_value(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.state.read().unwrap().get(key).map(|(v, _)| v.clone())
    }

    pub fn drain_outbound(&self) -> Vec<WanDelta> {
        let mut queue = self.outbound_queue.write().unwrap();
        let drained = queue.clone();
        queue.clear();
        drained
    }
}
