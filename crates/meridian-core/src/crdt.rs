//! Conflict-Free Replicated Data Types (CRDTs) (Phase 14):
//! PN-Counters and LWW-Element-Sets for Zero-Coordination Multi-Cluster Mesh Convergence.

use std::collections::HashMap;

/// State-based Positive-Negative Counter for active-active multi-region clusters.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PnCounter {
    p_vector: HashMap<u32, i64>, // cluster_id -> positive increments
    n_vector: HashMap<u32, i64>, // cluster_id -> negative decrements
}

impl PnCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments positive counter for a specific cluster region.
    pub fn increment(&mut self, cluster_id: u32, amount: i64) {
        if amount > 0 {
            *self.p_vector.entry(cluster_id).or_insert(0) += amount;
        } else if amount < 0 {
            *self.n_vector.entry(cluster_id).or_insert(0) += -amount;
        }
    }

    /// Decrements counter for a specific cluster region.
    pub fn decrement(&mut self, cluster_id: u32, amount: i64) {
        if amount > 0 {
            *self.n_vector.entry(cluster_id).or_insert(0) += amount;
        } else if amount < 0 {
            *self.p_vector.entry(cluster_id).or_insert(0) += -amount;
        }
    }

    /// Computes the net global value across all cluster regions.
    pub fn value(&self) -> i64 {
        let p_sum: i64 = self.p_vector.values().sum();
        let n_sum: i64 = self.n_vector.values().sum();
        p_sum - n_sum
    }

    /// Merges another PN-Counter state using monotonic lattice max (LUB).
    pub fn merge(&mut self, other: &PnCounter) {
        for (&cluster_id, &val) in &other.p_vector {
            let entry = self.p_vector.entry(cluster_id).or_insert(0);
            if val > *entry {
                *entry = val;
            }
        }
        for (&cluster_id, &val) in &other.n_vector {
            let entry = self.n_vector.entry(cluster_id).or_insert(0);
            if val > *entry {
                *entry = val;
            }
        }
    }
}

/// Last-Write-Wins Observed-Removed Set with Lamport timestamp resolution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LwwSet {
    add_set: HashMap<Vec<u8>, u64>,    // element -> add_timestamp
    remove_set: HashMap<Vec<u8>, u64>, // element -> remove_timestamp
}

impl LwwSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an element with a Lamport timestamp.
    pub fn add(&mut self, element: Vec<u8>, timestamp: u64) {
        let entry = self.add_set.entry(element).or_insert(0);
        if timestamp > *entry {
            *entry = timestamp;
        }
    }

    /// Removes an element with a Lamport timestamp.
    pub fn remove(&mut self, element: Vec<u8>, timestamp: u64) {
        let entry = self.remove_set.entry(element).or_insert(0);
        if timestamp > *entry {
            *entry = timestamp;
        }
    }

    /// Checks membership based on Last-Write-Wins causality rule.
    pub fn contains(&self, element: &[u8]) -> bool {
        if let Some(&add_ts) = self.add_set.get(element) {
            if let Some(&rem_ts) = self.remove_set.get(element) {
                return add_ts > rem_ts;
            }
            return true;
        }
        false
    }

    /// Returns all elements currently present in the set.
    pub fn elements(&self) -> Vec<Vec<u8>> {
        self.add_set
            .keys()
            .filter(|&k| self.contains(k))
            .cloned()
            .collect()
    }

    /// Merges another LWW-Set state (idempotent, commutative, associative).
    pub fn merge(&mut self, other: &LwwSet) {
        for (elem, &ts) in &other.add_set {
            let entry = self.add_set.entry(elem.clone()).or_insert(0);
            if ts > *entry {
                *entry = ts;
            }
        }
        for (elem, &ts) in &other.remove_set {
            let entry = self.remove_set.entry(elem.clone()).or_insert(0);
            if ts > *entry {
                *entry = ts;
            }
        }
    }
}
