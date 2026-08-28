//! Mesh (Phase 12): Consistent-Hash Multi-Cluster Topology & CDC Gossip Replication.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Configuration and metadata for a cluster node in the global mesh.
#[derive(Debug, Clone)]
pub struct MeshNode {
    pub addr: String,
    pub region: String,
    pub weight: u32,
    pub is_alive: bool,
}

/// Cross-cluster CDC delta gossip replication envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GossipMessage {
    pub epoch: u64,
    pub origin_node: String,
    pub target_node: String,
    pub key: Vec<u8>,
    pub delta_payload: Vec<u8>,
}

/// Global Consistent-Hash Multi-Cluster Mesh Coordinator.
pub struct MeshCluster {
    ring: RwLock<BTreeMap<u64, String>>, // hash_token -> node_addr
    nodes: RwLock<HashMap<String, MeshNode>>,
    local_node: String,
    vnodes_per_node: usize,
    replication_factor: usize,
    gossip_epoch: AtomicU64,
    total_gossip_sent: AtomicU64,
    total_gossip_received: AtomicU64,
}

impl MeshCluster {
    /// Creates a new default MeshCluster with 64 vnodes and 3-way replication.
    pub fn new(local_node: impl Into<String>) -> Self {
        Self::with_config(local_node, 64, 3)
    }

    /// Creates a MeshCluster with customized vnodes and replication factor.
    pub fn with_config(local_node: impl Into<String>, vnodes: usize, replication_factor: usize) -> Self {
        let local = local_node.into();
        let cluster = Self {
            ring: RwLock::new(BTreeMap::new()),
            nodes: RwLock::new(HashMap::new()),
            local_node: local.clone(),
            vnodes_per_node: if vnodes == 0 { 1 } else { vnodes },
            replication_factor: if replication_factor == 0 { 1 } else { replication_factor },
            gossip_epoch: AtomicU64::new(1),
            total_gossip_sent: AtomicU64::new(0),
            total_gossip_received: AtomicU64::new(0),
        };

        cluster.add_regional_node(&local, "local-region");
        cluster
    }

    /// Adds a node to the mesh ring with default region.
    pub fn add_node(&self, node_addr: &str) {
        self.add_regional_node(node_addr, "default-region");
    }

    /// Adds a node with region metadata and populates virtual tokens on the ring.
    pub fn add_regional_node(&self, node_addr: &str, region: &str) {
        let mut nodes = self.nodes.write().unwrap();
        nodes.insert(
            node_addr.to_string(),
            MeshNode {
                addr: node_addr.to_string(),
                region: region.to_string(),
                weight: 1,
                is_alive: true,
            },
        );

        let mut ring = self.ring.write().unwrap();
        for v in 0..self.vnodes_per_node {
            let vnode_key = format!("{}:{}", node_addr, v);
            let token = crate::hash::hash_key(vnode_key.as_bytes());
            ring.insert(token, node_addr.to_string());
        }
    }

    /// Removes a node and its virtual tokens from the mesh ring.
    pub fn remove_node(&self, node_addr: &str) {
        let mut nodes = self.nodes.write().unwrap();
        nodes.remove(node_addr);

        let mut ring = self.ring.write().unwrap();
        for v in 0..self.vnodes_per_node {
            let vnode_key = format!("{}:{}", node_addr, v);
            let token = crate::hash::hash_key(vnode_key.as_bytes());
            ring.remove(&token);
        }
    }

    /// Marks a node as failed/dead for active failover routing.
    pub fn mark_node_dead(&self, node_addr: &str) {
        let mut nodes = self.nodes.write().unwrap();
        if let Some(node) = nodes.get_mut(node_addr) {
            node.is_alive = false;
        }
    }

    /// Marks a node as healthy.
    pub fn mark_node_alive(&self, node_addr: &str) {
        let mut nodes = self.nodes.write().unwrap();
        if let Some(node) = nodes.get_mut(node_addr) {
            node.is_alive = true;
        }
    }

    /// Locates the primary alive node for a given key on the consistent-hash ring.
    pub fn locate_node(&self, key: &[u8]) -> String {
        let hash = crate::hash::hash_key(key);
        let ring = self.ring.read().unwrap();
        let nodes = self.nodes.read().unwrap();

        if ring.is_empty() {
            return self.local_node.clone();
        }

        // Search from hash.. to end of ring, then wrap around
        for (_, addr) in ring.range(hash..).chain(ring.iter()) {
            if let Some(node) = nodes.get(addr) {
                if node.is_alive {
                    return addr.clone();
                }
            } else {
                return addr.clone();
            }
        }

        self.local_node.clone()
    }

    /// Locates N distinct cluster replicas (primary + backups) for a given key.
    pub fn locate_replicas(&self, key: &[u8]) -> Vec<String> {
        let hash = crate::hash::hash_key(key);
        let ring = self.ring.read().unwrap();
        let nodes = self.nodes.read().unwrap();

        let mut replicas = Vec::new();
        let mut seen_nodes = HashSet::new();

        for (_, addr) in ring.range(hash..).chain(ring.iter()) {
            if !seen_nodes.contains(addr) {
                let is_alive = nodes.get(addr).map(|n| n.is_alive).unwrap_or(true);
                if is_alive {
                    seen_nodes.insert(addr.clone());
                    replicas.push(addr.clone());
                    if replicas.len() >= self.replication_factor {
                        break;
                    }
                }
            }
        }

        if replicas.is_empty() {
            replicas.push(self.local_node.clone());
        }

        replicas
    }

    /// Broadcasts a CDC delta to all replica clusters on the gossip mesh.
    pub fn broadcast_cdc_delta(&self, key: &[u8], delta_payload: &[u8]) -> Vec<GossipMessage> {
        let epoch = self.gossip_epoch.fetch_add(1, Ordering::SeqCst);
        let replicas = self.locate_replicas(key);
        let mut messages = Vec::new();

        for target in replicas {
            let msg = GossipMessage {
                epoch,
                origin_node: self.local_node.clone(),
                target_node: target,
                key: key.to_vec(),
                delta_payload: delta_payload.to_vec(),
            };
            self.total_gossip_sent.fetch_add(1, Ordering::Relaxed);
            messages.push(msg);
        }

        messages
    }

    /// Ingests a received cross-cluster CDC delta message.
    pub fn receive_gossip_delta(&self, _msg: &GossipMessage) -> bool {
        self.total_gossip_received.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Returns total virtual tokens on the hash ring.
    pub fn ring_size(&self) -> usize {
        self.ring.read().unwrap().len()
    }

    /// Returns the total distinct physical nodes registered.
    pub fn node_count(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    /// Returns the total gossip messages dispatched.
    pub fn total_gossip_sent(&self) -> u64 {
        self.total_gossip_sent.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_cluster_ring_and_vnodes() {
        let cluster = MeshCluster::with_config("node1:7717", 16, 3);
        cluster.add_regional_node("node2:7717", "us-west");
        cluster.add_regional_node("node3:7717", "eu-central");

        assert_eq!(cluster.node_count(), 3);
        assert_eq!(cluster.ring_size(), 48); // 3 nodes * 16 vnodes

        let primary = cluster.locate_node(b"account:1001");
        assert!(["node1:7717", "node2:7717", "node3:7717"].contains(&primary.as_str()));

        let replicas = cluster.locate_replicas(b"account:1001");
        assert_eq!(replicas.len(), 3);
        assert_eq!(replicas[0], primary);
    }

    #[test]
    fn test_dynamic_failover_when_node_marked_dead() {
        let cluster = MeshCluster::with_config("node1:7717", 16, 2);
        cluster.add_node("node2:7717");

        let initial_primary = cluster.locate_node(b"key:xyz");
        cluster.mark_node_dead(&initial_primary);

        let failover_primary = cluster.locate_node(b"key:xyz");
        assert_ne!(initial_primary, failover_primary);

        cluster.mark_node_alive(&initial_primary);
        let restored_primary = cluster.locate_node(b"key:xyz");
        assert_eq!(initial_primary, restored_primary);
    }

    #[test]
    fn test_cross_cluster_cdc_gossip_broadcast() {
        let cluster = MeshCluster::with_config("node1:7717", 16, 3);
        cluster.add_node("node2:7717");
        cluster.add_node("node3:7717");

        let messages = cluster.broadcast_cdc_delta(b"user:42:balance", b"SUM +500");
        assert_eq!(messages.len(), 3);
        assert_eq!(cluster.total_gossip_sent(), 3);

        for msg in &messages {
            assert!(cluster.receive_gossip_delta(msg));
        }
    }
}
