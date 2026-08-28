//! Hierarchical Wildcard Pub/Sub Router.

use std::collections::HashMap;
use std::sync::RwLock;

pub struct PubSubBus {
    exact_subscribers: RwLock<HashMap<String, Vec<u64>>>, // topic -> subscriber_ids
    pattern_subscribers: RwLock<Vec<(String, u64)>>,       // (pattern, subscriber_id)
}

impl PubSubBus {
    pub fn new() -> Self {
        Self {
            exact_subscribers: RwLock::new(HashMap::new()),
            pattern_subscribers: RwLock::new(Vec::new()),
        }
    }

    pub fn subscribe(&self, topic: &str, sub_id: u64) {
        let mut exact = self.exact_subscribers.write().unwrap();
        exact.entry(topic.to_string()).or_default().push(sub_id);
    }

    pub fn psubscribe(&self, pattern: &str, sub_id: u64) {
        let mut patterns = self.pattern_subscribers.write().unwrap();
        patterns.push((pattern.to_string(), sub_id));
    }

    pub fn unsubscribe(&self, topic: &str, sub_id: u64) {
        let mut exact = self.exact_subscribers.write().unwrap();
        if let Some(subs) = exact.get_mut(topic) {
            subs.retain(|&id| id != sub_id);
        }
    }

    /// Matches a topic against exact subscribers and wildcard pattern subscribers.
    pub fn publish(&self, topic: &str, _payload: &[u8]) -> Vec<u64> {
        let mut matched = Vec::new();

        // 1. Exact matches
        let exact = self.exact_subscribers.read().unwrap();
        if let Some(subs) = exact.get(topic) {
            matched.extend(subs);
        }

        // 2. Pattern matches
        let patterns = self.pattern_subscribers.read().unwrap();
        for (pat, sub_id) in patterns.iter() {
            if Self::matches_glob(pat, topic) {
                matched.push(*sub_id);
            }
        }

        matched.sort_unstable();
        matched.dedup();
        matched
    }

    fn matches_glob(pattern: &str, target: &str) -> bool {
        if pattern == "*" || pattern == ">" {
            return true;
        }
        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len() - 1];
            return target.starts_with(prefix);
        }
        pattern == target
    }
}

impl Default for PubSubBus {
    fn default() -> Self {
        Self::new()
    }
}
