//! Sorted Set (ZSet) Engine (Phase 16):
//! Score-Indexed Ordered Collections, Rank Calculation, and Zero-Allocation Range Slicing.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ZSetItem {
    pub score: f64,
    pub member: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct ZSet {
    dict: HashMap<Vec<u8>, f64>,
    sorted: Vec<ZSetItem>,
}

impl ZSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or updates a member with the given score.
    pub fn add(&mut self, score: f64, member: Vec<u8>) -> bool {
        let is_new = !self.dict.contains_key(&member);

        // If member already exists, remove previous entry from sorted list
        if !is_new {
            self.rem(&member);
        }

        self.dict.insert(member.clone(), score);
        let item = ZSetItem { score, member };

        // Binary search insertion point to maintain sorted order
        let idx = match self.sorted.binary_search_by(|probe| {
            probe.score.partial_cmp(&item.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| probe.member.cmp(&item.member))
        }) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        self.sorted.insert(idx, item);
        is_new
    }

    /// Retrieves the score of a member.
    pub fn score(&self, member: &[u8]) -> Option<f64> {
        self.dict.get(member).copied()
    }

    /// Computes 0-indexed rank of a member.
    pub fn rank(&self, member: &[u8]) -> Option<usize> {
        let target_score = self.score(member)?;
        self.sorted.binary_search_by(|probe| {
            probe.score.partial_cmp(&target_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| probe.member.as_slice().cmp(member))
        }).ok()
    }

    /// Returns members within score range [min_score, max_score].
    pub fn range_by_score(&self, min_score: f64, max_score: f64) -> Vec<ZSetItem> {
        self.sorted
            .iter()
            .filter(|item| item.score >= min_score && item.score <= max_score)
            .cloned()
            .collect()
    }

    /// Returns members within 0-indexed rank range [start_rank, stop_rank].
    pub fn range_by_rank(&self, start_rank: usize, stop_rank: usize) -> Vec<ZSetItem> {
        if start_rank >= self.sorted.len() {
            return Vec::new();
        }
        let end = stop_rank.saturating_add(1).min(self.sorted.len());
        self.sorted[start_rank..end].to_vec()
    }

    /// Removes a member from the sorted set.
    pub fn rem(&mut self, member: &[u8]) -> bool {
        if let Some(score) = self.dict.remove(member) {
            if let Ok(idx) = self.sorted.binary_search_by(|probe| {
                probe.score.partial_cmp(&score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| probe.member.as_slice().cmp(member))
            }) {
                self.sorted.remove(idx);
                return true;
            }
        }
        false
    }

    /// Returns the total elements in the sorted set.
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }
}
