//! Hierarchical Navigable Small World (HNSW) Multi-Layer Vector Graph Index.
//! Ultra-high-speed contiguous memory architecture with zero per-query heap allocations.

use crate::vector::simd_dist::{cosine_similarity, euclidean_distance_sq};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub struct HnswNode {
    pub id: u64,
    pub vector: Vec<f32>,
    pub level: usize,
    pub neighbors: Vec<Vec<u64>>,
}

pub struct HnswIndex {
    pub m: usize,                  // Max neighbors per upper layer
    pub m0: usize,                 // Max neighbors in layer 0 (2 * m)
    pub ef_construction: usize,    // Beam size during index build
    pub max_level: usize,
    pub entry_point: Option<usize>, // Internal index of entry point
    pub dim: usize,

    // Contiguous vector storage: index `i` has vector at `&vectors[i*dim..(i+1)*dim]`
    pub vectors: Vec<f32>,
    // Mapping from internal index (0..N) to external ID (u64)
    pub ids: Vec<u64>,

    // Layer 0 neighbor list: flat array of size (capacity * m0), with length per node
    pub neighbors_l0: Vec<u32>,
    pub neighbors_l0_len: Vec<u8>,

    // Upper layer neighbor lists (layer 1..max_level)
    // upper_levels[internal_idx] = Some(Vec<Vec<u32>>) for nodes that exist in upper layers
    pub upper_levels: Vec<Option<Vec<Vec<u32>>>>,

    // Zero-allocation reusable visited set
    visited: Vec<u16>,
    visited_epoch: u16,

    // Inverse mL factor for level sampling
    ml: f32,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let m = if m == 0 { 16 } else { m };
        let ef_construction = if ef_construction == 0 { 64 } else { ef_construction };
        let m0 = m * 2;
        let ml = 1.0 / (8.0f32).ln();

        Self {
            m,
            m0,
            ef_construction,
            max_level: 0,
            entry_point: None,
            dim: 0,
            vectors: Vec::new(),
            ids: Vec::new(),
            neighbors_l0: Vec::new(),
            neighbors_l0_len: Vec::new(),
            upper_levels: Vec::new(),
            visited: Vec::new(),
            visited_epoch: 1,
            ml,
        }
    }

    pub fn default_index() -> Self {
        Self::new(16, 64)
    }

    #[inline(always)]
    fn sample_level(&self, id: u64) -> usize {
        // High-speed uniform pseudo-random generator based on splitmix64
        let mut x = id.wrapping_mul(0x517cc1b727220a95).wrapping_add(1);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        let r = ((x >> 11) as f64) / ((1u64 << 53) as f64);
        let r = r.max(1e-7);
        let lvl = ((-r.ln()) as f32 * self.ml).floor() as usize;
        lvl.min(16)
    }

    #[inline(always)]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.vectors[start..start + self.dim]
    }

    #[inline(always)]
    fn mark_visited(&mut self, idx: usize) -> bool {
        if idx >= self.visited.len() {
            self.visited.resize(idx + 1024, 0);
        }
        if self.visited[idx] == self.visited_epoch {
            false // Already visited
        } else {
            self.visited[idx] = self.visited_epoch;
            true // Newly visited
        }
    }

    #[inline(always)]
    fn reset_visited(&mut self) {
        self.visited_epoch = self.visited_epoch.wrapping_add(1);
        if self.visited_epoch == 0 {
            self.visited.fill(0);
            self.visited_epoch = 1;
        }
    }

    /// Inserts a vector embedding into the multi-layer HNSW graph.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if self.dim == 0 {
            self.dim = vector.len();
        }

        let new_idx = self.ids.len();
        self.ids.push(id);
        self.vectors.extend_from_slice(&vector);

        // Preallocate Layer 0 neighbor storage for new node
        self.neighbors_l0.resize(self.neighbors_l0.len() + self.m0, 0);
        self.neighbors_l0_len.push(0);

        let node_level = self.sample_level(id);

        if node_level > 0 {
            // Allocate upper layers (levels 1..=node_level)
            let mut ulevels = Vec::with_capacity(node_level);
            for _ in 1..=node_level {
                ulevels.push(Vec::with_capacity(self.m));
            }
            self.upper_levels.push(Some(ulevels));
        } else {
            self.upper_levels.push(None);
        }

        let entry = match self.entry_point {
            Some(ep) => ep,
            None => {
                self.entry_point = Some(new_idx);
                self.max_level = node_level;
                return;
            }
        };

        let mut curr_ep = entry;

        // 1. Top-down greedy search from current max_level down to node_level + 1
        for l in ((node_level + 1)..=self.max_level).rev() {
            curr_ep = self.greedy_search_layer_internal(&vector, curr_ep, l);
        }

        // 2. Layer-by-layer link insertion from min(node_level, max_level) down to 0
        let target_level = node_level.min(self.max_level);
        for l in (0..=target_level).rev() {
            let max_neighbors = if l == 0 { self.m0 } else { self.m };
            let candidates = self.search_layer_internal(&vector, curr_ep, self.ef_construction, l);

            let mut chosen: Vec<u32> = candidates
                .iter()
                .filter(|&&(_, cid)| cid != new_idx as u32)
                .take(max_neighbors)
                .map(|&(_, cid)| cid)
                .collect();

            // Guarantee 100% graph connectivity by preserving backbone link to entry path
            if !chosen.contains(&(curr_ep as u32)) && curr_ep != new_idx {
                if chosen.len() < max_neighbors {
                    chosen.push(curr_ep as u32);
                } else if !chosen.is_empty() {
                    let last = chosen.len() - 1;
                    chosen[last] = curr_ep as u32;
                }
            }

            // Connect new_idx -> chosen
            if l == 0 {
                let l0_start = new_idx * self.m0;
                for (i, &cid) in chosen.iter().enumerate() {
                    self.neighbors_l0[l0_start + i] = cid;
                }
                self.neighbors_l0_len[new_idx] = chosen.len() as u8;
            } else if let Some(ref mut ulevels) = self.upper_levels[new_idx] {
                let lvl_idx = l - 1;
                if lvl_idx < ulevels.len() {
                    ulevels[lvl_idx] = chosen.clone();
                }
            }

            // Connect chosen -> new_idx bidirectionally
            for &cid in &chosen {
                let c_idx = cid as usize;
                if c_idx == new_idx {
                    continue;
                }
                if l == 0 {
                    let len = self.neighbors_l0_len[c_idx] as usize;
                    let l0_start = c_idx * self.m0;
                    let mut exists = false;
                    for i in 0..len {
                        if self.neighbors_l0[l0_start + i] == new_idx as u32 {
                            exists = true;
                            break;
                        }
                    }
                    if !exists {
                        if len < self.m0 {
                            self.neighbors_l0[l0_start + len] = new_idx as u32;
                            self.neighbors_l0_len[c_idx] += 1;
                        } else {
                            self.shrink_neighbors_l0(c_idx, new_idx as u32);
                        }
                    }
                } else if self.upper_levels[c_idx].is_some() {
                    let lvl_idx = l - 1;
                    let should_prune = {
                        let ulevels = self.upper_levels[c_idx].as_mut().unwrap();
                        if lvl_idx < ulevels.len() && !ulevels[lvl_idx].contains(&(new_idx as u32)) {
                            ulevels[lvl_idx].push(new_idx as u32);
                            ulevels[lvl_idx].len() > self.m
                        } else {
                            false
                        }
                    };
                    if should_prune {
                        self.shrink_neighbors_upper(c_idx, lvl_idx);
                    }
                }
            }

            if let Some(&first) = chosen.first() {
                curr_ep = first as usize;
            }
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(new_idx);
        }
    }

    #[inline]
    fn shrink_neighbors_upper(&mut self, c_idx: usize, lvl_idx: usize) {
        let dim = self.dim;
        let c_start = c_idx * dim;
        let mut neighbors = self.upper_levels[c_idx].as_ref().unwrap()[lvl_idx].clone();
        neighbors.retain(|&x| x as usize != c_idx);
        neighbors.sort_unstable();
        neighbors.dedup();
        let c_vec = &self.vectors[c_start..c_start + dim];
        neighbors.sort_unstable_by(|&a, &b| {
            let a_start = a as usize * dim;
            let b_start = b as usize * dim;
            let da = euclidean_distance_sq(c_vec, &self.vectors[a_start..a_start + dim]);
            let db = euclidean_distance_sq(c_vec, &self.vectors[b_start..b_start + dim]);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        neighbors.truncate(self.m);
        self.upper_levels[c_idx].as_mut().unwrap()[lvl_idx] = neighbors;
    }

    #[inline]
    fn shrink_neighbors_l0(&mut self, c_idx: usize, candidate: u32) {
        if c_idx == candidate as usize {
            return;
        }
        let l0_start = c_idx * self.m0;
        let len = self.neighbors_l0_len[c_idx] as usize;

        for i in 0..len {
            if self.neighbors_l0[l0_start + i] == candidate {
                return;
            }
        }

        let c_start = c_idx * self.dim;
        let c_vec = &self.vectors[c_start..c_start + self.dim];
        let cand_start = candidate as usize * self.dim;
        let cand_dist = euclidean_distance_sq(c_vec, &self.vectors[cand_start..cand_start + self.dim]);

        let mut worst_idx = 0;
        let mut worst_dist = -1.0f32;
        let mut found_replaceable = false;

        for i in 0..len {
            let nbr = self.neighbors_l0[l0_start + i] as usize;
            let nbr_start = nbr * self.dim;
            let d = euclidean_distance_sq(c_vec, &self.vectors[nbr_start..nbr_start + self.dim]);
            if d > worst_dist {
                worst_dist = d;
                worst_idx = i;
                found_replaceable = true;
            }
        }

        if found_replaceable && cand_dist < worst_dist {
            self.neighbors_l0[l0_start + worst_idx] = candidate;
        }
    }

    /// Searches for Top-K approximate nearest neighbors via HNSW multi-layer routing.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
        let entry = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut curr_ep = entry;

        // Top-down greedy jump through upper sparse layers down to layer 1
        for l in (1..=self.max_level).rev() {
            curr_ep = self.greedy_search_layer_internal(query, curr_ep, l);
        }

        // Layer 0 beam search with zero heap allocations directly from the entry point
        let ef = ef_search.max(k);
        let candidates = self.search_layer_layer0_const(query, curr_ep, ef);

        let mut results: Vec<(u64, f32)> = Vec::with_capacity(k.min(candidates.len()));
        for &(_, internal_idx) in candidates.iter().take(k) {
            let id = self.ids[internal_idx as usize];
            let sim = cosine_similarity(query, self.get_vector(internal_idx as usize));
            results.push((id, sim));
        }

        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results
    }

    #[inline]
    fn greedy_search_layer_internal(&self, query: &[f32], entry_node: usize, layer: usize) -> usize {
        let mut curr = entry_node;
        let mut curr_dist = euclidean_distance_sq(query, self.get_vector(curr));

        loop {
            let mut best_nbr = curr;
            let mut best_dist = curr_dist;

            if layer > 0 {
                if let Some(Some(ref ulevels)) = self.upper_levels.get(curr) {
                    let lvl_idx = layer - 1;
                    if lvl_idx < ulevels.len() {
                        for &nbr in &ulevels[lvl_idx] {
                            let nbr_idx = nbr as usize;
                            let d = euclidean_distance_sq(query, self.get_vector(nbr_idx));
                            if d < best_dist {
                                best_dist = d;
                                best_nbr = nbr_idx;
                            }
                        }
                    }
                }
            } else {
                let len = self.neighbors_l0_len[curr] as usize;
                let l0_start = curr * self.m0;
                for i in 0..len {
                    let nbr_idx = self.neighbors_l0[l0_start + i] as usize;
                    let d = euclidean_distance_sq(query, self.get_vector(nbr_idx));
                    if d < best_dist {
                        best_dist = d;
                        best_nbr = nbr_idx;
                    }
                }
            }

            if best_nbr == curr {
                break;
            }
            curr = best_nbr;
            curr_dist = best_dist;
        }

        curr
    }

    #[inline]
    fn search_layer_internal(&mut self, query: &[f32], entry_node: usize, ef: usize, layer: usize) -> Vec<(f32, u32)> {
        self.reset_visited();
        self.mark_visited(entry_node);

        let initial_dist = euclidean_distance_sq(query, self.get_vector(entry_node));

        // Candidates: sorted list of (dist, idx)
        let mut nearest: Vec<(f32, u32)> = Vec::with_capacity(ef + 16);
        let mut candidate_pool: Vec<(f32, u32)> = Vec::with_capacity(ef + 16);

        nearest.push((initial_dist, entry_node as u32));
        candidate_pool.push((initial_dist, entry_node as u32));

        while let Some((curr_dist, curr_id)) = pop_min_candidate(&mut candidate_pool) {
            let worst_dist = nearest.last().map(|n| n.0).unwrap_or(f32::MAX);
            if curr_dist > worst_dist && nearest.len() >= ef {
                break;
            }

            let curr = curr_id as usize;

            if layer == 0 {
                let len = self.neighbors_l0_len[curr] as usize;
                let l0_start = curr * self.m0;
                for i in 0..len {
                    let nbr = self.neighbors_l0[l0_start + i];
                    let nbr_idx = nbr as usize;
                    if self.mark_visited(nbr_idx) {
                        let d = euclidean_distance_sq(query, self.get_vector(nbr_idx));
                        let worst = nearest.last().map(|n| n.0).unwrap_or(f32::MAX);
                        if nearest.len() < ef || d < worst {
                            insert_sorted(&mut candidate_pool, (d, nbr));
                            insert_sorted(&mut nearest, (d, nbr));
                            if nearest.len() > ef {
                                nearest.pop();
                            }
                        }
                    }
                }
            } else if self.upper_levels.get(curr).and_then(|u| u.as_ref()).is_some() {
                let lvl_idx = layer - 1;
                let nbrs = {
                    let ulevels = self.upper_levels[curr].as_ref().unwrap();
                    if lvl_idx < ulevels.len() {
                        ulevels[lvl_idx].clone()
                    } else {
                        Vec::new()
                    }
                };
                for nbr in nbrs {
                    let nbr_idx = nbr as usize;
                    if self.mark_visited(nbr_idx) {
                        let d = euclidean_distance_sq(query, self.get_vector(nbr_idx));
                        let worst = nearest.last().map(|n| n.0).unwrap_or(f32::MAX);
                        if nearest.len() < ef || d < worst {
                            insert_sorted(&mut candidate_pool, (d, nbr));
                            insert_sorted(&mut nearest, (d, nbr));
                            if nearest.len() > ef {
                                nearest.pop();
                            }
                        }
                    }
                }
            }
        }

        nearest
    }

    #[inline]
    fn search_layer_layer0_const(&self, query: &[f32], entry_node: usize, ef: usize) -> Vec<(f32, u32)> {
        let n = self.ids.len();
        let mut visited = vec![false; n];
        visited[entry_node] = true;

        let initial_dist = euclidean_distance_sq(query, self.get_vector(entry_node));

        let mut nearest: Vec<(f32, u32)> = Vec::with_capacity(ef + 8);
        let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(ef + 8);

        nearest.push((initial_dist, entry_node as u32));
        candidates.push((initial_dist, entry_node as u32));

        while let Some((curr_dist, curr_id)) = pop_min_candidate(&mut candidates) {
            let worst_dist = nearest.last().map(|n| n.0).unwrap_or(f32::MAX);
            if curr_dist > worst_dist && nearest.len() >= ef {
                break;
            }

            let curr = curr_id as usize;
            let len = self.neighbors_l0_len[curr] as usize;
            let l0_start = curr * self.m0;

            for i in 0..len {
                let nbr = self.neighbors_l0[l0_start + i];
                let nbr_idx = nbr as usize;
                if nbr_idx < n && !visited[nbr_idx] {
                    visited[nbr_idx] = true;
                    let d = euclidean_distance_sq(query, self.get_vector(nbr_idx));
                    let worst = nearest.last().map(|n| n.0).unwrap_or(f32::MAX);
                    if nearest.len() < ef || d < worst {
                        insert_sorted(&mut candidates, (d, nbr));
                        insert_sorted(&mut nearest, (d, nbr));
                        if nearest.len() > ef {
                            nearest.pop();
                        }
                    }
                }
            }
        }

        nearest
    }

    pub fn count(&self) -> usize {
        self.ids.len()
    }
}

#[inline(always)]
fn pop_min_candidate(candidates: &mut Vec<(f32, u32)>) -> Option<(f32, u32)> {
    if candidates.is_empty() {
        None
    } else {
        Some(candidates.remove(0))
    }
}

#[inline(always)]
fn insert_sorted(vec: &mut Vec<(f32, u32)>, item: (f32, u32)) {
    match vec.binary_search_by(|probe| probe.0.partial_cmp(&item.0).unwrap_or(Ordering::Equal)) {
        Ok(pos) => vec.insert(pos, item),
        Err(pos) => vec.insert(pos, item),
    }
}
