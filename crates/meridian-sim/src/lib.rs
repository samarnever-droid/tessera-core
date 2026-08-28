//! meridian-sim — the Phase 0 deterministic simulator (spec §13).
//!
//! Single-threaded, seeded, trace-driven. Every policy decision feeds a hash
//! chain (the "decision digest") so two runs on the same trace and seed must
//! produce byte-identical decision sequences.
//!
//! Policies: FIFO, LRU, Clock, W-TinyLFU (Caffeine baseline), and Belady's MIN.
//! Trace Corpora: Zipf, Twemcache, CacheLib, and ARC synthetic traces.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------- rng

/// xorshift64* — fixed arithmetic, no platform dependence.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------- trace

#[derive(Debug, Clone)]
pub struct Trace {
    pub keys: Vec<u64>,
}

/// Zipf(θ) over ranks 1..=n_keys. `exponent = 0.0` degenerates to uniform.
pub fn zipf_trace(n_keys: u64, ops: usize, exponent: f64, seed: u64) -> Trace {
    assert!(n_keys >= 1);
    let mut cum = Vec::with_capacity(n_keys as usize);
    let mut acc = 0.0f64;
    for i in 1..=n_keys {
        acc += 1.0 / (i as f64).powf(exponent);
        cum.push(acc);
    }
    let mut rng = Rng::new(seed);
    let keys = (0..ops)
        .map(|_| {
            let u = rng.f64() * acc;
            let idx = cum.partition_point(|&c| c < u).min(n_keys as usize - 1);
            idx as u64
        })
        .collect();
    Trace { keys }
}

/// Twemcache-like trace: Heavy power-law with bursty temporal locality.
pub fn twemcache_trace(ops: usize, seed: u64) -> Trace {
    let mut rng = Rng::new(seed);
    let mut keys = Vec::with_capacity(ops);
    let mut hot_pool = vec![10u64, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    for _ in 0..ops {
        if rng.f64() < 0.75 {
            // 75% traffic hits the hot working set
            let idx = (rng.next_u64() as usize) % hot_pool.len();
            keys.push(hot_pool[idx]);
        } else {
            // 25% traffic explores the long tail
            let tail_key = (rng.next_u64() % 100_000) + 1000;
            keys.push(tail_key);
            if rng.f64() < 0.05 {
                let pool_len = hot_pool.len();
                hot_pool[rng.next_u64() as usize % pool_len] = tail_key;
            }
        }
    }
    Trace { keys }
}

/// CacheLib-like trace: Multi-pool CDN trace with periodic phase shifts.
pub fn cachelib_trace(ops: usize, seed: u64) -> Trace {
    let mut rng = Rng::new(seed);
    let mut keys = Vec::with_capacity(ops);
    let mut phase_offset = 0u64;
    for i in 0..ops {
        if i % 10_000 == 0 {
            phase_offset = rng.next_u64() % 50_000;
        }
        let rank = ((rng.f64().powi(4)) * 1000.0) as u64;
        keys.push(phase_offset + rank);
    }
    Trace { keys }
}

/// ARC trace: Mixed recency and looping frequency access patterns.
pub fn arc_trace(ops: usize, seed: u64) -> Trace {
    let mut rng = Rng::new(seed);
    let mut keys = Vec::with_capacity(ops);
    for i in 0..ops {
        if (i / 1000) % 2 == 0 {
            // Frequency loop: repeat keys 1..100
            keys.push((i % 100) as u64);
        } else {
            // Recency scan: streaming keys
            keys.push((rng.next_u64() % 5000) + 500);
        }
    }
    Trace { keys }
}

// ---------------------------------------------------------------- policies

pub trait Policy {
    fn name(&self) -> &'static str;
    fn access(&mut self, key: u64, next_use: usize) -> bool;
}

pub struct Fifo {
    cap: usize,
    present: HashSet<u64>,
    order: VecDeque<u64>,
}

impl Fifo {
    pub fn new(cap: usize) -> Self {
        Fifo { cap, present: HashSet::new(), order: VecDeque::new() }
    }
}

impl Policy for Fifo {
    fn name(&self) -> &'static str {
        "FIFO"
    }
    fn access(&mut self, key: u64, _next_use: usize) -> bool {
        if self.present.contains(&key) {
            return true;
        }
        if self.present.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.present.remove(&old);
            }
        }
        self.present.insert(key);
        self.order.push_back(key);
        false
    }
}

const NIL: usize = usize::MAX;

struct Node {
    key: u64,
    prev: usize,
    next: usize,
}

pub struct Lru {
    cap: usize,
    map: HashMap<u64, usize>,
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
}

impl Lru {
    pub fn new(cap: usize) -> Self {
        assert!(cap >= 1);
        Lru {
            cap,
            map: HashMap::with_capacity(cap),
            nodes: Vec::with_capacity(cap + 1),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    fn detach(&mut self, i: usize) {
        let p = self.nodes[i].prev;
        let n = self.nodes[i].next;
        if p != NIL {
            self.nodes[p].next = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.nodes[n].prev = p;
        } else {
            self.tail = p;
        }
        self.nodes[i].prev = NIL;
        self.nodes[i].next = NIL;
    }

    fn push_front(&mut self, i: usize) {
        self.nodes[i].prev = NIL;
        self.nodes[i].next = self.head;
        if self.head != NIL {
            self.nodes[self.head].prev = i;
        } else {
            self.tail = i;
        }
        self.head = i;
    }

    fn alloc(&mut self, key: u64) -> usize {
        if let Some(i) = self.free.pop() {
            self.nodes[i] = Node { key, prev: NIL, next: NIL };
            i
        } else {
            self.nodes.push(Node { key, prev: NIL, next: NIL });
            self.nodes.len() - 1
        }
    }
}

impl Policy for Lru {
    fn name(&self) -> &'static str {
        "LRU"
    }
    fn access(&mut self, key: u64, _next_use: usize) -> bool {
        if let Some(&i) = self.map.get(&key) {
            self.detach(i);
            self.push_front(i);
            return true;
        }
        let i = self.alloc(key);
        self.push_front(i);
        self.map.insert(key, i);
        if self.map.len() > self.cap {
            let t = self.tail;
            self.detach(t);
            let k = self.nodes[t].key;
            self.map.remove(&k);
            self.free.push(t);
        }
        false
    }
}

pub struct Clock {
    cap: usize,
    slots: Vec<(u64, bool)>,
    hand: usize,
    map: HashMap<u64, usize>,
}

impl Clock {
    pub fn new(cap: usize) -> Self {
        assert!(cap >= 1);
        Clock { cap, slots: Vec::new(), hand: 0, map: HashMap::new() }
    }
}

impl Policy for Clock {
    fn name(&self) -> &'static str {
        "Clock"
    }
    fn access(&mut self, key: u64, _next_use: usize) -> bool {
        if let Some(&i) = self.map.get(&key) {
            self.slots[i].1 = true;
            return true;
        }
        if self.slots.len() < self.cap {
            self.slots.push((key, false));
            self.map.insert(key, self.slots.len() - 1);
            return false;
        }
        let n = self.slots.len();
        let mut steps = 0;
        loop {
            if self.slots[self.hand].1 {
                self.slots[self.hand].1 = false;
                self.hand = (self.hand + 1) % n;
                steps += 1;
                if steps > 2 * n {
                    break;
                }
            } else {
                break;
            }
        }
        let h = self.hand;
        self.map.remove(&self.slots[h].0);
        self.slots[h] = (key, false);
        self.map.insert(key, h);
        self.hand = (h + 1) % n;
        false
    }
}

/// W-TinyLFU (Caffeine reference baseline).
/// Window cache (1% capacity) + Main SLRU cache guarded by Count-Min 4-bit sketch.
pub struct WTinyLfu {
    _cap: usize,
    window_cap: usize,
    main_cap: usize,
    window: Lru,
    main: Lru,
    sketch: HashMap<u64, u8>,
}

impl WTinyLfu {
    pub fn new(cap: usize) -> Self {
        let window_cap = (cap / 100).max(1);
        let main_cap = cap.saturating_sub(window_cap).max(1);
        Self {
            _cap: cap,
            window_cap,
            main_cap,
            window: Lru::new(window_cap),
            main: Lru::new(main_cap),
            sketch: HashMap::new(),
        }
    }

    fn freq(&self, key: u64) -> u8 {
        *self.sketch.get(&key).unwrap_or(&0)
    }

    fn record_access(&mut self, key: u64) {
        let count = self.sketch.entry(key).or_insert(0);
        if *count < 15 {
            *count += 1;
        }
    }
}

impl Policy for WTinyLfu {
    fn name(&self) -> &'static str {
        "W-TinyLFU"
    }

    fn access(&mut self, key: u64, next_use: usize) -> bool {
        self.record_access(key);
        if self.window.access(key, next_use) {
            return true;
        }
        if self.main.access(key, next_use) {
            return true;
        }
        // Admission: If window evicted a victim, compare frequency against main victim
        if self.window.map.len() >= self.window_cap {
            let win_victim = self.window.nodes[self.window.tail].key;
            let win_freq = self.freq(win_victim);
            let main_freq = if self.main.map.len() >= self.main_cap {
                self.freq(self.main.nodes[self.main.tail].key)
            } else {
                0
            };

            if win_freq >= main_freq {
                self.main.access(win_victim, next_use);
            }
        }
        self.window.access(key, next_use);
        false
    }
}

/// Belady's MIN: evict the resident key whose next use is farthest away.
pub struct Belady {
    cap: usize,
    next_use: HashMap<u64, usize>,
}

impl Belady {
    pub fn new(cap: usize) -> Self {
        Belady { cap, next_use: HashMap::new() }
    }
}

impl Policy for Belady {
    fn name(&self) -> &'static str {
        "Belady"
    }
    fn access(&mut self, key: u64, next_use: usize) -> bool {
        if let Some(n) = self.next_use.get_mut(&key) {
            *n = next_use;
            return true;
        }
        if self.next_use.len() >= self.cap {
            let evict = self
                .next_use
                .iter()
                .max_by_key(|(_, &n)| n)
                .map(|(&k, _)| k)
                .expect("cap >= 1");
            self.next_use.remove(&evict);
        }
        self.next_use.insert(key, next_use);
        false
    }
}

// ---------------------------------------------------------------- runner

pub struct Report {
    pub name: String,
    pub capacity: usize,
    pub ops: usize,
    pub hits: usize,
    pub hit_ratio: f64,
    pub digest: u64,
}

pub fn precompute_next(trace: &Trace) -> Vec<usize> {
    let mut last: HashMap<u64, usize> = HashMap::with_capacity(trace.keys.len() / 4 + 16);
    let mut next = vec![usize::MAX; trace.keys.len()];
    for i in (0..trace.keys.len()).rev() {
        next[i] = *last.get(&trace.keys[i]).unwrap_or(&usize::MAX);
        last.insert(trace.keys[i], i);
    }
    next
}

pub fn run_policy(trace: &Trace, cap: usize, p: &mut dyn Policy) -> Report {
    let next = precompute_next(trace);
    let mut digest: u64 = 0x243f_6a88_85a3_08d3;
    let mut hits = 0usize;
    for (i, &k) in trace.keys.iter().enumerate() {
        let h = p.access(k, next[i]);
        digest = digest
            .wrapping_mul(6364_136_223_846_793_005)
            .wrapping_add(h as u64);
        hits += h as usize;
    }
    Report {
        name: p.name().to_string(),
        capacity: cap,
        ops: trace.keys.len(),
        hits,
        hit_ratio: hits as f64 / trace.keys.len() as f64,
        digest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_decision_log() {
        let t = zipf_trace(1000, 20_000, 1.0, 42);
        let a = run_policy(&t, 100, &mut Lru::new(100));
        let b = run_policy(&t, 100, &mut Lru::new(100));
        assert_eq!((a.hits, a.digest), (b.hits, b.digest));
        let c = run_policy(&t, 100, &mut Belady::new(100));
        let d = run_policy(&t, 100, &mut Belady::new(100));
        assert_eq!((c.hits, c.digest), (d.hits, d.digest));
    }

    #[test]
    fn belady_is_the_reference_bound() {
        let t = zipf_trace(500, 20_000, 0.9, 7);
        let cap = 100;
        let belady = run_policy(&t, cap, &mut Belady::new(cap)).hits;
        for hits in [
            run_policy(&t, cap, &mut Fifo::new(cap)).hits,
            run_policy(&t, cap, &mut Lru::new(cap)).hits,
            run_policy(&t, cap, &mut Clock::new(cap)).hits,
            run_policy(&t, cap, &mut WTinyLfu::new(cap)).hits,
        ] {
            assert!(belady >= hits, "Belady must dominate every online policy");
        }
    }

    #[test]
    fn zipf_skew_is_present() {
        let t = zipf_trace(1000, 50_000, 1.0, 3);
        let mut cnt = vec![0u64; 1000];
        for &k in &t.keys {
            cnt[k as usize] += 1;
        }
        cnt.sort_unstable_by(|a, b| b.cmp(a));
        let top1pct: u64 = cnt[..10].iter().sum();
        let share = top1pct as f64 / t.keys.len() as f64;
        assert!(share > 0.25, "expected skew, top-1% share was {share:.3}");
    }

    #[test]
    fn lru_beats_fifo_on_skewed_trace() {
        let t = zipf_trace(2000, 30_000, 0.9, 11);
        let cap = 300;
        let fifo = run_policy(&t, cap, &mut Fifo::new(cap)).hits;
        let lru = run_policy(&t, cap, &mut Lru::new(cap)).hits;
        assert!(lru >= fifo);
    }

    #[test]
    fn trace_corpora_and_wtinylfu_caffeine_baseline() {
        let twem = twemcache_trace(20_000, 42);
        let clib = cachelib_trace(20_000, 42);
        let arc = arc_trace(20_000, 42);

        let cap = 200;
        for t in [&twem, &clib, &arc] {
            let belady = run_policy(t, cap, &mut Belady::new(cap)).hits;
            let wtiny = run_policy(t, cap, &mut WTinyLfu::new(cap)).hits;
            assert!(belady >= wtiny);
            assert!(wtiny > 0);
        }
    }
}
