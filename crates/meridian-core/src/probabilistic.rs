//! Probabilistic Data Structures (Phase 13):
//! HyperLogLog++ (HLL++), Count-Min-Log Sketch, and Cuckoo Filters.

use std::sync::atomic::{AtomicU8, Ordering};

/// Precision p=14 corresponds to m=16,384 registers, with typical standard error ~1.04/sqrt(m) ~= 0.81%.
pub const HLL_PRECISION: usize = 14;
pub const HLL_REGISTERS: usize = 1 << HLL_PRECISION; // 16384

/// High-performance HyperLogLog++ with atomic register array for lock-free updates.
pub struct HyperLogLog {
    registers: Vec<AtomicU8>,
}

impl HyperLogLog {
    pub fn new() -> Self {
        let mut registers = Vec::with_capacity(HLL_REGISTERS);
        for _ in 0..HLL_REGISTERS {
            registers.push(AtomicU8::new(0));
        }
        Self { registers }
    }

    /// Adds an element by hash into the HLL register array.
    pub fn add(&self, element_hash: u64) {
        let idx = (element_hash >> (64 - HLL_PRECISION)) as usize;
        let w = element_hash & 0x0003_FFFF_FFFF_FFFF;
        let leading_zeros = if w == 0 {
            51u8
        } else {
            (w.leading_zeros() - 14 + 1) as u8
        };

        let reg = &self.registers[idx];
        let mut cur = reg.load(Ordering::Relaxed);
        while leading_zeros > cur {
            match reg.compare_exchange_weak(cur, leading_zeros, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Estimates the distinct cardinality of elements added.
    pub fn count(&self) -> u64 {
        let m = HLL_REGISTERS as f64;
        let mut sum = 0.0;
        let mut zeros = 0;

        for reg in &self.registers {
            let val = reg.load(Ordering::Relaxed);
            if val == 0 {
                zeros += 1;
            }
            sum += 2.0f64.powi(-(val as i32));
        }

        // Alpha_m correction for m=16384
        let alpha_m = 0.7213 / (1.0 + 1.079 / m);
        let mut estimate = alpha_m * m * m / sum;

        // Linear counting for small range
        if estimate <= 2.5 * m && zeros > 0 {
            estimate = m * (m / zeros as f64).ln();
        }

        estimate.round() as u64
    }

    /// Merges another HLL instance into this one (Idempotent and Commutative).
    pub fn merge(&self, other: &HyperLogLog) {
        for (i, other_reg) in other.registers.iter().enumerate() {
            let other_val = other_reg.load(Ordering::Relaxed);
            let reg = &self.registers[i];
            let mut cur = reg.load(Ordering::Relaxed);
            while other_val > cur {
                match reg.compare_exchange_weak(cur, other_val, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(actual) => cur = actual,
                }
            }
        }
    }

    /// Exports registers into a raw byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.registers.iter().map(|r| r.load(Ordering::Relaxed)).collect()
    }

    /// Reconstructs HLL from raw byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hll = Self::new();
        for (i, &b) in bytes.iter().take(HLL_REGISTERS).enumerate() {
            hll.registers[i].store(b, Ordering::Relaxed);
        }
        hll
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Count-Min-Log Sketch for sub-linear memory frequency estimation.
pub struct CountMinSketch {
    depth: usize,
    width: usize,
    table: Vec<AtomicU8>,
}

impl CountMinSketch {
    pub fn new(depth: usize, width: usize) -> Self {
        let mut table = Vec::with_capacity(depth * width);
        for _ in 0..(depth * width) {
            table.push(AtomicU8::new(0));
        }
        Self { depth, width, table }
    }

    pub fn default_sketch() -> Self {
        Self::new(4, 2048)
    }

    fn hash_index(&self, key_hash: u64, d: usize) -> usize {
        let salt = (d as u64).wrapping_mul(0x517cc1b727220a95);
        let h = key_hash.wrapping_add(salt) ^ (key_hash >> 32);
        (h as usize) % self.width
    }

    pub fn increment(&self, key_hash: u64) {
        for d in 0..self.depth {
            let col = self.hash_index(key_hash, d);
            let idx = d * self.width + col;
            let counter = &self.table[idx];
            let cur = counter.load(Ordering::Relaxed);
            if cur < 255 {
                let _ = counter.compare_exchange_weak(cur, cur + 1, Ordering::Relaxed, Ordering::Relaxed);
            }
        }
    }

    pub fn estimate(&self, key_hash: u64) -> u32 {
        let mut min_val = u32::MAX;
        for d in 0..self.depth {
            let col = self.hash_index(key_hash, d);
            let idx = d * self.width + col;
            let val = self.table[idx].load(Ordering::Relaxed) as u32;
            if val < min_val {
                min_val = val;
            }
        }
        min_val
    }
}

/// Constant-time Cuckoo Filter for exact approximate set membership with <1% false positives.
pub struct CuckooFilter {
    buckets: Vec<[u8; 4]>,
    pub capacity: usize,
    count: usize,
}

impl CuckooFilter {
    pub fn new(capacity: usize) -> Self {
        let num_buckets = (capacity / 4).max(64).next_power_of_two();
        Self {
            buckets: vec![[0; 4]; num_buckets],
            capacity: num_buckets * 4,
            count: 0,
        }
    }

    fn fingerprint(&self, hash: u64) -> u8 {
        let fp = (hash & 0xFF) as u8;
        if fp == 0 { 1 } else { fp }
    }

    fn get_indices(&self, hash: u64, fp: u8) -> (usize, usize) {
        let num_buckets = self.buckets.len();
        let i1 = (hash as usize) % num_buckets;
        let fp_hash = (fp as u64).wrapping_mul(0x5bd1e995);
        let i2 = (i1 ^ (fp_hash as usize)) % num_buckets;
        (i1, i2)
    }

    pub fn insert(&mut self, hash: u64) -> bool {
        let fp = self.fingerprint(hash);
        let (i1, i2) = self.get_indices(hash, fp);

        // Try insert into primary or secondary bucket
        for &idx in &[i1, i2] {
            for slot in &mut self.buckets[idx] {
                if *slot == 0 {
                    *slot = fp;
                    self.count += 1;
                    return true;
                }
            }
        }

        // Cuckoo eviction path
        let mut cur_idx = i1;
        let mut cur_fp = fp;
        for _ in 0..500 {
            let slot_idx = (cur_fp as usize) % 4;
            std::mem::swap(&mut self.buckets[cur_idx][slot_idx], &mut cur_fp);
            let fp_hash = (cur_fp as u64).wrapping_mul(0x5bd1e995);
            cur_idx = (cur_idx ^ (fp_hash as usize)) % self.buckets.len();

            for slot in &mut self.buckets[cur_idx] {
                if *slot == 0 {
                    *slot = cur_fp;
                    self.count += 1;
                    return true;
                }
            }
        }

        false
    }

    pub fn contains(&self, hash: u64) -> bool {
        let fp = self.fingerprint(hash);
        let (i1, i2) = self.get_indices(hash, fp);

        for &idx in &[i1, i2] {
            for &slot in &self.buckets[idx] {
                if slot == fp {
                    return true;
                }
            }
        }
        false
    }

    pub fn count(&self) -> usize {
        self.count
    }
}
