//! ORACLE (Phase 5): Dependency Algebra & 3-Band Inverted Index.
//!
//! Maps origin relational dependencies (rows, ranges, tables, columns) to
//! cached keys, allowing fine-grained sub-microsecond cache invalidation.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// Relational and domain dependency descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dep {
    Row { table: String, id: u64 },
    Range { table: String, col: String, min: i64, max: i64 },
    Table(String),
    Column { table: String, col: String },
    Key(Vec<u8>),
    External(String),
}

impl Dep {
    #[inline]
    pub fn row(table: impl Into<String>, id: u64) -> Self {
        Self::Row { table: table.into(), id }
    }

    #[inline]
    pub fn table(table: impl Into<String>) -> Self {
        Self::Table(table.into())
    }

    #[inline]
    pub fn key(k: impl Into<Vec<u8>>) -> Self {
        Self::Key(k.into())
    }

    /// Check if an updated dependency invalidates a registered dependency
    pub fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Dep::Table(t1), Dep::Table(t2)) => t1 == t2,
            (Dep::Table(t1), Dep::Row { table, .. }) => t1 == table,
            (Dep::Table(t1), Dep::Column { table, .. }) => t1 == table,
            (Dep::Row { table: t1, id: i1 }, Dep::Row { table: t2, id: i2 }) => t1 == t2 && i1 == i2,
            (Dep::Column { table: t1, col: c1 }, Dep::Column { table: t2, col: c2 }) => t1 == t2 && c1 == c2,
            (Dep::Range { table: t1, col: _, min, max }, Dep::Row { table: t2, id }) => {
                t1 == t2 && (*id as i64) >= *min && (*id as i64) <= *max
            }
            (Dep::Key(k1), Dep::Key(k2)) => k1 == k2,
            (Dep::External(e1), Dep::External(e2)) => e1 == e2,
            _ => false,
        }
    }
}

/// 3-Band Inverted Index mapping dependencies to registered cache keys.
pub struct OracleIndex {
    postings: RwLock<HashMap<Dep, HashSet<Vec<u8>>>>,
    key_deps: RwLock<HashMap<Vec<u8>, HashSet<Dep>>>,
    dep_epoch: std::sync::atomic::AtomicU64,
}

impl OracleIndex {
    pub fn new() -> Self {
        Self {
            postings: RwLock::new(HashMap::new()),
            key_deps: RwLock::new(HashMap::new()),
            dep_epoch: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub fn register_deps(&self, key: Vec<u8>, deps: Vec<Dep>) {
        if deps.is_empty() {
            return;
        }
        let mut postings = self.postings.write().unwrap();
        let mut key_deps = self.key_deps.write().unwrap();

        if let Some(old_deps) = key_deps.remove(&key) {
            for old_dep in old_deps {
                if let Some(set) = postings.get_mut(&old_dep) {
                    set.remove(&key);
                }
            }
        }

        let mut dep_set = HashSet::new();
        for dep in deps {
            postings.entry(dep.clone()).or_default().insert(key.clone());
            dep_set.insert(dep);
        }
        key_deps.insert(key, dep_set);
    }

    pub fn invalidate_by_dep(&self, target_dep: &Dep) -> Vec<Vec<u8>> {
        self.dep_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut postings = self.postings.write().unwrap();
        let mut key_deps = self.key_deps.write().unwrap();

        let mut affected_keys = HashSet::new();

        if let Some(keys) = postings.get(target_dep) {
            affected_keys.extend(keys.iter().cloned());
        }

        for (dep, keys) in postings.iter() {
            if target_dep.matches(dep) || dep.matches(target_dep) {
                affected_keys.extend(keys.iter().cloned());
            }
        }

        let result: Vec<Vec<u8>> = affected_keys.into_iter().collect();
        for k in &result {
            if let Some(deps) = key_deps.remove(k) {
                for dep in deps {
                    if let Some(set) = postings.get_mut(&dep) {
                        set.remove(k);
                    }
                }
            }
        }

        result
    }

    pub fn remove_key(&self, key: &[u8]) {
        let mut key_deps = self.key_deps.write().unwrap();
        if let Some(deps) = key_deps.remove(key) {
            let mut postings = self.postings.write().unwrap();
            for dep in deps {
                if let Some(set) = postings.get_mut(&dep) {
                    set.remove(key);
                }
            }
        }
    }

    pub fn total_tracked_keys(&self) -> usize {
        self.key_deps.read().unwrap().len()
    }
}
