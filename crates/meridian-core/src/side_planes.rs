//! Capability side-planes (spec §4.2 / architect-thinking 03).
//!
//! Keeps the hot entry layout strictly at 16 bytes (one cache line per probe)
//! while providing parallel, indexed per-shard storage for optional capabilities:
//! - prov[]: Provenance dependency set reference + created_lsn (8 B)
//! - gen[] : Posting reuse generation (1 B)
//! - vers[]: Commit LSN version chain for snapshots (11 B)
//! - fid[] : Fidelity level & error bounds (2 B)
//! - maint[]: Differential operator state (6 B)
//! - dl[]  : Latency EWMA and slack tracking (4 B)

/// Bit flags in the entry `ctl` field indicating which planes are allocated.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanesMask(pub u8);

impl PlanesMask {
    pub const NONE: Self = Self(0);
    pub const PROV: Self = Self(1 << 0);
    pub const VERS: Self = Self(1 << 1);
    pub const FID: Self = Self(1 << 2);
    pub const MAINT: Self = Self(1 << 3);
    pub const DL: Self = Self(1 << 4);

    #[inline]
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// Provenance & Dependency tracking plane (8 bytes)
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ProvEntry {
    pub dep_set_id: u32,
    pub created_lsn: u64,
}

/// Version chain plane for CHRONOS snapshot isolation (11 bytes packed)
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VersEntry {
    /// Commit LSN at which this version became valid
    pub valid_from: u64,
    /// Cell reference or payload index
    pub payload_ref: u64,
    /// Previous version slot index (0 = none)
    pub prev_version_idx: u32,
}

/// Fidelity plane for SPECTRUM approximate caching (2 bytes)
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FidEntry {
    /// 0=Exact, 1=Projected, 2=Summarized, 3=Quantized, 4=Absent
    pub level: u8,
    /// Declared error bound in basis points (e.g. 100 = 1.0%)
    pub error_bps: u16,
}

/// Differential maintenance operator state plane (6 bytes)
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MaintEntry {
    pub op_code: u8,
    pub delta_seq: u16,
    pub plan_id: u32,
}

/// Latency statistics & deadline tracking plane (4 bytes)
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DlEntry {
    pub ewma_latency_us: u16,
    pub slack_budget_us: u16,
}

/// Per-shard parallel capability storage arrays.
pub struct ShardSidePlanes {
    pub prov: Vec<ProvEntry>,
    pub gen: Vec<u8>,
    pub vers: Vec<VersEntry>,
    pub fid: Vec<FidEntry>,
    pub maint: Vec<MaintEntry>,
    pub dl: Vec<DlEntry>,
}

impl ShardSidePlanes {
    pub fn new(capacity: usize) -> Self {
        Self {
            prov: vec![ProvEntry::default(); capacity],
            gen: vec![0; capacity],
            vers: vec![VersEntry::default(); capacity],
            fid: vec![FidEntry::default(); capacity],
            maint: vec![MaintEntry::default(); capacity],
            dl: vec![DlEntry::default(); capacity],
        }
    }

    pub fn resize(&mut self, new_capacity: usize) {
        self.prov.resize(new_capacity, ProvEntry::default());
        self.gen.resize(new_capacity, 0);
        self.vers.resize(new_capacity, VersEntry::default());
        self.fid.resize(new_capacity, FidEntry::default());
        self.maint.resize(new_capacity, MaintEntry::default());
        self.dl.resize(new_capacity, DlEntry::default());
    }

    #[inline]
    pub fn clear_slot(&mut self, slot: usize) {
        if slot < self.gen.len() {
            self.gen[slot] = self.gen[slot].wrapping_add(1);
            self.prov[slot] = ProvEntry::default();
            self.vers[slot] = VersEntry::default();
            self.fid[slot] = FidEntry::default();
            self.maint[slot] = MaintEntry::default();
            self.dl[slot] = DlEntry::default();
        }
    }
}
