//! 16,384 CRC16 Hash Slot Topology & State Machine (Phase 25).

use std::sync::RwLock;

pub const TOTAL_SLOTS: usize = 16384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Stable { owner: u64 },
    Migrating { source: u64, target: u64 },
    Importing { source: u64, target: u64 },
}

/// CRC16-CCITT implementation for deterministic slot mapping.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0x0000;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

pub fn get_slot(key: &[u8]) -> u16 {
    crc16(key) % (TOTAL_SLOTS as u16)
}

pub struct SlotTable {
    slots: RwLock<Vec<SlotState>>,
}

impl SlotTable {
    pub fn new(default_owner: u64) -> Self {
        let mut slots = Vec::with_capacity(TOTAL_SLOTS);
        for _ in 0..TOTAL_SLOTS {
            slots.push(SlotState::Stable { owner: default_owner });
        }
        Self {
            slots: RwLock::new(slots),
        }
    }

    pub fn assign_range(&self, start: u16, end: u16, owner: u64) {
        let mut slots = self.slots.write().unwrap();
        for s in start..=end.min(TOTAL_SLOTS as u16 - 1) {
            slots[s as usize] = SlotState::Stable { owner };
        }
    }

    pub fn get_slot_state(&self, slot: u16) -> SlotState {
        self.slots.read().unwrap()[slot as usize]
    }

    pub fn set_slot_state(&self, slot: u16, state: SlotState) {
        self.slots.write().unwrap()[slot as usize] = state;
    }

    /// Routes key to target node, indicating if ASK redirect is required.
    pub fn route_key(&self, key: &[u8]) -> (u64, bool) {
        let slot = get_slot(key);
        match self.get_slot_state(slot) {
            SlotState::Stable { owner } => (owner, false),
            SlotState::Migrating { target, .. } => (target, true), // ASK redirect
            SlotState::Importing { target, .. } => (target, false),
        }
    }
}
