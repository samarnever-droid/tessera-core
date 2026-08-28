//! 3-Phase Zero-Downtime Live Slot Migration Controller.

use crate::cluster::slots::{SlotState, SlotTable};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationPhase {
    BulkSnapshot,
    DeltaCatchup,
    AtomicCutover,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationTask {
    pub slot: u16,
    pub source: u64,
    pub target: u64,
    pub phase: MigrationPhase,
    pub keys_transferred: usize,
    pub deltas_forwarded: usize,
}

pub struct MigrationController {
    tasks: RwLock<HashMap<u16, MigrationTask>>,
}

impl MigrationController {
    pub fn new() -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
        }
    }

    pub fn start_migration(
        &self,
        slot: u16,
        source: u64,
        target: u64,
        slot_table: &SlotTable,
    ) {
        let task = MigrationTask {
            slot,
            source,
            target,
            phase: MigrationPhase::BulkSnapshot,
            keys_transferred: 0,
            deltas_forwarded: 0,
        };

        self.tasks.write().unwrap().insert(slot, task);
        slot_table.set_slot_state(slot, SlotState::Migrating { source, target });
    }

    pub fn advance_to_delta_catchup(&self, slot: u16, bulk_count: usize) {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(&slot) {
            task.phase = MigrationPhase::DeltaCatchup;
            task.keys_transferred = bulk_count;
        }
    }

    pub fn forward_delta(&self, slot: u16) {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(&slot) {
            task.deltas_forwarded += 1;
        }
    }

    pub fn commit_cutover(&self, slot: u16, slot_table: &SlotTable) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(&slot) {
            task.phase = MigrationPhase::Completed;
            slot_table.set_slot_state(slot, SlotState::Stable { owner: task.target });
            true
        } else {
            false
        }
    }

    pub fn get_task(&self, slot: u16) -> Option<MigrationTask> {
        self.tasks.read().unwrap().get(&slot).cloned()
    }
}

impl Default for MigrationController {
    fn default() -> Self {
        Self::new()
    }
}
