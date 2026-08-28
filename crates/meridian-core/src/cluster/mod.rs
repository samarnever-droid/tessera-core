//! MERIDIAN Dynamic Resharding & Multi-Region WAN Mesh Subsystem.

pub mod slots;
pub mod migration;
pub mod wan;

pub use slots::{SlotTable, SlotState, get_slot, crc16, TOTAL_SLOTS};
pub use migration::{MigrationController, MigrationTask, MigrationPhase};
pub use wan::{WanMesh, WanDelta, VectorClock};
