//! MERIDIAN Append-Only Log (AOF) & Durability Subsystem.

pub mod frame;
pub mod writer;
pub mod recovery;

pub use frame::{AofOpcode, AofRecord, AOF_MAGIC};
pub use writer::{AofWriter, AofSyncPolicy};
pub use recovery::{AofRecovery, AofRecoveryResult};
