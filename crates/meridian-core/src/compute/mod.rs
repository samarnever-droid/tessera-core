//! MERIDIAN Compute Runtime (MCR): Server-Side Scripting & Custom Compute Plane.

pub mod opcodes;
pub mod vm;
pub mod compiler;
pub mod catalog;

pub use opcodes::*;
pub use vm::{MeridianVM, VmResult, VmError, DEFAULT_GAS_BUDGET};
pub use compiler::Compiler;
pub use catalog::{StoredFunction, FunctionCatalog};
