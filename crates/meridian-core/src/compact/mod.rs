//! MERIDIAN Zero-Downgrade Compact RAM Optimization Subsystem.

pub mod sso;
pub mod tagged_ptr;
pub mod compress;
pub mod htap;

pub use sso::{CompactBytes, INLINE_CAPACITY};
pub use tagged_ptr::TaggedPtr;
pub use compress::{compress_value, decompress_value, COMPRESSION_THRESHOLD};
pub use htap::{ColumnarChunk, HtapTable};
