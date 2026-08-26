//! `axiom-core`: Foundation kernels, matrix-vector primitives, SIMD utilities,
//! top-k selection, numerically-stable softmax, circular copy buffer, and
//! online Hebbian associative memory for AXIOM.

pub mod activations;
pub mod buffer;
pub mod hebbian;
pub mod matvec;
pub mod reference;
pub mod softmax;
pub mod tensor;
pub mod topk;

pub use activations::*;
pub use buffer::*;
pub use hebbian::*;
pub use matvec::*;
pub use softmax::*;
pub use tensor::*;
pub use topk::*;
