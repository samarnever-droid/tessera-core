//! MERIDIAN Vector Search & HNSW Graph Indexing Engine.

pub mod simd_dist;
pub mod hnsw;
pub mod bq;

pub use simd_dist::{cosine_similarity, euclidean_distance_sq, dot_product};
pub use hnsw::{HnswIndex, HnswNode};
pub use bq::{BqIndex, BqVector, quantize_1bit};
