//! Quantized Embedding Vector Plane (Phase 17):
//! In-Cache Int8 Quantization and Sub-Microsecond Cosine Similarity Search for Semantic AI Caching.

#[derive(Debug, Clone)]
pub struct QuantizedVector {
    pub id: u64,
    pub data: Vec<i8>,
    pub scale: f32,
    pub norm: f32,
}

impl QuantizedVector {
    /// Quantizes an FP32 embedding vector into Int8 bytes.
    pub fn from_f32(id: u64, raw: &[f32]) -> Self {
        let max_abs = raw.iter().fold(1e-6f32, |acc, &x| acc.max(x.abs()));
        let scale = max_abs / 127.0;
        let mut norm_sq = 0.0f32;
        let mut data = Vec::with_capacity(raw.len());

        for &val in raw {
            let q = ((val / scale).round()).clamp(-127.0, 127.0) as i8;
            data.push(q);
            norm_sq += val * val;
        }

        Self {
            id,
            data,
            scale,
            norm: norm_sq.sqrt(),
        }
    }

    /// Computes cosine similarity against a query FP32 vector.
    pub fn cosine_similarity(&self, query: &[f32], query_norm: f32) -> f32 {
        if self.norm == 0.0 || query_norm == 0.0 || self.data.len() != query.len() {
            return 0.0;
        }

        let mut dot_prod = 0.0f32;
        for (i, &q_val) in self.data.iter().enumerate() {
            let dequant = (q_val as f32) * self.scale;
            dot_prod += dequant * query[i];
        }

        dot_prod / (self.norm * query_norm)
    }
}

/// In-Cache Semantic Vector Index.
#[derive(Debug, Clone, Default)]
pub struct VectorIndex {
    vectors: Vec<QuantizedVector>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or updates a vector in the index.
    pub fn add(&mut self, id: u64, embedding: &[f32]) {
        self.vectors.retain(|v| v.id != id);
        self.vectors.push(QuantizedVector::from_f32(id, embedding));
    }

    /// Searches for Top-K nearest neighbor vectors by cosine similarity.
    pub fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        let query_norm = query_norm_sq.sqrt();

        let mut scores: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|v| (v.id, v.cosine_similarity(query, query_norm)))
            .collect();

        // Sort descending by similarity score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}
