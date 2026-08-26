//! Circular token copy buffer and single-query buffer attention kernels.

use crate::softmax::softmax;
use crate::tensor::dot;

/// Fixed-capacity circular ring buffer for storing recent or surprising tokens.
/// Zero dynamic allocations after initialization.
#[derive(Debug, Clone)]
pub struct CircularTokenBuffer {
    capacity: usize,
    head: usize,
    len: usize,
    buffer: Vec<u32>,
}

impl CircularTokenBuffer {
    /// Create a pre-allocated circular token buffer of fixed capacity B.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be > 0");
        Self {
            capacity,
            head: 0,
            len: 0,
            buffer: vec![0; capacity],
        }
    }

    /// Reset buffer state to empty.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Push a token into the ring buffer, overwriting the oldest entry if full. O(1).
    #[inline]
    pub fn push(&mut self, token: u32) {
        self.buffer[self.head] = token;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Number of tokens currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Is buffer empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over current tokens in chronological order (oldest to newest).
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        (0..self.len).map(move |i| {
            let idx = (start + i) % self.capacity;
            self.buffer[idx]
        })
    }

    /// Internal raw slice access.
    #[inline]
    pub fn raw_slice(&self) -> &[u32] {
        &self.buffer[..self.len]
    }
}

/// Compute single-query buffer attention (§3.4):
/// 1. For each token in the buffer, compute score s_i = (query · embed(token_i)) / sqrt(d)
/// 2. Softmax over valid buffer scores -> weights alpha_i
/// 3. Scatter weights into vocabulary distribution: copy_dist[token_i] += alpha_i
/// 4. Interpolate final distribution: out_dist = (1 - alpha_mix) * out_dist + alpha_mix * copy_dist
///
/// Scratch buffers (scratch_scores, scratch_probs) must be at least buffer.len() in size.
#[inline]
pub fn buffer_attention(
    buffer: &CircularTokenBuffer,
    query: &[f32],
    embed_table: &[f32],
    vocab_size: usize,
    embed_dim: usize,
    scratch_scores: &mut [f32],
    scratch_probs: &mut [f32],
    out_vocab_dist: &mut [f32],
    alpha_mix: f32,
) {
    let buf_len = buffer.len();
    if buf_len == 0 {
        return;
    }
    debug_assert!(scratch_scores.len() >= buf_len);
    debug_assert!(scratch_probs.len() >= buf_len);
    debug_assert_eq!(out_vocab_dist.len(), vocab_size);

    let scale = 1.0f32 / (embed_dim as f32).sqrt();

    // 1. Compute dot-product attention scores
    for (i, token) in buffer.iter().enumerate() {
        let t = token as usize;
        if t < vocab_size {
            let token_embed = &embed_table[t * embed_dim..(t + 1) * embed_dim];
            scratch_scores[i] = dot(query, token_embed) * scale;
        } else {
            scratch_scores[i] = f32::NEG_INFINITY;
        }
    }

    // 2. Softmax over buffer
    let active_scores = &scratch_scores[..buf_len];
    let active_probs = &mut scratch_probs[..buf_len];
    softmax(active_scores, active_probs);

    // 3. Interpolate: scale original distribution by (1 - alpha_mix)
    let inv_alpha = 1.0f32 - alpha_mix;
    for p in out_vocab_dist.iter_mut() {
        *p *= inv_alpha;
    }

    // 4. Scatter copy distribution
    for (i, token) in buffer.iter().enumerate() {
        let t = token as usize;
        if t < vocab_size {
            out_vocab_dist[t] += alpha_mix * active_probs[i];
        }
    }
}
