//! Dataset loader and batch generator for character/byte-level corpora.

use rand::rngs::StdRng;
use rand::Rng;
use std::fs;
use std::path::Path;

/// Character/byte-level dataset supporting fast contiguous batch sampling.
#[derive(Debug, Clone)]
pub struct CharDataset {
    pub data: Vec<u8>,
    pub vocab_size: usize,
}

impl CharDataset {
    /// Load from a text or binary file (e.g. tiny-shakespeare or enwik8) as byte sequence.
    pub fn from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let data = fs::read(path)?;
        Ok(Self {
            data,
            vocab_size: 256, // Full byte vocabulary (0..255)
        })
    }

    /// Load with a maximum byte limit (e.g. first 10MB of enwik8 for fast training).
    pub fn from_file_limit<P: AsRef<Path>>(path: P, max_bytes: usize) -> std::io::Result<Self> {
        let mut data = fs::read(path)?;
        if data.len() > max_bytes {
            data.truncate(max_bytes);
        }
        Ok(Self {
            data,
            vocab_size: 256,
        })
    }

    /// Split dataset into (train_dataset, val_dataset) by split_ratio (e.g. 0.9).
    pub fn split(&self, split_ratio: f32) -> (Self, Self) {
        let split_idx = ((self.data.len() as f32) * split_ratio) as usize;
        let train_data = self.data[..split_idx].to_vec();
        let val_data = self.data[split_idx..].to_vec();
        (
            Self {
                data: train_data,
                vocab_size: self.vocab_size,
            },
            Self {
                data: val_data,
                vocab_size: self.vocab_size,
            },
        )
    }

    /// Total number of tokens/bytes in dataset.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sample a random batch of sequences: (batch_size, seq_len)
    /// Returns (inputs, targets) where targets are inputs shifted by 1.
    pub fn sample_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        rng: &mut StdRng,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        assert!(self.data.len() > seq_len + 1, "Dataset too small for sequence length");
        let max_start = self.data.len() - seq_len - 1;

        let mut inputs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let start = rng.gen_range(0..=max_start);
            let seq = &self.data[start..start + seq_len + 1];
            let x: Vec<usize> = seq[..seq_len].iter().map(|&b| b as usize).collect();
            let y: Vec<usize> = seq[1..seq_len + 1].iter().map(|&b| b as usize).collect();
            inputs.push(x);
            targets.push(y);
        }

        (inputs, targets)
    }
}
