//! Small String Optimization (SSO) for Zero-Allocation Short Keys and Values.

use std::sync::Arc;

pub const INLINE_CAPACITY: usize = 15;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompactBytes {
    Inline { len: u8, data: [u8; INLINE_CAPACITY] },
    Heap(Arc<[u8]>),
}

impl CompactBytes {
    pub fn new(slice: &[u8]) -> Self {
        if slice.len() <= INLINE_CAPACITY {
            let mut data = [0u8; INLINE_CAPACITY];
            data[..slice.len()].copy_from_slice(slice);
            CompactBytes::Inline {
                len: slice.len() as u8,
                data,
            }
        } else {
            CompactBytes::Heap(Arc::from(slice))
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            CompactBytes::Inline { len, data } => &data[..*len as usize],
            CompactBytes::Heap(arc) => arc.as_ref(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CompactBytes::Inline { len, .. } => *len as usize,
            CompactBytes::Heap(arc) => arc.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_inline(&self) -> bool {
        matches!(self, CompactBytes::Inline { .. })
    }
}

impl AsRef<[u8]> for CompactBytes {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
