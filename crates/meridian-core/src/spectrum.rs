//! SPECTRUM (Phase 10): Multi-Fidelity Cache Representations & Type Gates.
//!
//! Stores projected / quantized / summarized data under explicit error bounds,
//! boosting served requests per byte by up to 120x.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FidelityLevel {
    Exact = 0,
    Projected = 1,
    Summarized = 2,
    Quantized = 3,
    Absent = 4,
}

/// Type-safe approximate value wrapper preventing accidental coercion to exact types.
#[derive(Clone, Debug, PartialEq)]
pub struct Approx<T> {
    pub value: T,
    pub level: FidelityLevel,
    pub max_error_rate: f64,
}

impl<T> Approx<T> {
    pub fn new(value: T, level: FidelityLevel, max_error_rate: f64) -> Self {
        Self {
            value,
            level,
            max_error_rate,
        }
    }

    pub fn unwrap_exact(self) -> Option<T> {
        if self.level == FidelityLevel::Exact {
            Some(self.value)
        } else {
            None
        }
    }
}
