//! Deadline Scheduling (Phase 11): Contractual p99.9 & Proactive Tier Degradation.
//!
//! Evaluates remaining request latency budget against tier EWMAs, degrading
//! gracefully (stale -> projected -> partial) before exceeding the deadline.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DegradeAction {
    ServeExact,
    ServeStaleSnapshot,
    ServeLowerFidelity,
    CancelWithDeadlineExceeded,
}

pub struct DeadlineScheduler {
    pub default_budget_us: u32,
}

impl DeadlineScheduler {
    pub fn new(default_budget_us: u32) -> Self {
        Self { default_budget_us }
    }

    /// Evaluates what action to take given elapsed time and tier expected latency.
    pub fn evaluate_tier(&self, elapsed_us: u32, budget_us: u32, tier_expected_latency_us: u32) -> DegradeAction {
        let slack = budget_us.saturating_sub(elapsed_us);
        if slack >= tier_expected_latency_us {
            DegradeAction::ServeExact
        } else if slack >= 50 {
            DegradeAction::ServeLowerFidelity
        } else {
            DegradeAction::CancelWithDeadlineExceeded
        }
    }
}
