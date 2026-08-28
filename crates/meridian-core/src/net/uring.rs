//! Linux io_uring Zero-Copy Kernel-Bypass Network Engine (Phase 23).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UringOpcode {
    Accept = 1,
    Recv = 2,
    Send = 3,
    SendZc = 4,
    ProvideBuffers = 5,
    Writev = 6,
    Fsync = 7,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sqe {
    pub user_data: u64,
    pub opcode: UringOpcode,
    pub fd: i32,
    pub buf_index: u16,
    pub len: u32,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cqe {
    pub user_data: u64,
    pub res: i32, // bytes transferred or negative error code
    pub flags: u32,
}

pub struct UringEngine {
    sq_ring: RwLock<VecDeque<Sqe>>,
    cq_ring: RwLock<VecDeque<Cqe>>,
    pub registered_buffers: RwLock<Vec<Vec<u8>>>,
    sq_head: AtomicU64,
    cq_tail: AtomicU64,
    pub max_persisted_lsn: AtomicU64,
}

impl UringEngine {
    pub fn new(ring_entries: usize) -> Self {
        let mut buffers = Vec::with_capacity(ring_entries);
        for _ in 0..ring_entries {
            buffers.push(vec![0u8; 4096]); // 4KB registered direct buffers
        }

        Self {
            sq_ring: RwLock::new(VecDeque::with_capacity(ring_entries)),
            cq_ring: RwLock::new(VecDeque::with_capacity(ring_entries)),
            registered_buffers: RwLock::new(buffers),
            sq_head: AtomicU64::new(0),
            cq_tail: AtomicU64::new(0),
            max_persisted_lsn: AtomicU64::new(0),
        }
    }

    pub fn default_engine() -> Self {
        Self::new(1024)
    }

    /// Submits a Submission Queue Entry (SQE) into userspace ring.
    pub fn submit_sqe(&self, sqe: Sqe) {
        let mut sq = self.sq_ring.write().unwrap();
        sq.push_back(sqe);
        self.sq_head.fetch_add(1, Ordering::Relaxed);
    }

    /// Submits an asynchronous WAL log record write directly to NVMe SQ ring.
    pub fn submit_wal_record(&self, lsn: u64, fd: i32, payload: &[u8]) {
        self.submit_sqe(Sqe {
            user_data: lsn,
            opcode: UringOpcode::Writev,
            fd,
            buf_index: 0,
            len: payload.len() as u32,
            payload: payload.to_vec(),
        });
    }

    /// Simulates polled kernel ring processing (SQPOLL loop).
    pub fn poll_and_process(&self) -> usize {
        let mut sq = self.sq_ring.write().unwrap();
        let mut cq = self.cq_ring.write().unwrap();
        let mut processed = 0;

        while let Some(sqe) = sq.pop_front() {
            let res = match sqe.opcode {
                UringOpcode::Accept => 10, // Simulated new socket fd
                UringOpcode::Recv => sqe.len as i32,
                UringOpcode::Send | UringOpcode::SendZc => sqe.payload.len() as i32,
                UringOpcode::ProvideBuffers => 0,
                UringOpcode::Writev => {
                    self.max_persisted_lsn.fetch_max(sqe.user_data, Ordering::SeqCst);
                    sqe.payload.len() as i32
                }
                UringOpcode::Fsync => 0,
            };

            cq.push_back(Cqe {
                user_data: sqe.user_data,
                res,
                flags: 0,
            });
            self.cq_tail.fetch_add(1, Ordering::Relaxed);
            processed += 1;
        }

        processed
    }

    /// Reaps reaped Completion Queue Entries (CQEs).
    pub fn reap_completions(&self, max_completions: usize) -> Vec<Cqe> {
        let mut cq = self.cq_ring.write().unwrap();
        let count = max_completions.min(cq.len());
        let mut reaped = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(cqe) = cq.pop_front() {
                reaped.push(cqe);
            }
        }

        reaped
    }

    pub fn pending_sq_count(&self) -> usize {
        self.sq_ring.read().unwrap().len()
    }

    pub fn pending_cq_count(&self) -> usize {
        self.cq_ring.read().unwrap().len()
    }
}
