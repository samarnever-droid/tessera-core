//! 2-Stage Crash Recovery & Torn-Write Truncation Scanner.

use crate::aof::frame::AofRecord;
use std::collections::HashMap;

pub struct AofRecoveryResult {
    pub records_replayed: usize,
    pub bytes_truncated: usize,
    pub final_state: HashMap<Vec<u8>, Vec<u8>>,
    pub max_lsn: u64,
}

pub struct AofRecovery;

impl AofRecovery {
    /// Scans raw AOF bytes, validates CRC32 on every record, and replays state.
    pub fn replay(raw_bytes: &[u8]) -> AofRecoveryResult {
        let mut state = HashMap::new();
        let mut pos = 0;
        let mut records_replayed = 0;
        let mut max_lsn = 0;

        while pos < raw_bytes.len() {
            match AofRecord::decode(&raw_bytes[pos..]) {
                Some((rec, frame_len)) => {
                    max_lsn = max_lsn.max(rec.lsn);
                    match rec.opcode {
                        crate::aof::frame::AofOpcode::Set | crate::aof::frame::AofOpcode::Delta => {
                            state.insert(rec.key, rec.value);
                        }
                        crate::aof::frame::AofOpcode::Del => {
                            state.remove(&rec.key);
                        }
                        _ => {}
                    }
                    pos += frame_len;
                    records_replayed += 1;
                }
                None => {
                    // Corrupted or incomplete trailing bytes encountered -> Truncate safely!
                    break;
                }
            }
        }

        let bytes_truncated = raw_bytes.len() - pos;

        AofRecoveryResult {
            records_replayed,
            bytes_truncated,
            final_state: state,
            max_lsn,
        }
    }
}
