//! AOF Binary Frame Layout, CRC32 Checksums, and Monotonic LSNs (Phase 21).

pub const AOF_MAGIC: [u8; 4] = *b"MERI";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AofOpcode {
    Set = 1,
    Del = 2,
    Delta = 3,
    Stream = 4,
}

impl AofOpcode {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            1 => Some(AofOpcode::Set),
            2 => Some(AofOpcode::Del),
            3 => Some(AofOpcode::Delta),
            4 => Some(AofOpcode::Stream),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AofRecord {
    pub lsn: u64,
    pub opcode: AofOpcode,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub timestamp: u64,
}

impl AofRecord {
    /// Simple CRC32 implementation for frame validation without external deps.
    pub fn compute_crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                let mask = if (crc & 1) != 0 { 0xEDB88320 } else { 0 };
                crc = (crc >> 1) ^ mask;
            }
        }
        !crc
    }

    /// Encodes record into a binary framed slice.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + self.key.len() + self.value.len());
        buf.extend_from_slice(&AOF_MAGIC);
        buf.extend_from_slice(&self.lsn.to_le_bytes());
        buf.push(self.opcode as u8);
        buf.extend_from_slice(&(self.key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(self.value.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&self.value);

        let checksum = Self::compute_crc32(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Decodes a single record from byte slice, returning record and consumed bytes.
    pub fn decode(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.len() < 31 {
            return None;
        }

        if &bytes[0..4] != &AOF_MAGIC {
            return None;
        }

        let lsn = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        let opcode = AofOpcode::from_u8(bytes[12])?;
        let key_len = u16::from_le_bytes(bytes[13..15].try_into().ok()?) as usize;
        let val_len = u32::from_le_bytes(bytes[15..19].try_into().ok()?) as usize;
        let timestamp = u64::from_le_bytes(bytes[19..27].try_into().ok()?);

        let total_frame_len = 27 + key_len + val_len + 4;
        if bytes.len() < total_frame_len {
            return None; // Incomplete trailing frame
        }

        let payload_end = 27 + key_len + val_len;
        let key = bytes[27..27 + key_len].to_vec();
        let value = bytes[27 + key_len..payload_end].to_vec();

        let expected_checksum = u32::from_le_bytes(bytes[payload_end..payload_end + 4].try_into().ok()?);
        let computed_checksum = Self::compute_crc32(&bytes[0..payload_end]);

        if expected_checksum != computed_checksum {
            return None; // Checksum mismatch / Corrupt frame
        }

        Some((
            AofRecord {
                lsn,
                opcode,
                key,
                value,
                timestamp,
            },
            total_frame_len,
        ))
    }
}
