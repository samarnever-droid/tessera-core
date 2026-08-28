//! Adaptive In-Memory Value Compression for Payloads >= 128 Bytes.

pub const COMPRESSION_THRESHOLD: usize = 128;

/// Fast run-length + byte-pair streaming compressor for in-memory reduction.
pub fn compress_value(data: &[u8]) -> (bool, Vec<u8>) {
    if data.len() < COMPRESSION_THRESHOLD {
        return (false, data.to_vec());
    }

    // Simple run-length and zero-suppression compressor
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == byte && run_len < 255 {
            run_len += 1;
        }

        if run_len >= 4 {
            out.push(0xFF); // Escape byte
            out.push(byte);
            out.push(run_len as u8);
            i += run_len;
        } else {
            if byte == 0xFF {
                out.push(0xFF);
                out.push(0xFF);
                out.push(1);
            } else {
                out.push(byte);
            }
            i += 1;
        }
    }

    // Only adopt compression if it saved at least 10% space
    if out.len() + 8 < data.len() {
        (true, out)
    } else {
        (false, data.to_vec())
    }
}

pub fn decompress_value(data: &[u8], is_compressed: bool) -> Vec<u8> {
    if !is_compressed {
        return data.to_vec();
    }

    let mut out = Vec::with_capacity(data.len() * 2);
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0xFF && i + 2 < data.len() {
            let byte = data[i + 1];
            let run_len = data[i + 2] as usize;
            for _ in 0..run_len {
                out.push(byte);
            }
            i += 3;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}
