//! RESP protocol (Redis Serialization Protocol) versions 2 and 3.
//!
//! The decoder accepts what clients send: arrays of bulk strings, plus inline
//! commands as a fallback. The encoder produces both RESP2 and RESP3 forms;
//! maps and nulls degrade to their RESP2 equivalents when the connection has
//! not negotiated protocol 3 via HELLO 3.

const MAX_BULK: usize = 512 * 1024 * 1024;
const MAX_ARRAY: i64 = 1_000_000;
const MAX_LINE: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Simple(String),
    Error(String),
    Int(i64),
    Bulk(Vec<u8>),
    Null,
    Array(Vec<Frame>),
    Map(Vec<(Frame, Frame)>),
}

impl Frame {
    pub fn encode(&self, out: &mut Vec<u8>, proto3: bool) {
        match self {
            Frame::Simple(s) => write_line(out, b'+', s),
            Frame::Error(s) => write_line(out, b'-', s),
            Frame::Int(i) => {
                out.extend_from_slice(format!(":{i}\r\n").as_bytes());
            }
            Frame::Bulk(b) => {
                out.extend_from_slice(format!("${}\r\n", b.len()).as_bytes());
                out.extend_from_slice(b);
                out.extend_from_slice(b"\r\n");
            }
            Frame::Null => {
                if proto3 {
                    out.extend_from_slice(b"_\r\n");
                } else {
                    out.extend_from_slice(b"$-1\r\n");
                }
            }
            Frame::Array(items) => {
                out.extend_from_slice(format!("*{}\r\n", items.len()).as_bytes());
                for it in items {
                    it.encode(out, proto3);
                }
            }
            Frame::Map(pairs) => {
                if proto3 {
                    out.extend_from_slice(format!("%{}\r\n", pairs.len()).as_bytes());
                    for (k, v) in pairs {
                        k.encode(out, proto3);
                        v.encode(out, proto3);
                    }
                } else {
                    out.extend_from_slice(format!("*{}\r\n", pairs.len() * 2).as_bytes());
                    for (k, v) in pairs {
                        k.encode(out, proto3);
                        v.encode(out, proto3);
                    }
                }
            }
        }
    }
}

fn write_line(out: &mut Vec<u8>, prefix: u8, s: &str) {
    out.push(prefix);
    for &b in s.as_bytes() {
        // keep the wire format intact: no embedded CRLF in single-line frames
        out.push(if b == b'\r' || b == b'\n' { b' ' } else { b });
    }
    out.extend_from_slice(b"\r\n");
}

#[derive(Debug, PartialEq)]
pub enum ProtoError {
    Malformed(&'static str),
    TooLarge,
}

pub struct Decoder {
    buf: Vec<u8>,
    pos: usize,
}

impl Decoder {
    pub fn new() -> Self {
        Decoder { buf: Vec::new(), pos: 0 }
    }

    pub fn feed(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    fn compact(&mut self) {
        if self.pos > 0 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
    }

    /// Returns the bytes of one line (without CRLF) and the offset just past
    /// its LF, or None if the line has not fully arrived yet.
    fn line(&self, start: usize) -> Result<Option<(&[u8], usize)>, ProtoError> {
        if start >= self.buf.len() {
            return Ok(None);
        }
        match self.buf[start..].iter().position(|&b| b == b'\n') {
            Some(off) => {
                let end = start + off;
                let mut s = &self.buf[start..end];
                if s.ends_with(b"\r") {
                    s = &s[..s.len() - 1];
                }
                if s.len() > MAX_LINE {
                    return Err(ProtoError::Malformed("line too long"));
                }
                Ok(Some((s, end + 1)))
            }
            None => {
                if self.buf.len() - start > MAX_LINE {
                    Err(ProtoError::Malformed("line too long"))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn parse_command_array(&self) -> Result<Option<(Vec<Vec<u8>>, usize)>, ProtoError> {
        let Some((line, mut at)) = self.line(self.pos)? else {
            return Ok(None);
        };
        if line.first() != Some(&b'*') {
            return Err(ProtoError::Malformed("expected array header"));
        }
        let n = parse_int(&line[1..]).ok_or(ProtoError::Malformed("bad array length"))?;
        if n < 0 || n > MAX_ARRAY {
            return Err(ProtoError::Malformed("array length out of range"));
        }
        let mut out = Vec::with_capacity((n as usize).min(16));
        for _ in 0..n {
            let Some((l2, nat)) = self.line(at)? else {
                return Ok(None); // incomplete: wait for more bytes
            };
            if l2.first() != Some(&b'$') {
                return Err(ProtoError::Malformed("expected bulk string"));
            }
            let len = parse_int(&l2[1..]).ok_or(ProtoError::Malformed("bad bulk length"))?;
            if len < 0 {
                return Err(ProtoError::Malformed("negative bulk length"));
            }
            if len as usize > MAX_BULK {
                return Err(ProtoError::TooLarge);
            }
            let start = nat;
            let end = start + len as usize + 2;
            if self.buf.len() < end {
                return Ok(None); // incomplete
            }
            if &self.buf[start + len as usize..end] != b"\r\n" {
                return Err(ProtoError::Malformed("bad bulk terminator"));
            }
            out.push(self.buf[start..start + len as usize].to_vec());
            at = end;
        }
        Ok(Some((out, at)))
    }

    /// Decode one complete command from the buffer. Returns Ok(None) when
    /// more bytes are needed. The buffer is compacted only when the drain
    /// stalls, not per command — a pipelined batch pays one memmove, not
    /// one per command (OPT-1).
    pub fn try_command(&mut self) -> Result<Option<Vec<Vec<u8>>>, ProtoError> {
        loop {
            if self.pos >= self.buf.len() {
                self.compact();
                return Ok(None);
            }
            if self.buf[self.pos] == b'*' {
                return match self.parse_command_array()? {
                    Some((cmd, end)) => {
                        self.pos = end;
                        Ok(Some(cmd))
                    }
                    None => Ok(None),
                };
            }
            // inline command fallback
            match self.line(self.pos)? {
                Some((l, next)) => {
                    // own the data before mutating `pos` (the slice borrows the buffer)
                    let cmd: Option<Vec<Vec<u8>>> = if l.is_empty() {
                        None
                    } else {
                        Some(
                            String::from_utf8_lossy(l)
                                .split_ascii_whitespace()
                                .map(|s| s.as_bytes().to_vec())
                                .collect(),
                        )
                    };
                    self.pos = next;
                    match cmd {
                        Some(c) => return Ok(Some(c)),
                        None => continue,
                    }
                }
                None => {
                    self.compact();
                    return Ok(None);
                }
            }
        }
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_int(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let (neg, digits) = match b[0] {
        b'-' => (true, &b[1..]),
        b'+' => (false, &b[1..]),
        _ => (false, b),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((d - b'0') as i64)?;
    }
    Some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_array_command() {
        let mut d = Decoder::new();
        d.feed(b"*3\r\n$3\r\nSET\r\n$3\r\nfoo");
        assert_eq!(d.try_command().unwrap(), None);
        d.feed(b"\r\n$3\r\nbar\r\n");
        assert_eq!(
            d.try_command().unwrap(),
            Some(vec![b"SET".to_vec(), b"foo".to_vec(), b"bar".to_vec()])
        );
    }

    #[test]
    fn decodes_inline_command() {
        let mut d = Decoder::new();
        d.feed(b"PING\r\n");
        assert_eq!(d.try_command().unwrap(), Some(vec![b"PING".to_vec()]));
    }

    #[test]
    fn skips_blank_inline_lines() {
        let mut d = Decoder::new();
        d.feed(b"\r\nPING\r\n");
        assert_eq!(d.try_command().unwrap(), Some(vec![b"PING".to_vec()]));
    }

    #[test]
    fn rejects_non_bulk_array_element() {
        let mut d = Decoder::new();
        d.feed(b"*1\r\n:5\r\n");
        assert_eq!(d.try_command(), Err(ProtoError::Malformed("expected bulk string")));
    }

    #[test]
    fn encodes_resp2() {
        let mut out = Vec::new();
        Frame::Simple("OK".into()).encode(&mut out, false);
        assert_eq!(out, b"+OK\r\n");
        out.clear();
        Frame::Null.encode(&mut out, false);
        assert_eq!(out, b"$-1\r\n");
        out.clear();
        Frame::Bulk(b"v1".to_vec()).encode(&mut out, false);
        assert_eq!(out, b"$2\r\nv1\r\n");
        out.clear();
        Frame::Int(7).encode(&mut out, false);
        assert_eq!(out, b":7\r\n");
    }

    #[test]
    fn encodes_resp3_null_and_map() {
        let mut out = Vec::new();
        Frame::Null.encode(&mut out, true);
        assert_eq!(out, b"_\r\n");
        out.clear();
        let m = Frame::Map(vec![
            (Frame::Bulk(b"a".to_vec()), Frame::Int(1)),
            (Frame::Bulk(b"b".to_vec()), Frame::Int(2)),
        ]);
        m.encode(&mut out, true);
        assert_eq!(out, b"%2\r\n$1\r\na\r\n:1\r\n$1\r\nb\r\n:2\r\n");
        out.clear();
        m.encode(&mut out, false); // RESP2: flattened array
        assert_eq!(out, b"*4\r\n$1\r\na\r\n:1\r\n$1\r\nb\r\n:2\r\n");
    }
}
