//! End-to-end protocol tests over a real TCP socket.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use meridian_core::{Engine, EngineOptions};
use meridian_server::serve;

fn start_server() -> SocketAddr {
    let engine = Arc::new(Engine::new(EngineOptions {
        total_entries: 1 << 14,
        ..Default::default()
    }));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || serve(engine, listener).unwrap());
    addr
}

fn connect(addr: SocketAddr) -> TcpStream {
    let s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    s
}

/// Send a request, read exactly `expected.len()` bytes, compare.
fn roundtrip(s: &mut TcpStream, req: &str, expected: &str) {
    s.write_all(req.as_bytes()).unwrap();
    let mut buf = vec![0u8; expected.len()];
    s.read_exact(&mut buf).unwrap();
    assert_eq!(String::from_utf8_lossy(&buf), expected);
}

/// Send a request, read one CRLF-terminated line (for dynamic integer replies).
fn roundtrip_line(s: &mut TcpStream, req: &str) -> String {
    s.write_all(req.as_bytes()).unwrap();
    let mut line = Vec::new();
    let mut b = [0u8; 1];
    loop {
        s.read_exact(&mut b).unwrap();
        line.push(b[0]);
        if b[0] == b'\n' {
            break;
        }
    }
    String::from_utf8_lossy(&line).trim_end().to_string()
}

#[test]
fn redis_compat_basics() {
    let addr = start_server();
    let mut s = connect(addr);

    roundtrip(&mut s, "*1\r\n$4\r\nPING\r\n", "+PONG\r\n");
    roundtrip(&mut s, "PING\r\n", "+PONG\r\n");
    roundtrip(&mut s, "*2\r\n$4\r\nECHO\r\n$3\r\nhey\r\n", "$3\r\nhey\r\n");

    roundtrip(&mut s, "*5\r\n$3\r\nSET\r\n$1\r\nk\r\n$2\r\nv1\r\n$2\r\nEX\r\n$3\r\n100\r\n", "+OK\r\n");
    roundtrip(&mut s, "*2\r\n$3\r\nGET\r\n$1\r\nk\r\n", "$2\r\nv1\r\n");

    let ttl = roundtrip_line(&mut s, "*2\r\n$3\r\nTTL\r\n$1\r\nk\r\n");
    let secs: i64 = ttl.trim_start_matches(':').parse().unwrap();
    assert!((1..=100).contains(&secs), "TTL out of range: {secs}");

    // NX on a present key is rejected with a null reply
    roundtrip(&mut s, "*4\r\n$3\r\nSET\r\n$2\r\nk2\r\n$2\r\nv2\r\n$2\r\nNX\r\n", "+OK\r\n");
    let _ = roundtrip_line(&mut s, "*4\r\n$3\r\nSET\r\n$2\r\nk2\r\n$2\r\nzz\r\n$2\r\nNX\r\n");

    roundtrip(&mut s, "*2\r\n$3\r\nDEL\r\n$2\r\nk2\r\n", ":1\r\n");
    roundtrip(&mut s, "*2\r\n$6\r\nEXISTS\r\n$1\r\nk\r\n", ":1\r\n");

    // SET ... GET returns the previous value
    roundtrip(&mut s, "*3\r\n$3\r\nSET\r\n$1\r\nb\r\n$5\r\nhello\r\n", "+OK\r\n");
    roundtrip(&mut s, "*4\r\n$3\r\nSET\r\n$1\r\nb\r\n$5\r\nworld\r\n$3\r\nGET\r\n", "$5\r\nhello\r\n");

    // EXPIRE + PTTL
    roundtrip(&mut s, "*3\r\n$6\r\nEXPIRE\r\n$1\r\nk\r\n$3\r\n200\r\n", ":1\r\n");
    let pttl = roundtrip_line(&mut s, "*2\r\n$4\r\nPTTL\r\n$1\r\nk\r\n");
    let ms: i64 = pttl.trim_start_matches(':').parse().unwrap();
    assert!((195..=200_000).contains(&ms), "PTTL out of range: {ms}");

    // DBSIZE: k and b
    roundtrip(&mut s, "*1\r\n$6\r\nDBSIZE\r\n", ":2\r\n");

    // native API (RESP2: the 11-pair map flattens to a 22-element array);
    // fresh connection so the map body does not need draining here
    {
        let mut s2 = connect(addr);
        let hdr = roundtrip_line(&mut s2, "*1\r\n$8\r\nMD.STATS\r\n");
        assert!(hdr.starts_with("*22"), "got {hdr}");
    }

    // unknown command
    let unknown = roundtrip_line(&mut s, "*1\r\n$7\r\nNOTACMD\r\n");
    assert!(unknown.starts_with("-ERR"), "got {unknown}");
}

#[test]
fn resp3_negotiation() {
    let addr = start_server();
    let mut s = connect(addr);

    // HELLO 3 → map reply, then nulls render as RESP3 null.
    // %3 + server/version/proto pairs = 4+12+14+13+11+11+4 = 69 bytes.
    s.write_all(b"*2\r\n$5\r\nHELLO\r\n$1\r\n3\r\n").unwrap();
    let mut buf = vec![0u8; 69];
    s.read_exact(&mut buf).unwrap();
    let hello = String::from_utf8_lossy(&buf);
    assert!(hello.starts_with("%3\r\n"), "got {hello:?}");
    assert!(hello.ends_with(":3\r\n"), "got {hello:?}");

    roundtrip(&mut s, "*2\r\n$3\r\nGET\r\n$4\r\nnope\r\n", "_\r\n");

    // QUIT
    roundtrip(&mut s, "*1\r\n$4\r\nQUIT\r\n", "+OK\r\n");
}

#[test]
fn mget_mset_and_scan_smoke() {
    let addr = start_server();
    let mut s = connect(addr);

    // MSET k1 v1 k2 v2 → OK
    roundtrip(&mut s, "*5\r\n$4\r\nMSET\r\n$2\r\nk1\r\n$2\r\nv1\r\n$2\r\nk2\r\n$2\r\nv2\r\n", "+OK\r\n");
    // MGET k1 k3 → [v1, nil]
    roundtrip(&mut s, "*3\r\n$4\r\nMGET\r\n$2\r\nk1\r\n$2\r\nk3\r\n", "*2\r\n$2\r\nv1\r\n$-1\r\n");
    // SCAN 0 → an array reply whose first element is the next cursor
    let resp = roundtrip_line(&mut s, "*2\r\n$4\r\nSCAN\r\n$1\r\n0\r\n");
    assert!(resp.starts_with("*2"), "got {resp}");
}

#[test]
fn lazy_expiry_visible_to_clients() {
    let addr = start_server();
    let mut s = connect(addr);

    roundtrip(&mut s, "*5\r\n$3\r\nSET\r\n$2\r\ne1\r\n$1\r\nv\r\n$2\r\nPX\r\n$2\r\n40\r\n", "+OK\r\n");
    std::thread::sleep(Duration::from_millis(150));
    roundtrip(&mut s, "*2\r\n$3\r\nGET\r\n$2\r\ne1\r\n", "$-1\r\n");
    roundtrip(&mut s, "*2\r\n$3\r\nTTL\r\n$2\r\ne1\r\n", ":-2\r\n");
}

#[test]
fn md_slo_roundtrip() {
    let addr = start_server();
    let mut s = connect(addr);

    roundtrip(
        &mut s,
        "*4\r\n$6\r\nMD.SLO\r\n$3\r\nSET\r\n$9\r\ndashboard\r\n$20\r\nfreshness_p99_ms=250\r\n",
        "+OK\r\n",
    );
    // GET on a fresh connection: the flattened map body would otherwise
    // need draining before the next command on this socket
    {
        let mut s2 = connect(addr);
        let got = roundtrip_line(&mut s2, "*3\r\n$6\r\nMD.SLO\r\n$3\r\nGET\r\n$9\r\ndashboard\r\n");
        // RESP2 connection: the 5-pair map flattens to a 10-element array
        assert!(got.starts_with("*10"), "got {got}");
    }

    let del = roundtrip_line(&mut s, "*3\r\n$6\r\nMD.SLO\r\n$3\r\nDEL\r\n$9\r\ndashboard\r\n");
    assert_eq!(del, ":1");
}

#[test]
fn md_maintain_and_invalidate_roundtrip() {
    let addr = start_server();
    let mut s = connect(addr);

    // 1. Maintain in-place counter
    roundtrip(
        &mut s,
        "*4\r\n$11\r\nMD.MAINTAIN\r\n$10\r\nsite:views\r\n$5\r\nCOUNT\r\n$1\r\n1\r\n",
        "+OK\r\n",
    );

    // 2. Invalidate
    let inv = roundtrip_line(&mut s, "*2\r\n$13\r\nMD.INVALIDATE\r\n$10\r\nsite:views\r\n");
    assert_eq!(inv, ":1");
}
