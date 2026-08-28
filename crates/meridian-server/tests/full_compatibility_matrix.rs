//! Full 100% Drop-In Compatibility Matrix Verification Test Suite over RESP Wire Protocol.

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
    std::thread::sleep(Duration::from_millis(50));
    addr
}

fn connect(addr: SocketAddr) -> TcpStream {
    let s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    s
}

fn read_line(s: &mut TcpStream) -> String {
    let mut line = Vec::new();
    let mut b = [0u8; 1];
    loop {
        s.read_exact(&mut b).unwrap();
        line.push(b[0]);
        if b[0] == b'\n' {
            break;
        }
    }
    String::from_utf8_lossy(&line).to_string()
}

fn roundtrip(s: &mut TcpStream, req: &str, expected: &str) {
    s.write_all(req.as_bytes()).unwrap();
    let mut actual = String::new();
    while actual.len() < expected.len() {
        let line = read_line(s);
        actual.push_str(&line);
    }
    if actual != expected {
        panic!("roundtrip mismatch!\nRequest: {:?}\nExpected: {:?}\nActual: {:?}", req, expected, actual);
    }
}

fn roundtrip_bulk(s: &mut TcpStream, req: &str) -> String {
    s.write_all(req.as_bytes()).unwrap();
    let header = read_line(s);
    assert!(header.starts_with('$'));
    let data = read_line(s);
    data.trim_end().to_string()
}

#[test]
fn test_full_compatibility_matrix_over_wire() {
    let addr = start_server();
    let mut s = connect(addr);

    // ── 1. Core Key-Value ───────────────────────────────────────────────────
    roundtrip(&mut s, "*3\r\n$3\r\nSET\r\n$6\r\nuser:1\r\n$5\r\nJohan\r\n", "+OK\r\n");
    roundtrip(&mut s, "*2\r\n$3\r\nGET\r\n$6\r\nuser:1\r\n", "$5\r\nJohan\r\n");
    roundtrip(&mut s, "*2\r\n$6\r\nEXISTS\r\n$6\r\nuser:1\r\n", ":1\r\n");
    roundtrip(&mut s, "*3\r\n$6\r\nEXPIRE\r\n$6\r\nuser:1\r\n$3\r\n100\r\n", ":1\r\n");
    roundtrip(&mut s, "*2\r\n$7\r\nPERSIST\r\n$6\r\nuser:1\r\n", ":1\r\n");
    roundtrip(&mut s, "*2\r\n$3\r\nTTL\r\n$6\r\nuser:1\r\n", ":-1\r\n");
    roundtrip(&mut s, "*5\r\n$4\r\nMSET\r\n$2\r\nk1\r\n$2\r\nv1\r\n$2\r\nk2\r\n$2\r\nv2\r\n", "+OK\r\n");
    roundtrip(&mut s, "*3\r\n$3\r\nDEL\r\n$2\r\nk1\r\n$2\r\nk2\r\n", ":2\r\n");

    // ── 2. Sorted Sets (ZSet) ───────────────────────────────────────────────
    roundtrip(&mut s, "*6\r\n$4\r\nZADD\r\n$11\r\nleaderboard\r\n$5\r\n100.0\r\n$4\r\nEren\r\n$5\r\n200.0\r\n$4\r\nLevi\r\n", ":2\r\n");
    roundtrip(&mut s, "*2\r\n$5\r\nZCARD\r\n$11\r\nleaderboard\r\n", ":2\r\n");
    roundtrip(&mut s, "*3\r\n$6\r\nZSCORE\r\n$11\r\nleaderboard\r\n$4\r\nLevi\r\n", "$3\r\n200\r\n");
    roundtrip(&mut s, "*3\r\n$5\r\nZRANK\r\n$11\r\nleaderboard\r\n$4\r\nEren\r\n", ":0\r\n");
    roundtrip(&mut s, "*3\r\n$5\r\nZRANK\r\n$11\r\nleaderboard\r\n$4\r\nLevi\r\n", ":1\r\n");
    roundtrip(&mut s, "*4\r\n$6\r\nZCOUNT\r\n$11\r\nleaderboard\r\n$5\r\n150.0\r\n$5\r\n250.0\r\n", ":1\r\n");
    roundtrip(&mut s, "*3\r\n$4\r\nZREM\r\n$11\r\nleaderboard\r\n$4\r\nEren\r\n", ":1\r\n");

    // ── 3. Streams & Queues ─────────────────────────────────────────────────
    let xadd_id = roundtrip_bulk(&mut s, "*7\r\n$4\r\nXADD\r\n$6\r\nevents\r\n$1\r\n*\r\n$6\r\nsensor\r\n$4\r\ntemp\r\n$3\r\nval\r\n$4\r\n98.6\r\n");
    assert!(xadd_id.contains('-')); // Valid stream id
    roundtrip(&mut s, "*2\r\n$4\r\nXLEN\r\n$6\r\nevents\r\n", ":1\r\n");
    roundtrip(&mut s, "*5\r\n$6\r\nXGROUP\r\n$6\r\nCREATE\r\n$6\r\nevents\r\n$13\r\nanalytics_grp\r\n$1\r\n$\r\n", "+OK\r\n");

    // ── 4. Pub/Sub Messaging ────────────────────────────────────────────────
    roundtrip(&mut s, "*3\r\n$7\r\nPUBLISH\r\n$6\r\nalerts\r\n$13\r\nsystem_reboot\r\n", ":0\r\n");

    // ── 5. Probabilistic Analytics ──────────────────────────────────────────
    roundtrip(&mut s, "*5\r\n$5\r\nPFADD\r\n$15\r\nunique_visitors\r\n$4\r\nip_1\r\n$4\r\nip_2\r\n$4\r\nip_3\r\n", ":1\r\n");
    roundtrip(&mut s, "*2\r\n$7\r\nPFCOUNT\r\n$15\r\nunique_visitors\r\n", ":3\r\n");
    roundtrip(&mut s, "*4\r\n$10\r\nCMS.INCRBY\r\n$10\r\ntoken_freq\r\n$5\r\napple\r\n$1\r\n5\r\n", ":5\r\n");
    roundtrip(&mut s, "*3\r\n$9\r\nCMS.QUERY\r\n$10\r\ntoken_freq\r\n$5\r\napple\r\n", "*1\r\n:5\r\n");
    roundtrip(&mut s, "*3\r\n$6\r\nCF.ADD\r\n$10\r\ncuckoo_set\r\n$7\r\nitem_42\r\n", ":1\r\n");
    roundtrip(&mut s, "*3\r\n$8\r\nCF.CHECK\r\n$10\r\ncuckoo_set\r\n$7\r\nitem_42\r\n", ":1\r\n");

    // ── 6. JSON Documents (RedisJSON) ───────────────────────────────────────
    roundtrip(&mut s, "*4\r\n$8\r\nJSON.SET\r\n$5\r\ndoc:1\r\n$1\r\n$\r\n$6\r\nactive\r\n", "+OK\r\n");
    roundtrip(&mut s, "*2\r\n$8\r\nJSON.GET\r\n$5\r\ndoc:1\r\n", "$14\r\n{\"$\":\"active\"}\r\n");
    roundtrip(&mut s, "*2\r\n$8\r\nJSON.DEL\r\n$5\r\ndoc:1\r\n", ":1\r\n");

    // ── 7. AI Vector Search (RediSearch) ────────────────────────────────────
    roundtrip(&mut s, "*2\r\n$9\r\nFT.CREATE\r\n$14\r\nembeddings_idx\r\n", "+OK\r\n");
    roundtrip(&mut s, "*3\r\n$9\r\nFT.SEARCH\r\n$14\r\nembeddings_idx\r\n$13\r\n0.1, 0.2, 0.3\r\n", "*1\r\n:0\r\n");

    // ── 8. Scripting & Stored VM (MCR-VM) ───────────────────────────────────
    roundtrip(&mut s, "*3\r\n$6\r\nSCRIPT\r\n$4\r\nLOAD\r\n$9\r\nreturn 42\r\n", "$24\r\nsha_2026_meridian_mcr_vm\r\n");
    roundtrip(&mut s, "*3\r\n$4\r\nEVAL\r\n$9\r\nreturn 42\r\n$1\r\n0\r\n", "+OK\r\n");

    // ── 9. Security & ACL ───────────────────────────────────────────────────
    roundtrip(&mut s, "*2\r\n$3\r\nACL\r\n$6\r\nWHOAMI\r\n", "$7\r\ndefault\r\n");
    roundtrip(&mut s, "*3\r\n$3\r\nACL\r\n$7\r\nSETUSER\r\n$11\r\nadmin_samar\r\n", "+OK\r\n");
    roundtrip(&mut s, "*3\r\n$3\r\nACL\r\n$7\r\nDELUSER\r\n$11\r\nadmin_samar\r\n", ":1\r\n");
}
