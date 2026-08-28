//! Single-connection pipeline load generator.
//!
//! cargo run --release -p meridian-server --example loadgen -- [addr] [ops] [distinct_keys]
//!
//! Indicative numbers only: this measures the TCP server + loopback path,
//! not the spec §12 in-process engine latencies (those are the Phase 1 gate).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

const BATCH: usize = 128;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let addr = args.first().cloned().unwrap_or_else(|| "127.0.0.1:7717".into());
    let ops: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let distinct: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let distinct = distinct.clamp(1, ops.max(1));

    let mut s = TcpStream::connect(&addr).expect("connect failed (is the server running?)");
    s.set_nodelay(true).unwrap();
    // A framing/accounting bug must surface as an error, never a hang.
    let guard = std::time::Duration::from_secs(10);
    s.set_read_timeout(Some(guard)).unwrap();
    s.set_write_timeout(Some(guard)).unwrap();

    let key = |i: usize| format!("k{:07}", i % distinct);

    // SET phase: round-robin over the distinct keys, pipelined.
    let t = Instant::now();
    let mut done = 0;
    while done < ops {
        let b = BATCH.min(ops - done);
        let mut out = Vec::with_capacity(b * 46);
        for j in 0..b {
            out.extend_from_slice(
                format!("*3\r\n$3\r\nSET\r\n$8\r\n{}\r\n$1\r\nv\r\n", key(done + j)).as_bytes(),
            );
        }
        s.write_all(&out).unwrap();
        let mut buf = vec![0u8; b * 5]; // b × "+OK\r\n"
        s.read_exact(&mut buf).unwrap();
        done += b;
    }
    let set_s = t.elapsed().as_secs_f64();

    // GET phase: same keys, so every reply is a 7-byte hit "$1\r\nv\r\n".
    let t = Instant::now();
    let mut done = 0;
    let mut missed = false;
    while done < ops {
        let b = BATCH.min(ops - done);
        let mut out = Vec::with_capacity(b * 38);
        for j in 0..b {
            out.extend_from_slice(
                format!("*2\r\n$3\r\nGET\r\n$8\r\n{}\r\n", key(done + j)).as_bytes(),
            );
        }
        s.write_all(&out).unwrap();
        let mut buf = vec![0u8; b * 7];
        s.read_exact(&mut buf).unwrap();
        if buf.windows(2).any(|w| w == b"$-".as_slice()) {
            missed = true;
            break;
        }
        done += b;
    }
    let get_s = t.elapsed().as_secs_f64();

    // Sequential PING RTT: one write + one read per op, no pipelining.
    let rtt_n = 2_000;
    let t = Instant::now();
    for _ in 0..rtt_n {
        s.write_all(b"*1\r\n$4\r\nPING\r\n").unwrap();
        let mut buf = [0u8; 6];
        s.read_exact(&mut buf).unwrap();
    }
    let rtt_us = t.elapsed().as_secs_f64() / rtt_n as f64 * 1e6;

    println!("loadgen → {addr}");
    println!(
        "SET   {:>8} ops in {:.3}s → {:>10.0} ops/s   (pipelined ×{BATCH}, {distinct} distinct keys)",
        ops, set_s, ops as f64 / set_s
    );
    if missed {
        println!("GET   {:>8} ops in {:.3}s → {:>10.0} ops/s   (STOPPED: misses observed — working set exceeded capacity)", done, get_s, done as f64 / get_s);
    } else {
        println!(
            "GET   {:>8} ops in {:.3}s → {:>10.0} ops/s   (all hits)",
            ops, get_s, ops as f64 / get_s
        );
    }
    println!("PING  sequential RTT over {rtt_n} ops → {rtt_us:.1} µs/op");
    println!("(single TCP connection, loopback; not comparable to the spec's in-process ns targets)");
}
