//! meridian-server — thread-per-connection TCP server speaking RESP2/RESP3.
//!
//! v0 uses blocking std::net with one thread per connection; the sharded,
//! lock-free-read engine means connections do not contend on the hot path.
//! The Phase 4 event loop and batching replace this without protocol changes.

pub mod commands;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use meridian_core::Engine;
use meridian_proto::Decoder;

pub fn serve_stream(engine: Arc<Engine>, mut stream: TcpStream) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut dec = Decoder::new();
    let mut out = Vec::with_capacity(4096);
    let mut proto3 = false;
    let mut chunk = [0u8; 16384];
    loop {
        // Drain every buffered command, accumulating replies into one write:
        // a pipelined batch of N commands costs one read and one send.
        let mut dirty = false;
        loop {
            match dec.try_command() {
                Ok(Some(cmd)) => {
                    dirty = true;
                    match commands::dispatch(&engine, &mut proto3, cmd) {
                        commands::Action::Reply(f) => f.encode(&mut out, proto3),
                        commands::Action::Quit => {
                            out.extend_from_slice(b"+OK\r\n");
                            let _ = stream.write_all(&out);
                            return Ok(());
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    out.extend_from_slice(b"-ERR protocol error\r\n");
                    let _ = stream.write_all(&out);
                    return Ok(());
                }
            }
        }
        if dirty {
            stream.write_all(&out)?;
            out.clear();
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Ok(());
        }
        dec.feed(&chunk[..n]);
    }
}

pub fn serve(engine: Arc<Engine>, listener: std::net::TcpListener) -> std::io::Result<()> {
    for socket in listener.incoming() {
        let socket = socket?;
        let engine = engine.clone();
        std::thread::spawn(move || {
            let _ = serve_stream(engine, socket);
        });
    }
    Ok(())
}
