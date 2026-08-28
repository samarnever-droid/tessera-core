use std::sync::Arc;
use std::time::Duration;

use meridian_core::{Engine, EngineOptions};
use meridian_server::serve;

struct Args {
    host: String,
    port: u16,
    shards: Option<usize>,
    cores: Option<usize>,
    entries: usize,
    min_buckets: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        host: "127.0.0.1".into(),
        port: 7717,
        shards: None,
        cores: None,
        entries: 1 << 20,
        min_buckets: 64,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let val = it.next();
        match (flag.as_str(), val) {
            ("--host", Some(v)) => a.host = v,
            ("--port", Some(v)) => a.port = v.parse().unwrap_or(7717),
            ("--shards", Some(v)) => a.shards = v.parse().ok(),
            ("--cores", Some(v)) => a.cores = v.parse().ok(),
            ("--entries", Some(v)) => a.entries = v.parse().unwrap_or(1 << 20),
            ("--min-buckets", Some(v)) => a.min_buckets = v.parse().unwrap_or(64),
            _ => {}
        }
    }
    a
}

fn main() -> std::io::Result<()> {
    let a = parse_args();
    let engine = Arc::new(Engine::new(EngineOptions {
        shard_hint: a.shards,
        cores: a.cores,
        total_entries: a.entries,
        min_buckets: a.min_buckets,
        ..Default::default()
    }));

    {
        let engine = engine.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(10));
            engine.sweep();
        });
    }

    let listener = std::net::TcpListener::bind((a.host.as_str(), a.port))?;
    let st = engine.stats();
    println!("meridian {} — HELIOS v5 MERIDIAN vertical slice (phases 0–1)", env!("CARGO_PKG_VERSION"));
    println!(
        "shards={} ways={} probe_limit={} items_capacity~{}",
        st.shards,
        meridian_core::WAYS,
        meridian_core::PROBE_LIMIT,
        a.entries
    );
    println!("listening on {}:{}", a.host, a.port);

    serve(engine, listener)
}
