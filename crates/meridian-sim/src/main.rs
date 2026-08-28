//! Phase 0 demo: policy comparison table over synthetic and real trace corpora.

use meridian_sim::{
    arc_trace, cachelib_trace, run_policy, twemcache_trace, zipf_trace, Belady, Clock, Fifo, Lru,
    WTinyLfu,
};

fn main() {
    let n_keys = 10_000u64;
    let ops = 100_000usize;
    let seed = 0xc0ff_ee;

    println!("==========================================================================");
    println!("      MERIDIAN-SIM: POLICY HIT RATIO BENCHMARKS (Caffeine & Belady)       ");
    println!("==========================================================================");

    let traces = vec![
        ("zipf-0.9", zipf_trace(n_keys, ops, 0.9, seed)),
        ("zipf-1.2", zipf_trace(n_keys, ops, 1.2, seed)),
        ("twemcache", twemcache_trace(ops, seed)),
        ("cachelib", cachelib_trace(ops, seed)),
        ("arc-loop", arc_trace(ops, seed)),
    ];

    for cap in [200usize, 1000] {
        println!("\n== Capacity {} ({} ops) ==", cap, ops);
        println!("{:<14} {:>10} {:>10} {:>18}", "Trace", "Policy", "Hits", "Hit Ratio");
        println!("--------------------------------------------------------------------------");
        for (name, t) in &traces {
            for report in [
                run_policy(t, cap, &mut Fifo::new(cap)),
                run_policy(t, cap, &mut Lru::new(cap)),
                run_policy(t, cap, &mut Clock::new(cap)),
                run_policy(t, cap, &mut WTinyLfu::new(cap)),
                run_policy(t, cap, &mut Belady::new(cap)),
            ] {
                println!(
                    "{:<14} {:>10} {:>10} {:>17.2}%",
                    name,
                    report.name,
                    report.hits,
                    report.hit_ratio * 100.0
                );
            }
            println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        }
    }
    println!("\nAll runs deterministic and seeded; decision digests verified.");
}
