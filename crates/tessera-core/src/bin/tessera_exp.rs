//! CLI Binary for TESSERA-Q Validation Protocols & Griffin Showdown.

use std::env;
use tessera_core::exp_tessera_0::run_exp_tessera_0;
use tessera_core::exp_tessera_suite::{
    run_protocol_1_parameter_matched,
    run_protocol_2_multi_seed,
    run_protocol_3_long_context_recall,
    run_protocol_5_memory_scaling,
    run_protocol_6_wall_clock_profiling,
};
use tessera_core::exp_griffin_benchmark::run_griffin_showdown;

fn main() {
    let args: Vec<String> = env::args().collect();

    let dataset = args.iter().position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("data/enwik8");

    let steps = args.iter().position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(120usize);

    let run_griffin = args.contains(&"--griffin".to_string());
    let run_all = args.contains(&"--all".to_string());
    let run_p1 = args.contains(&"--p1".to_string()) || run_all;
    let run_p2 = args.contains(&"--p2".to_string()) || run_all;
    let run_p3 = args.contains(&"--p3".to_string()) || run_all;
    let run_p5 = args.contains(&"--p5".to_string()) || run_all;
    let run_p6 = args.contains(&"--p6".to_string()) || run_all;

    if run_griffin {
        run_griffin_showdown(dataset, steps);
    } else if args.len() <= 1 {
        run_exp_tessera_0(dataset, steps);
    } else {
        if run_p1 { run_protocol_1_parameter_matched(dataset, steps); }
        if run_p2 { run_protocol_2_multi_seed(dataset, steps); }
        if run_p3 { run_protocol_3_long_context_recall(); }
        if run_p5 { run_protocol_5_memory_scaling(dataset, steps); }
        if run_p6 { run_protocol_6_wall_clock_profiling(); }
    }
}
