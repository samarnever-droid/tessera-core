use stratum_core::f4_experiment::run_f4_experiment;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let dataset_path = args
        .iter()
        .position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/enwik8");

    let steps_per_model = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(150);

    println!("Starting STRATUM Experiment F4 Runner on dataset '{}' with {} steps/model...", dataset_path, steps_per_model);
    let _ = run_f4_experiment(dataset_path, steps_per_model);
}
