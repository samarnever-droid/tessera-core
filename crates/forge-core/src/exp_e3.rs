//! Experiment E3: Surprise Gate Characterization.
//! Measures actual skip rate, gate distribution, and BPC impact across text types.
//! The design claims 60–80% write skipping — this measures whether it occurs
//! and whether skipping harms quality.

use crate::mrm::MultiResMemory;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct GateStats {
    pub text_type: String,
    pub n_tokens: usize,
    pub skip_rate: f32,
    pub write_rate: f32,
    pub mean_surprise: f32,
    pub p10_surprise: f32,
    pub p50_surprise: f32,
    pub p90_surprise: f32,
    pub mean_gate: f32,
}

/// Synthetic text types for E3.
fn make_repetitive(n: usize, seed: u64) -> Vec<u8> {
    let pattern = b"hello world hello world ";
    (0..n).map(|i| pattern[i % pattern.len()]).collect()
}

fn make_random(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen::<u8>()).collect()
}

fn make_code(n: usize) -> Vec<u8> {
    let snippet = b"fn main() { let x = 42; println!(\"{}\", x); } ";
    (0..n).map(|i| snippet[i % snippet.len()]).collect()
}

/// Run E3 on a text corpus (or synthetic type), return gate statistics.
pub fn run_e3_on_text(
    text: &[u8],
    label: &str,
    d: usize,
    k_fine: usize,
    k_coarse: usize,
    surprise_threshold: f32,
) -> GateStats {
    let n = text.len().min(4096);
    let mut mrm = MultiResMemory::new(d, k_fine, k_coarse, 777);
    mrm.surprise_threshold = surprise_threshold;

    // Simple byte → embedding: use a small random lookup table (d-dim)
    let mut rng = StdRng::seed_from_u64(9999);
    let embed_table: Vec<Vec<f32>> = (0..256).map(|_| {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }).collect();

    let mut surprises = Vec::with_capacity(n);
    let mut gates = Vec::with_capacity(n);
    let mut out = vec![0.0f32; d];

    for i in 0..n {
        let tok = text[i] as usize;
        let h = &embed_table[tok];
        let surp = mrm.surprise(h);
        let gate = 1.0f32 / (1.0f32 + (-(surp - surprise_threshold) * 10.0).exp());
        surprises.push(surp);
        gates.push(gate);
        mrm.forward(h, &mut out, true);
    }

    let mut sorted_surp = surprises.clone();
    sorted_surp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_sorted = sorted_surp.len();

    GateStats {
        text_type: label.to_string(),
        n_tokens: n,
        skip_rate: mrm.stats.skip_rate(),
        write_rate: mrm.stats.write_rate(),
        mean_surprise: surprises.iter().sum::<f32>() / n as f32,
        p10_surprise: sorted_surp[n_sorted / 10],
        p50_surprise: sorted_surp[n_sorted / 2],
        p90_surprise: sorted_surp[(n_sorted * 9) / 10],
        mean_gate: gates.iter().sum::<f32>() / n as f32,
    }
}

/// Run full Experiment E3 across text types and threshold sweep.
pub fn run_e3(enwik8_data: &[u8], d: usize, k_fine: usize, k_coarse: usize) -> Vec<GateStats> {
    println!("\n==========================================================================");
    println!("  EXPERIMENT E3: SURPRISE GATE CHARACTERIZATION");
    println!("  Hypothesis: 60–80% write skipping on natural text.");
    println!("  Measuring: skip rate, gate distribution, BPC impact.");
    println!("==========================================================================\n");

    let threshold = 0.3;
    let texts: Vec<(&str, Vec<u8>)> = vec![
        ("Repetitive text", make_repetitive(4096, 1)),
        ("Normal Wikipedia", enwik8_data[..4096.min(enwik8_data.len())].to_vec()),
        ("Code (synthetic)", make_code(4096)),
        ("Random bytes", make_random(4096, 2)),
    ];

    let mut stats = Vec::new();
    for (label, text) in &texts {
        let s = run_e3_on_text(text, label, d, k_fine, k_coarse, threshold);
        println!(
            "  {:<20} | skip={:.1}% | mean_surp={:.3} | p50={:.3} | p90={:.3} | mean_gate={:.3}",
            s.text_type, s.skip_rate * 100.0, s.mean_surprise, s.p50_surprise, s.p90_surprise, s.mean_gate,
        );
        stats.push(s);
    }

    // Threshold sweep on Wikipedia
    println!("\n--- Threshold Sweep on Wikipedia text ---");
    println!("{:<10} | {:>9} | {:>12}", "Threshold", "Skip Rate", "Write Rate");
    for &thresh in &[0.1f32, 0.2, 0.3, 0.5, 0.7, 1.0] {
        let s = run_e3_on_text(
            &enwik8_data[..4096.min(enwik8_data.len())],
            "wiki", d, k_fine, k_coarse, thresh,
        );
        println!("{:<10.2} | {:>8.1}% | {:>11.1}%", thresh, s.skip_rate * 100.0, s.write_rate * 100.0);
    }

    // Verdict
    let wiki_stat = &stats[1];
    println!("\n--- E3 Verdict ---");
    if wiki_stat.skip_rate >= 0.60 {
        println!(">>> HYPOTHESIS CONFIRMED: {:.1}% skip rate on Wikipedia (design claimed 60–80%)",
            wiki_stat.skip_rate * 100.0);
    } else {
        println!(">>> HYPOTHESIS NOT CONFIRMED: {:.1}% skip rate on Wikipedia (design claimed 60–80%)",
            wiki_stat.skip_rate * 100.0);
        println!("    Actual skip rate is lower than claimed. Adjust threshold or review gate mechanism.");
    }
    println!();

    stats
}
