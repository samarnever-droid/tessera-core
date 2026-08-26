//! Unified Runner for the MNEME Pre-Build Falsification Protocol (Experiments E1 - E7).

use mneme_core::exp_e1::run_e1;
use mneme_core::exp_e2::run_e2;
use mneme_core::exp_e3::run_e3;
use mneme_core::exp_e4::run_e4;
use mneme_core::exp_e5::run_e5;
use mneme_core::exp_e6::run_e6;
use mneme_core::exp_e7::run_e7;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let dataset = args.iter().position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("data/enwik8");

    let steps = args.iter().position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(120usize);

    let run_all = args.contains(&"--all".to_string()) || args.len() <= 1;
    let run_e1_flag = args.contains(&"--e1".to_string()) || run_all;
    let run_e2_flag = args.contains(&"--e2".to_string()) || run_all;
    let run_e3_flag = args.contains(&"--e3".to_string()) || run_all;
    let run_e4_flag = args.contains(&"--e4".to_string()) || run_all;
    let run_e5_flag = args.contains(&"--e5".to_string()) || run_all;
    let run_e6_flag = args.contains(&"--e6".to_string()) || run_all;
    let run_e7_flag = args.contains(&"--e7".to_string()) || run_all;

    println!("==========================================================================");
    println!("  MNEME PRE-BUILD FALSIFICATION & GATE-2 PROTOCOL RUNNER");
    println!("  Dataset: {} | Steps per arm: {}", dataset, steps);
    println!("==========================================================================\n");

    let mut e1_results = Vec::new();
    let mut e2_results = Vec::new();
    let mut e4_results = Vec::new();
    let mut e5_results = Vec::new();
    let mut e7_results = Vec::new();

    // ── E1: Frozen Control / MNEME-Nano ──────────────────────────────
    if run_e1_flag {
        e1_results = run_e1(dataset, steps);
    }

    // ── E2: The Knowledge / Computation Wall ─────────────────────────
    if run_e2_flag {
        e2_results = run_e2(dataset, steps);
    }

    // ── E3: Residency / Cache Wall ───────────────────────────────────
    if run_e3_flag {
        let _ = run_e3();
    }

    // ── E4: Recurrent Depth Reuse ────────────────────────────────────
    if run_e4_flag {
        e4_results = run_e4(dataset, steps);
    }

    // ── E5: Gated Delta-Rule Ablation ────────────────────────────────
    if run_e5_flag {
        e5_results = run_e5(dataset, steps);
    }

    // ── E6: Routing Stability Audit ──────────────────────────────────
    if run_e6_flag {
        let _ = run_e6(dataset, steps);
    }

    // ── E7: Quantization Cascade ─────────────────────────────────────
    if run_e7_flag {
        e7_results = run_e7(dataset, steps);
    }

    // ── Final Executive Verdict Evaluation ───────────────────────────
    println!("==========================================================================");
    println!("  MNEME FALSIFICATION PROTOCOL: EXECUTIVE GATE EVALUATION");
    println!("==========================================================================\n");

    // Primary Criterion: Loss vs Inference Bytes/Token
    let mut byte_win = false;
    if !e1_results.is_empty() {
        let t_bytes = e1_results[0].dram_bytes_per_token;
        let m_bytes = e1_results[3].dram_bytes_per_token;
        let t_bpc = e1_results[0].val_bpc;
        let m_bpc = e1_results[3].val_bpc;
        let ratio = t_bytes as f32 / m_bytes.max(1) as f32;

        println!("1. Primary Frontier: LOSS vs INFERENCE BYTES/TOKEN");
        println!("   Dense Transformer: {:.4} BPC @ {} B/tok", t_bpc, t_bytes);
        println!("   Full MNEME-Nano:   {:.4} BPC @ {} B/tok", m_bpc, m_bytes);
        println!("   Inference DRAM Byte Reduction: {:.1}x", ratio);

        if ratio >= 8.0 && (m_bpc - t_bpc) <= 0.35 {
            println!("   >>> GATE STATUS: PASS (Decisive >8x byte win with matched/competitive quality)");
            byte_win = true;
        } else {
            println!("   >>> GATE STATUS: FAIL (Insufficient byte reduction or uncompetitive quality)");
        }
    }

    // Secondary Criteria Check
    let mut k1_pass = true;
    if !e2_results.is_empty() {
        let delta = e2_results[0].val_bpc - e2_results.last().unwrap().val_bpc;
        println!("\n2. Kill Condition 1 (Knowledge Tier Scaling):");
        println!("   Quality improvement from scaling sparse experts: {:+.4} BPC", delta);
        if delta < 0.05 {
            println!("   >>> WARNING: Sparse knowledge tier scaling buys marginal quality.");
            k1_pass = false;
        } else {
            println!("   >>> PASS: Knowledge tier scaling is functional.");
        }
    }

    let mut k3_pass = true;
    if !e4_results.is_empty() {
        let delta_r = e4_results[0].val_bpc - e4_results[2].val_bpc; // R=1 vs R=4
        println!("\n3. Kill Condition 3 (Depth Reuse Knob):");
        println!("   R=1 to R=4 Quality Improvement: {:+.4} BPC (at 0 extra DRAM bytes)", delta_r);
        if delta_r < 0.05 {
            println!("   >>> WARNING: Depth reuse knob produces negligible gain.");
            k3_pass = false;
        } else {
            println!("   >>> PASS: Depth reuse operates as an effective runtime quality dial.");
        }
    }

    let mut k4_pass = true;
    if !e5_results.is_empty() {
        let delta_mixer = e5_results[0].val_bpc - e5_results.last().unwrap().val_bpc;
        println!("\n4. Kill Condition 4 (Gated Delta-Rule):");
        println!("   Gated Delta-Rule vs State-Free Mixer: {:+.4} BPC", delta_mixer);
        if delta_mixer < 0.05 {
            println!("   >>> WARNING: Delta rule does not contribute meaningful quality.");
            k4_pass = false;
        } else {
            println!("   >>> PASS: Gated delta rule with erasure provides clear quality boost.");
        }
    }

    let mut k5_pass = true;
    if !e7_results.is_empty() {
        let w4_gap = e7_results[3].bpc_r4 - e7_results[0].bpc_r4;
        println!("\n5. Kill Condition 5 (W4 Quantization Stability under Recurrence):");
        println!("   W4 vs FP32 degradation at R=4: {:+.4} BPC", w4_gap);
        if w4_gap > 0.20 {
            println!("   >>> WARNING: W4 quantization suffers degradation under depth recurrence.");
            k5_pass = false;
        } else {
            println!("   >>> PASS: Quantization error does not catastrophically compound.");
        }
    }

    println!("\n==========================================================================");
    let final_verdict = if byte_win && k1_pass && k3_pass && k4_pass && k5_pass {
        "PASS"
    } else if byte_win {
        "PARTIAL PASS"
    } else {
        "FAIL"
    };

    println!("  FINAL MNEME VERDICT: {}", final_verdict);
    println!("==========================================================================");
}
