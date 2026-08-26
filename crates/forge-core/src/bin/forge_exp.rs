//! FORGE Experiment Runner: orchestrates E1 → E2 → E3 → E4 → E5 → E6
//! in the correct sequence, sharing trained models across experiments to avoid
//! redundant computation.

use forge_core::exp_e1::run_e1;
use forge_core::exp_e2::run_e2;
use forge_core::exp_e3::run_e3;
use forge_core::exp_e4::run_e4;
use forge_core::exp_e5_e6::{run_e5, run_e6};
use forge_core::forge_model::{ForgeConfig, ForgeModel};
use forge_core::forge_trainer::{evaluate_forge_bpc, train_forge};
use axiom_baseline::transformer::TransformerModel;
use axiom_baseline::transformer_trainer::{evaluate_transformer_bpc, train_transformer};
use axiom_train::dataset::CharDataset;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let dataset = args.iter().position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("data/enwik8");

    let steps = args.iter().position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(100usize);

    let run_e1_flag = args.contains(&"--e1".to_string()) || args.contains(&"--all".to_string());
    let run_e2_flag = args.contains(&"--e2".to_string()) || args.contains(&"--all".to_string());
    let run_e3_flag = args.contains(&"--e3".to_string()) || args.contains(&"--all".to_string());
    let run_e4_flag = args.contains(&"--e4".to_string()) || args.contains(&"--all".to_string());
    let run_e5_flag = args.contains(&"--e5".to_string()) || args.contains(&"--all".to_string());

    println!("==========================================================================");
    println!("  FORGE PRE-BUILD FALSIFICATION SUITE");
    println!("  Dataset: {} | Steps: {} | Experiments: {:?}",
        dataset, steps,
        args.iter().filter(|a| a.starts_with("--e") || **a == "--all").collect::<Vec<_>>()
    );
    println!("==========================================================================");

    // Shared config
    let vocab_size = 256usize;
    let d_model    = 128usize;
    let d_ff       = 256usize;
    let n_blocks   = 2usize;
    let k_fine     = 64usize;
    let k_coarse   = 16usize;
    let max_seq    = 128usize;
    let batch_size = 32usize;
    let seq_len    = 64usize;
    let base_lr    = 3e-3_f32;
    let max_time   = 240.0_f64;

    // ── E1: Ablation ───────────────────────────────────────────────────
    let e1_results = if run_e1_flag {
        run_e1(dataset, steps)
    } else {
        println!("[skip] E1 (pass --e1 or --all)");
        vec![]
    };

    // ── E2: Recall ────────────────────────────────────────────────────
    if run_e2_flag {
        run_e2(d_model, k_fine, k_coarse, 30);
    } else {
        println!("[skip] E2 (pass --e2 or --all)");
    }

    // ── E3: Surprise Gate ─────────────────────────────────────────────
    if run_e3_flag {
        let raw = CharDataset::from_file(dataset).expect("E3: dataset load");
        run_e3(&raw.data, d_model, k_fine, k_coarse);
    } else {
        println!("[skip] E3 (pass --e3 or --all)");
    }

    // ── E4 + E5 + E6: need a trained FORGE Full model ────────────────
    if run_e4_flag || run_e5_flag {
        println!("\n[Shared] Training Full FORGE model for E4/E5/E6...");
        let raw = CharDataset::from_file(dataset).expect("E4/5/6: dataset");
        let (train_ds, val_ds) = raw.split(0.9);

        // Train Full FORGE
        let mut forge_full = ForgeModel::new(
            vocab_size, d_model, d_ff, n_blocks, k_fine, k_coarse, max_seq,
            ForgeConfig::full(), 42,
        );
        let forge_result = train_forge(
            &mut forge_full, &train_ds.data, &val_ds.data,
            batch_size, seq_len, steps, max_time, base_lr,
            "E-Full FORGE",
        );

        // Train Transformer (if not already done in E1)
        let mut transformer = TransformerModel::new(vocab_size, d_model, n_blocks, d_ff, max_seq, 42);
        let t_history = train_transformer(
            &mut transformer, &train_ds.data, &val_ds.data,
            batch_size, seq_len, max_time as u64, steps, base_lr,
        );
        let (t_loss, t_bpc) = evaluate_transformer_bpc(&transformer, &val_ds.data, 20, seq_len);
        let t_flops: f64 = {
            let fpt = 2.0 * n_blocks as f64 * (4.0 * d_model as f64 * d_model as f64 + 2.0 * d_model as f64 * seq_len as f64);
            fpt * (steps * batch_size * seq_len) as f64 * 3.0
        };

        // ── E4 ──────────────────────────────────────────────────────
        if run_e4_flag {
            run_e4(&train_ds.data, &forge_full, &[0, 1, 4, 8, 16, 32]);
        }

        // ── E5 ──────────────────────────────────────────────────────
        if run_e5_flag {
            let ctx_lengths = [1024usize, 2048, 4096, 8192, 16384];
            run_e5(&transformer, &mut forge_full, &ctx_lengths, 20);

            // ── E6 (always paired with E5) ──────────────────────────
            let train_tokens = steps * batch_size * seq_len;
            let (f_loss, f_bpc, _) = evaluate_forge_bpc(&mut forge_full, &val_ds.data, 20, seq_len, 999);
            run_e6(
                forge_result.total_train_flops,
                t_flops,
                f_bpc,
                t_bpc,
                train_tokens,
            );
        }
    }

    println!("==========================================================================");
    println!("  FORGE FALSIFICATION SUITE COMPLETE");
    println!("==========================================================================");
}
