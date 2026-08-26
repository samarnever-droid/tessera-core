//! Progressive Layer Freezing Diagnostic Trainer.
//! Trains Layer 1 to convergence on (H_0, y), freezes Layer 1, then trains Layer 2 on (H_1, y).

use crate::dataset::CharDataset;
use crate::optimizer::{AdamW, AdamWConfig};
use crate::LocalLossConfig;
use axiom_core::tensor::MatrixView;
use axiom_model::layer::{LayerScratch, SequenceCache};
use axiom_model::model::{AxiomSingleLayerModel, ModelGrads};
use axiom_model::stacked_model::AxiomModel;
use axiom_model::LayerState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

/// Run the Progressive Layer Freezing diagnostic.
/// Returns (Layer 1 Final Val BPC, Layer 2 Final Val BPC).
pub fn run_freezing_diagnostic(
    dataset_path: &str,
    steps_per_layer: usize,
) -> (f32, f32) {
    println!("=== Progressive Layer Freezing Diagnostic (Corpus: {}) ===", dataset_path);
    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    let config = axiom_model::AxiomConfig {
        vocab_size: 256,
        d_model: 128,
        num_layers: 2,
        num_experts: 8,
        active_experts: 2,
        buffer_capacity: 512,
        d_ffn: 512,
        hebbian_decay: 0.999,
        hebbian_lr: 1e-4,
    };

    let mut model = AxiomModel::new(config.clone(), 1024, 42);
    let batch_size = 32;
    let seq_len = 64;
    let d = config.d_model;

    // ==========================================
    // STAGE 1: Train Layer 1 to Convergence
    // ==========================================
    println!("\n--- Stage 1: Training Layer 1 (Embeddings + Layer 1) ---");
    let mut single_layer_model = AxiomSingleLayerModel {
        config: config.clone(),
        max_seq_len: 1024,
        embeddings: model.embeddings.clone(),
        pos_embeddings: model.pos_embeddings.clone(),
        layer: model.layers[0].clone(),
    };

    let trainer_config = crate::trainer::TrainerConfig {
        batch_size,
        seq_len,
        max_steps: steps_per_layer,
        max_time_secs: 150,
        base_lr: 5e-3,
        min_lr: 1e-4,
        warmup_steps: 25,
        eval_interval: 25,
        loss_weights: LocalLossConfig::default(),
    };

    let history_l1 = crate::trainer::train_single_layer_bptt(
        &mut single_layer_model,
        &train_data,
        &val_data,
        &trainer_config,
    );

    let final_l1_bpc = history_l1.last().map(|&(_, _, bpc, _)| bpc).unwrap_or(99.0);
    println!(">>> Stage 1 Finished | Layer 1 Final Val BPC: {:.4}", final_l1_bpc);

    // Save frozen Layer 1 and embeddings back to model
    model.embeddings = single_layer_model.embeddings.clone();
    model.pos_embeddings = single_layer_model.pos_embeddings.clone();
    model.layers[0] = single_layer_model.layer.clone();

    // ==========================================
    // STAGE 2: Freeze Layer 1, Train Layer 2 on H_1
    // ==========================================
    println!("\n--- Stage 2: Freezing Layer 1, Training Layer 2 on H_1 ---");
    let frozen_model = model.clone();
    let mut l2_optimizer = AdamW::new(
        AdamWConfig {
            lr: 5e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        },
        &single_layer_model,
    );

    let start_time_l2 = Instant::now();
    let mut master_rng = StdRng::seed_from_u64(1042);
    let mut final_l2_bpc = 99.0f32;

    for step in 1..=steps_per_layer {
        let (batch_x, batch_y) = train_data.sample_batch(batch_size, seq_len, &mut master_rng);

        let frozen_ref = &frozen_model;
        let l2_ref = &model.layers[1];
        let loss_weights = LocalLossConfig::default();

        let thread_results: Vec<(f32, axiom_model::layer::LayerGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                // 1. Pass through frozen Layer 0 to generate static representation H_1
                let mut state0 = LayerState::new(&frozen_ref.config);
                let mut scratch0 = LayerScratch::new(&frozen_ref.config);
                let mut h_1_seq = vec![0.0f32; seq_len * d];
                let mut h_in = vec![0.0f32; d];
                let mut h_out = vec![0.0f32; d];

                for t in 0..seq_len {
                    frozen_ref.embed_token_pos(x_seq[t], t, &mut h_in);
                    frozen_ref.layers[0].forward_infer(&h_in, &mut state0, &mut scratch0, &mut h_out);
                    h_1_seq[t * d..(t + 1) * d].copy_from_slice(&h_out);
                }

                // 2. Forward through trainable Layer 1 (Layer 2 of model)
                let mut state1 = LayerState::new(&frozen_ref.config);
                let mut scratch1 = LayerScratch::new(&frozen_ref.config);
                let mut cache1 = SequenceCache::new(&frozen_ref.config, seq_len);
                let mut dummy_grads = ModelGrads::new(&frozen_ref.config, 1024);

                let dummy_single = AxiomSingleLayerModel {
                    config: frozen_ref.config.clone(),
                    max_seq_len: 1024,
                    embeddings: vec![0.0f32; frozen_ref.config.vocab_size * d],
                    pos_embeddings: vec![0.0f32; 1024 * d],
                    layer: l2_ref.clone(),
                };

                let (lp, _) = dummy_single.forward_sequence_cache(
                    &x_seq,
                    &y_seq,
                    &mut state1,
                    &mut scratch1,
                    &mut cache1,
                );

                // Override cache1 h_in with actual H_1 from frozen Layer 0
                cache1.h_in_history.copy_from_slice(&h_1_seq);

                dummy_single.backward_sequence_bptt(
                    &x_seq,
                    &cache1,
                    loss_weights.lambda_pred,
                    loss_weights.lambda_recon,
                    loss_weights.lambda_residual,
                    &mut scratch1,
                    &mut dummy_grads,
                );

                (lp, dummy_grads.layer_grads)
            })
            .collect();

        // Aggregate gradients for Layer 2 only
        let mut total_l2_grads = axiom_model::layer::LayerGrads::new(&config);
        let mut batch_loss = 0.0f32;
        let scale = 1.0f32 / (batch_size * seq_len) as f32;

        for (lp, lg) in thread_results {
            batch_loss += lp;
            total_l2_grads.add(&lg);
        }

        axiom_core::tensor::vec_scale(&mut total_l2_grads.grad_w_s, scale);
        axiom_core::tensor::vec_scale(&mut total_l2_grads.grad_w_gate, scale);
        for eg in &mut total_l2_grads.expert_grads {
            axiom_core::tensor::vec_scale(&mut eg.grad_w_up, scale);
            axiom_core::tensor::vec_scale(&mut eg.grad_w_down, scale);
        }
        axiom_core::tensor::vec_scale(&mut total_l2_grads.grad_w_pred, scale);
        axiom_core::tensor::vec_scale(&mut total_l2_grads.grad_w_decode, scale);

        // Update Layer 2
        let current_lr = if step < 25 {
            let alpha = step as f32 / 25.0f32;
            1e-4 + alpha * (5e-3 - 1e-4)
        } else {
            let prog = (step - 25) as f32 / (steps_per_layer - 25) as f32;
            1e-4 + 0.5 * (1.0 + (std::f32::consts::PI * prog.min(1.0)).cos()) * (5e-3 - 1e-4)
        };

        // Apply optimizer to Layer 2
        let mut dummy_single = AxiomSingleLayerModel {
            config: config.clone(),
            max_seq_len: 1024,
            embeddings: vec![0.0f32; config.vocab_size * d],
            pos_embeddings: vec![0.0f32; 1024 * d],
            layer: model.layers[1].clone(),
        };
        let mut dummy_model_grads = ModelGrads::new(&config, 1024);
        dummy_model_grads.layer_grads = total_l2_grads;

        l2_optimizer.step(&mut dummy_single, &mut dummy_model_grads, current_lr);
        model.layers[1] = dummy_single.layer;

        if step % 25 == 0 || step == steps_per_layer {
            // Evaluate Layer 2 on validation set using frozen Layer 0
            let mut val_loss = 0.0f32;
            let mut val_tokens = 0usize;
            let mut val_rng = StdRng::seed_from_u64(999);

            for _ in 0..10 {
                let (inputs, targets) = val_data.sample_batch(1, seq_len, &mut val_rng);
                let x_seq = &inputs[0];
                let y_seq = &targets[0];

                let mut s0 = LayerState::new(&config);
                let mut sc0 = LayerScratch::new(&config);
                let mut s1 = LayerState::new(&config);
                let mut sc1 = LayerScratch::new(&config);

                let mut h_in = vec![0.0f32; d];
                let mut h_1 = vec![0.0f32; d];
                let mut h_2 = vec![0.0f32; d];
                let mut logits = vec![0.0f32; config.vocab_size];
                let mut probs = vec![0.0f32; config.vocab_size];
                let mut dummy_grad = vec![0.0f32; config.vocab_size];

                for t in 0..seq_len {
                    frozen_model.embed_token_pos(x_seq[t], t, &mut h_in);
                    frozen_model.layers[0].forward_infer(&h_in, &mut s0, &mut sc0, &mut h_1);
                    model.layers[1].forward_infer(&h_1, &mut s1, &mut sc1, &mut h_2);

                    let pred_v = MatrixView::new(&model.layers[1].w_pred, config.vocab_size, d);
                    axiom_core::matvec::matvec(&pred_v, &h_2, &mut logits);
                    let loss = axiom_core::softmax::cross_entropy_loss_and_grad(&logits, y_seq[t], &mut probs, &mut dummy_grad);
                    val_loss += loss;
                    val_tokens += 1;
                }
            }

            let mean_val_loss = val_loss / val_tokens as f32;
            let val_bpc = mean_val_loss / std::f32::consts::LN_2;
            final_l2_bpc = val_bpc;
            let elapsed = start_time_l2.elapsed().as_secs_f64();

            println!(
                "Stage 2 | Step {:>4} ({:>5.1}s) | Layer 2 Val Loss: {:.4} | Layer 2 Val BPC: {:.4} | LR: {:.2e}",
                step, elapsed, mean_val_loss, val_bpc, current_lr
            );
        }
    }

    println!("\n=======================================================");
    println!(">>> Progressive Freezing Diagnostic Results:");
    println!("    Layer 1 Final Val BPC (Frozen Baseline): {:.4} BPC", final_l1_bpc);
    println!("    Layer 2 Final Val BPC (Trained on H_1):   {:.4} BPC", final_l2_bpc);
    let delta = ((final_l2_bpc - final_l1_bpc) / final_l1_bpc) * 100.0;
    println!("    Delta (L2 vs L1):                        {:.2}%", delta);
    if final_l2_bpc < final_l1_bpc {
        println!("    Finding: Freezing restored monotonicity (L2 < L1), confirming drift caused inversion.");
        println!("             HOWEVER: Freezing strictly serializes training across layers, killing Claim C7.");
    } else {
        println!("    Finding: Even with frozen H_1, Layer 2 does NOT improve over Layer 1 (L2 >= L1).");
        println!("             Proves the greedy local loss actively discards cross-layer hierarchical information.");
    }
    println!("=======================================================\n");

    (final_l1_bpc, final_l2_bpc)
}
