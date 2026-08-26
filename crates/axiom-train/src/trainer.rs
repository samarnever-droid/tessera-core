//! Multi-threaded CPU batch trainer for single-layer AXIOM model with Truncated BPTT.

use crate::dataset::CharDataset;
use crate::optimizer::AdamW;
use crate::LocalLossConfig;
use axiom_model::layer::{LayerScratch, SequenceCache};
use axiom_model::model::{AxiomSingleLayerModel, ModelGrads};
use axiom_model::LayerState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

/// Training Hyperparameters for Single-Layer Model.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_steps: usize,
    pub max_time_secs: u64,
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub eval_interval: usize,
    pub loss_weights: LocalLossConfig,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            seq_len: 64,
            max_steps: 5000,
            max_time_secs: 300, // 5 minute hard budget per Claim C1
            base_lr: 5e-3,
            min_lr: 1e-4,
            warmup_steps: 100,
            eval_interval: 50,
            loss_weights: LocalLossConfig::default(),
        }
    }
}

/// Compute cosine decayed learning rate with linear warmup.
pub fn get_lr(step: usize, config: &TrainerConfig) -> f32 {
    if step < config.warmup_steps {
        let alpha = (step as f32) / (config.warmup_steps.max(1) as f32);
        config.min_lr + alpha * (config.base_lr - config.min_lr)
    } else {
        let progress = ((step - config.warmup_steps) as f32)
            / ((config.max_steps - config.warmup_steps).max(1) as f32);
        let progress = progress.min(1.0);
        let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
        config.min_lr + cosine_decay * (config.base_lr - config.min_lr)
    }
}

/// Evaluate model validation Cross-Entropy loss and Bits Per Character (BPC).
pub fn evaluate_bpc(
    model: &AxiomSingleLayerModel,
    val_dataset: &CharDataset,
    num_batches: usize,
    seq_len: usize,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(999);
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;

    let d = model.config.d_model;
    let mut state = LayerState::new(&model.config);
    let mut scratch = LayerScratch::new(&model.config);
    let mut h_in = vec![0.0f32; d];
    let mut h_out = vec![0.0f32; d];

    for _ in 0..num_batches {
        let (inputs, targets) = val_dataset.sample_batch(1, seq_len, &mut rng);
        let x_seq = &inputs[0];
        let y_seq = &targets[0];

        state.reset();
        for t in 0..seq_len {
            let (loss_pred, _) = model.forward_train_step(
                x_seq[t],
                t,
                y_seq[t],
                &mut state,
                &mut scratch,
                &mut h_in,
                &mut h_out,
            );
            total_loss += loss_pred;
            total_tokens += 1;
        }
    }

    let mean_loss = if total_tokens > 0 {
        total_loss / total_tokens as f32
    } else {
        0.0f32
    };
    let bpc = mean_loss / std::f32::consts::LN_2;
    (mean_loss, bpc)
}

/// Train a single-layer AXIOM model on CPU with within-layer BPTT.
pub fn train_single_layer_bptt(
    model: &mut AxiomSingleLayerModel,
    train_dataset: &CharDataset,
    val_dataset: &CharDataset,
    config: &TrainerConfig,
) -> Vec<(usize, f32, f32, f64)> {
    let mut optimizer = AdamW::new(
        crate::optimizer::AdamWConfig {
            lr: config.base_lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        },
        model,
    );

    let start_time = Instant::now();
    let mut history = Vec::new();
    let mut master_rng = StdRng::seed_from_u64(42);

    println!(
        "Starting AXIOM BPTT Single-Layer Training | Batch Size: {}, Seq Len: {}, Max Time: {}s",
        config.batch_size, config.seq_len, config.max_time_secs
    );

    for step in 1..=config.max_steps {
        let elapsed_sec = start_time.elapsed().as_secs_f64();
        if elapsed_sec >= config.max_time_secs as f64 {
            println!("\nReached max time limit of {}s at step {}", config.max_time_secs, step);
            break;
        }

        // 1. Sample batch
        let (batch_x, batch_y) = train_dataset.sample_batch(config.batch_size, config.seq_len, &mut master_rng);

        // 2. Parallel sequence forward + backward (BPTT) across threads
        let model_ref = &*model;
        let loss_weights = config.loss_weights.clone();
        let seq_len = config.seq_len;

        let thread_results: Vec<(f32, f32, ModelGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut state = LayerState::new(&model_ref.config);
                let mut scratch = LayerScratch::new(&model_ref.config);
                let mut cache = SequenceCache::new(&model_ref.config, seq_len);
                let mut grads = ModelGrads::new(&model_ref.config, model_ref.max_seq_len);

                // Forward pass across sequence caching activations
                let (lp, lr) = model_ref.forward_sequence_cache(
                    &x_seq,
                    &y_seq,
                    &mut state,
                    &mut scratch,
                    &mut cache,
                );

                // Truncated BPTT Backward pass across sequence
                model_ref.backward_sequence_bptt(
                    &x_seq,
                    &cache,
                    loss_weights.lambda_pred,
                    loss_weights.lambda_recon,
                    loss_weights.lambda_residual,
                    &mut scratch,
                    &mut grads,
                );

                (lp, lr, grads)
            })
            .collect();

        // 3. Aggregate gradients and losses
        let mut total_grads = ModelGrads::new(&model.config, model.max_seq_len);
        let mut batch_loss_pred = 0.0f32;
        let mut _batch_loss_recon = 0.0f32;
        let scale = 1.0f32 / (config.batch_size * config.seq_len) as f32;

        for (lp, lr, g) in thread_results {
            batch_loss_pred += lp;
            _batch_loss_recon += lr;
            total_grads.add(&g);
        }

        // Scale gradients by 1 / (batch_size * seq_len)
        axiom_core::tensor::vec_scale(&mut total_grads.grad_embeddings, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_pos_embeddings, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.layer_grads.grad_w_s, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.layer_grads.grad_w_gate, scale);
        for eg in &mut total_grads.layer_grads.expert_grads {
            axiom_core::tensor::vec_scale(&mut eg.grad_w_up, scale);
            axiom_core::tensor::vec_scale(&mut eg.grad_w_down, scale);
        }
        axiom_core::tensor::vec_scale(&mut total_grads.layer_grads.grad_w_pred, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.layer_grads.grad_w_decode, scale);

        // 4. Optimizer update
        let current_lr = get_lr(step, config);
        optimizer.step(model, &mut total_grads, current_lr);

        // 5. Periodic evaluation and logging
        if step % config.eval_interval == 0 || step == 1 {
            let mean_train_loss = (batch_loss_pred * scale) * config.seq_len as f32;
            let (val_loss, val_bpc) = evaluate_bpc(model, val_dataset, 10, config.seq_len);
            let elapsed = start_time.elapsed().as_secs_f64();
            let tokens_per_sec = ((step * config.batch_size * config.seq_len) as f64) / elapsed;

            println!(
                "Step {:>4} ({:>5.1}s) | Train Loss: {:.4} | Val Loss: {:.4} | Val BPC: {:.4} | LR: {:.2e} | Tok/s: {:.0}",
                step, elapsed, mean_train_loss, val_loss, val_bpc, current_lr, tokens_per_sec
            );

            history.push((step, val_loss, val_bpc, elapsed));
        }
    }

    history
}
