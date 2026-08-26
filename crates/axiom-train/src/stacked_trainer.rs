//! Decoupled parallel multi-layer batch trainer for stacked AXIOM models.

use crate::dataset::CharDataset;
use crate::optimizer::{AdamWConfig, ParameterState};
use axiom_model::layer::{LayerScratch, SequenceCache};
use axiom_model::stacked_model::{AxiomModel, StackedModelGrads};
use axiom_model::LayerState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

/// Stacked AdamW Optimizer tracking moments for all L layers and embeddings.
#[derive(Debug, Clone)]
pub struct StackedAdamW {
    pub config: AdamWConfig,
    pub step: usize,
    pub state_embed: ParameterState,
    pub state_pos_embed: ParameterState,
    pub state_ws: Vec<ParameterState>,
    pub state_w_gate: Vec<ParameterState>,
    pub state_experts_up: Vec<Vec<ParameterState>>,
    pub state_experts_down: Vec<Vec<ParameterState>>,
    pub state_w_pred: Vec<ParameterState>,
    pub state_w_decode: Vec<ParameterState>,
}

impl StackedAdamW {
    pub fn new(config: AdamWConfig, model: &AxiomModel) -> Self {
        let num_layers = model.config.num_layers;
        let mut state_ws = Vec::with_capacity(num_layers);
        let mut state_w_gate = Vec::with_capacity(num_layers);
        let mut state_experts_up = Vec::with_capacity(num_layers);
        let mut state_experts_down = Vec::with_capacity(num_layers);
        let mut state_w_pred = Vec::with_capacity(num_layers);
        let mut state_w_decode = Vec::with_capacity(num_layers);

        for layer in &model.layers {
            state_ws.push(ParameterState::new(layer.w_s.len()));
            state_w_gate.push(ParameterState::new(layer.w_gate.len()));

            let mut exp_up = Vec::with_capacity(model.config.num_experts);
            let mut exp_down = Vec::with_capacity(model.config.num_experts);
            for exp in &layer.experts {
                exp_up.push(ParameterState::new(exp.w_up.len()));
                exp_down.push(ParameterState::new(exp.w_down.len()));
            }
            state_experts_up.push(exp_up);
            state_experts_down.push(exp_down);

            state_w_pred.push(ParameterState::new(layer.w_pred.len()));
            state_w_decode.push(ParameterState::new(layer.w_decode.len()));
        }

        Self {
            config,
            step: 0,
            state_embed: ParameterState::new(model.embeddings.len()),
            state_pos_embed: ParameterState::new(model.pos_embeddings.len()),
            state_ws,
            state_w_gate,
            state_experts_up,
            state_experts_down,
            state_w_pred,
            state_w_decode,
        }
    }

    pub fn compute_grad_norm(&self, grads: &StackedModelGrads) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &grads.grad_embeddings {
            sum_sq += g * g;
        }
        for &g in &grads.grad_pos_embeddings {
            sum_sq += g * g;
        }
        for lg in &grads.layer_grads {
            for &g in &lg.grad_w_s {
                sum_sq += g * g;
            }
            for &g in &lg.grad_w_gate {
                sum_sq += g * g;
            }
            for eg in &lg.expert_grads {
                for &g in &eg.grad_w_up {
                    sum_sq += g * g;
                }
                for &g in &eg.grad_w_down {
                    sum_sq += g * g;
                }
            }
            for &g in &lg.grad_w_pred {
                sum_sq += g * g;
            }
            for &g in &lg.grad_w_decode {
                sum_sq += g * g;
            }
        }
        sum_sq.sqrt()
    }

    pub fn step(&mut self, model: &mut AxiomModel, grads: &mut StackedModelGrads, current_lr: f32) {
        self.step += 1;
        let t = self.step as f32;

        let grad_norm = self.compute_grad_norm(grads);
        let clip_scale = if grad_norm > self.config.max_grad_norm && grad_norm > 1e-8 {
            self.config.max_grad_norm / grad_norm
        } else {
            1.0f32
        };

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        let bc1 = 1.0f32 - beta1.powf(t);
        let bc2 = 1.0f32 - beta2.powf(t);
        let inv_bc1 = 1.0f32 / bc1;
        let inv_bc2 = 1.0f32 / bc2;

        let update_p = |p: &mut [f32], g: &[f32], state: &mut ParameterState| {
            for ((param, &grad), (m, v)) in p.iter_mut().zip(g.iter()).zip(state.m.iter_mut().zip(state.v.iter_mut())) {
                let scaled_grad = grad * clip_scale;
                *m = beta1 * *m + (1.0 - beta1) * scaled_grad;
                *v = beta2 * *v + (1.0 - beta2) * scaled_grad * scaled_grad;

                let m_hat = *m * inv_bc1;
                let v_hat = *v * inv_bc2;

                let step_val = m_hat / (v_hat.sqrt() + eps) + wd * *param;
                *param -= current_lr * step_val;
            }
        };

        update_p(&mut model.embeddings, &grads.grad_embeddings, &mut self.state_embed);
        update_p(&mut model.pos_embeddings, &grads.grad_pos_embeddings, &mut self.state_pos_embed);

        for l in 0..model.config.num_layers {
            let layer = &mut model.layers[l];
            let lg = &grads.layer_grads[l];

            update_p(&mut layer.w_s, &lg.grad_w_s, &mut self.state_ws[l]);
            update_p(&mut layer.w_gate, &lg.grad_w_gate, &mut self.state_w_gate[l]);

            for (i, exp) in layer.experts.iter_mut().enumerate() {
                update_p(&mut exp.w_up, &lg.expert_grads[i].grad_w_up, &mut self.state_experts_up[l][i]);
                update_p(&mut exp.w_down, &lg.expert_grads[i].grad_w_down, &mut self.state_experts_down[l][i]);
            }

            update_p(&mut layer.w_pred, &lg.grad_w_pred, &mut self.state_w_pred[l]);
            update_p(&mut layer.w_decode, &lg.grad_w_decode, &mut self.state_w_decode[l]);
        }
    }
}

/// Evaluate per-layer validation BPC across all L stacked layers.
pub fn evaluate_stacked_bpc(
    model: &AxiomModel,
    val_dataset: &CharDataset,
    num_batches: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(999);
    let num_layers = model.config.num_layers;
    let d = model.config.d_model;
    let mut total_losses = vec![0.0f32; num_layers];
    let mut total_tokens = 0usize;

    let mut states: Vec<LayerState> = (0..num_layers)
        .map(|_| LayerState::new(&model.config))
        .collect();
    let mut scratches: Vec<LayerScratch> = (0..num_layers)
        .map(|_| LayerScratch::new(&model.config))
        .collect();
    let mut caches: Vec<SequenceCache> = (0..num_layers)
        .map(|_| SequenceCache::new(&model.config, seq_len))
        .collect();
    let mut layer_h_seqs: Vec<Vec<f32>> = (0..=num_layers)
        .map(|_| vec![0.0f32; seq_len * d])
        .collect();

    for _ in 0..num_batches {
        let (inputs, targets) = val_dataset.sample_batch(1, seq_len, &mut rng);
        let x_seq = &inputs[0];
        let y_seq = &targets[0];

        for s in &mut states {
            s.reset();
        }

        let losses = model.forward_sequence_stacked(
            x_seq,
            y_seq,
            &mut states,
            &mut scratches,
            &mut caches,
            &mut layer_h_seqs,
        );

        for l in 0..num_layers {
            total_losses[l] += losses[l];
        }
        total_tokens += seq_len;
    }

    total_losses
        .into_iter()
        .map(|loss| (loss / total_tokens as f32) / std::f32::consts::LN_2)
        .collect()
}

/// Train stacked multi-layer AXIOM model on CPU with decoupled parallel layer backprop.
pub fn train_stacked_model(
    model: &mut AxiomModel,
    train_dataset: &CharDataset,
    val_dataset: &CharDataset,
    config: &crate::trainer::TrainerConfig,
) -> Vec<(usize, Vec<f32>, f64)> {
    let mut optimizer = StackedAdamW::new(
        AdamWConfig {
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
    let mut master_rng = StdRng::seed_from_u64(42);
    let mut history = Vec::new();
    let num_layers = model.config.num_layers;

    println!(
        "Starting AXIOM Stacked Training (L={}) | Batch: {}, Seq: {}, Max Time: {}s",
        num_layers, config.batch_size, config.seq_len, config.max_time_secs
    );

    for step in 1..=config.max_steps {
        let elapsed_sec = start_time.elapsed().as_secs_f64();
        if elapsed_sec >= config.max_time_secs as f64 {
            println!("\nReached max time limit of {}s at step {}", config.max_time_secs, step);
            break;
        }

        // 1. Sample batch
        let (batch_x, batch_y) = train_dataset.sample_batch(config.batch_size, config.seq_len, &mut master_rng);

        // 2. Parallel processing over batch sequences
        let model_ref = &*model;
        let loss_weights = config.loss_weights.clone();
        let seq_len = config.seq_len;
        let d = model_ref.config.d_model;

        let thread_results: Vec<(Vec<f32>, StackedModelGrads)> = batch_x
            .into_par_iter()
            .zip(batch_y.into_par_iter())
            .map(|(x_seq, y_seq)| {
                let mut states: Vec<LayerState> = (0..num_layers)
                    .map(|_| LayerState::new(&model_ref.config))
                    .collect();
                let mut scratches: Vec<LayerScratch> = (0..num_layers)
                    .map(|_| LayerScratch::new(&model_ref.config))
                    .collect();
                let mut caches: Vec<SequenceCache> = (0..num_layers)
                    .map(|_| SequenceCache::new(&model_ref.config, seq_len))
                    .collect();
                let mut layer_h_seqs: Vec<Vec<f32>> = (0..=num_layers)
                    .map(|_| vec![0.0f32; seq_len * d])
                    .collect();
                let mut grads = StackedModelGrads::new(&model_ref.config, model_ref.max_seq_len);

                // Sequential forward pass across layers (L stages)
                let losses = model_ref.forward_sequence_stacked(
                    &x_seq,
                    &y_seq,
                    &mut states,
                    &mut scratches,
                    &mut caches,
                    &mut layer_h_seqs,
                );

                // Decoupled parallel backward pass across all L layers
                model_ref.backward_decoupled_parallel(
                    &x_seq,
                    &caches,
                    &mut scratches,
                    loss_weights.lambda_pred,
                    loss_weights.lambda_recon,
                    loss_weights.lambda_residual,
                    &mut grads,
                );

                (losses, grads)
            })
            .collect();

        // 3. Aggregate gradients and losses
        let mut total_grads = StackedModelGrads::new(&model.config, model.max_seq_len);
        let scale = 1.0f32 / (config.batch_size * config.seq_len) as f32;

        for (_losses, g) in thread_results {
            total_grads.add(&g);
        }

        // Scale gradients
        axiom_core::tensor::vec_scale(&mut total_grads.grad_embeddings, scale);
        axiom_core::tensor::vec_scale(&mut total_grads.grad_pos_embeddings, scale);
        for lg in &mut total_grads.layer_grads {
            axiom_core::tensor::vec_scale(&mut lg.grad_w_s, scale);
            axiom_core::tensor::vec_scale(&mut lg.grad_w_gate, scale);
            for eg in &mut lg.expert_grads {
                axiom_core::tensor::vec_scale(&mut eg.grad_w_up, scale);
                axiom_core::tensor::vec_scale(&mut eg.grad_w_down, scale);
            }
            axiom_core::tensor::vec_scale(&mut lg.grad_w_pred, scale);
            axiom_core::tensor::vec_scale(&mut lg.grad_w_decode, scale);
        }

        // 4. Optimizer update
        let current_lr = if step < config.warmup_steps {
            let alpha = step as f32 / config.warmup_steps.max(1) as f32;
            config.min_lr + alpha * (config.base_lr - config.min_lr)
        } else {
            let prog = (step - config.warmup_steps) as f32 / (config.max_steps - config.warmup_steps).max(1) as f32;
            config.min_lr + 0.5 * (1.0 + (PI * prog.min(1.0)).cos()) * (config.base_lr - config.min_lr)
        };

        optimizer.step(model, &mut total_grads, current_lr);

        // 5. Periodic evaluation
        if step % config.eval_interval == 0 || step == 1 {
            let bpcs = evaluate_stacked_bpc(model, val_dataset, 10, config.seq_len);
            let elapsed = start_time.elapsed().as_secs_f64();
            let tok_s = (step * config.batch_size * config.seq_len) as f64 / elapsed;

            let bpc_str: Vec<String> = bpcs.iter().enumerate().map(|(l, &b)| format!("L{}:{:.3}", l + 1, b)).collect();
            println!(
                "Step {:>4} ({:>5.1}s) | Val BPC: [{}] | LR: {:.2e} | Tok/s: {:.0}",
                step, elapsed, bpc_str.join(" "), current_lr, tok_s
            );

            history.push((step, bpcs, elapsed));
        }
    }

    history
}
