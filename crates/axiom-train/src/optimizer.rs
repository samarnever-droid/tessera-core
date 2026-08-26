//! Pure Rust AdamW optimizer with gradient norm clipping and weight decay.

use axiom_model::model::{AxiomSingleLayerModel, ModelGrads};

/// AdamW Optimizer Configuration.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

/// AdamW state tracker for parameter buffers.
#[derive(Debug, Clone)]
pub struct ParameterState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParameterState {
    pub fn new(len: usize) -> Self {
        Self {
            m: vec![0.0f32; len],
            v: vec![0.0f32; len],
        }
    }
}

/// AdamW optimizer holding momentum and second moment states for an AxiomSingleLayerModel.
#[derive(Debug, Clone)]
pub struct AdamW {
    pub config: AdamWConfig,
    pub step: usize,
    pub state_embed: ParameterState,
    pub state_pos_embed: ParameterState,
    pub state_ws: ParameterState,
    pub state_w_gate: ParameterState,
    pub state_experts_up: Vec<ParameterState>,
    pub state_experts_down: Vec<ParameterState>,
    pub state_w_pred: ParameterState,
    pub state_w_decode: ParameterState,
}

impl AdamW {
    pub fn new(config: AdamWConfig, model: &AxiomSingleLayerModel) -> Self {
        let mut state_experts_up = Vec::with_capacity(model.config.num_experts);
        let mut state_experts_down = Vec::with_capacity(model.config.num_experts);
        for exp in &model.layer.experts {
            state_experts_up.push(ParameterState::new(exp.w_up.len()));
            state_experts_down.push(ParameterState::new(exp.w_down.len()));
        }

        Self {
            config,
            step: 0,
            state_embed: ParameterState::new(model.embeddings.len()),
            state_pos_embed: ParameterState::new(model.pos_embeddings.len()),
            state_ws: ParameterState::new(model.layer.w_s.len()),
            state_w_gate: ParameterState::new(model.layer.w_gate.len()),
            state_experts_up,
            state_experts_down,
            state_w_pred: ParameterState::new(model.layer.w_pred.len()),
            state_w_decode: ParameterState::new(model.layer.w_decode.len()),
        }
    }

    /// Compute total gradient L2 norm across all model parameters.
    pub fn compute_grad_norm(&self, grads: &ModelGrads) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &grads.grad_embeddings {
            sum_sq += g * g;
        }
        for &g in &grads.grad_pos_embeddings {
            sum_sq += g * g;
        }
        for &g in &grads.layer_grads.grad_w_s {
            sum_sq += g * g;
        }
        for &g in &grads.layer_grads.grad_w_gate {
            sum_sq += g * g;
        }
        for eg in &grads.layer_grads.expert_grads {
            for &g in &eg.grad_w_up {
                sum_sq += g * g;
            }
            for &g in &eg.grad_w_down {
                sum_sq += g * g;
            }
        }
        for &g in &grads.layer_grads.grad_w_pred {
            sum_sq += g * g;
        }
        for &g in &grads.layer_grads.grad_w_decode {
            sum_sq += g * g;
        }
        sum_sq.sqrt()
    }

    /// Step optimizer: update parameters using accumulated gradients.
    pub fn step(&mut self, model: &mut AxiomSingleLayerModel, grads: &mut ModelGrads, current_lr: f32) {
        self.step += 1;
        let t = self.step as f32;

        // Gradient clipping
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

        let update_param = |p: &mut [f32], g: &[f32], state: &mut ParameterState| {
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

        // 1. Update token and positional embeddings
        update_param(&mut model.embeddings, &grads.grad_embeddings, &mut self.state_embed);
        update_param(&mut model.pos_embeddings, &grads.grad_pos_embeddings, &mut self.state_pos_embed);

        // 2. Update W_s
        update_param(&mut model.layer.w_s, &grads.layer_grads.grad_w_s, &mut self.state_ws);

        // 3. Update W_gate
        update_param(&mut model.layer.w_gate, &grads.layer_grads.grad_w_gate, &mut self.state_w_gate);

        // 4. Update Experts
        for (i, expert) in model.layer.experts.iter_mut().enumerate() {
            update_param(&mut expert.w_up, &grads.layer_grads.expert_grads[i].grad_w_up, &mut self.state_experts_up[i]);
            update_param(&mut expert.w_down, &grads.layer_grads.expert_grads[i].grad_w_down, &mut self.state_experts_down[i]);
        }

        // 5. Update W_pred
        update_param(&mut model.layer.w_pred, &grads.layer_grads.grad_w_pred, &mut self.state_w_pred);

        // 6. Update W_decode
        update_param(&mut model.layer.w_decode, &grads.layer_grads.grad_w_decode, &mut self.state_w_decode);
    }
}
