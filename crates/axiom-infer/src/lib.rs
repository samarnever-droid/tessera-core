//! `axiom-infer`: Autoregressive generation, circular buffer mixing, and sampling.

use axiom_core::softmax::softmax_temperature;
use axiom_model::layer::LayerScratch;
use axiom_model::model::AxiomSingleLayerModel;
use axiom_model::stacked_model::AxiomModel;
use axiom_model::LayerState;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Inference sampling configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub copy_threshold_tau: f32,
    pub copy_mix_alpha: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            copy_threshold_tau: 2.0,
            copy_mix_alpha: 0.3,
        }
    }
}

/// Sample a token from a probability distribution.
pub fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> usize {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Autoregressive text generation using single-layer AXIOM model.
pub fn generate_text(
    model: &AxiomSingleLayerModel,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
) -> String {
    let mut rng = StdRng::seed_from_u64(seed);
    let d = model.config.d_model;
    let v = model.config.vocab_size;

    let mut state = LayerState::new(&model.config);
    let mut scratch = LayerScratch::new(&model.config);
    let mut h_in = vec![0.0f32; d];
    let mut h_out = vec![0.0f32; d];
    let mut logits = vec![0.0f32; v];
    let mut probs = vec![0.0f32; v];

    let prompt_bytes = prompt.as_bytes();
    let mut generated_bytes = prompt_bytes.to_vec();

    // 1. Prime the model with prompt tokens
    for (pos, &b) in prompt_bytes.iter().enumerate() {
        let token = b as usize;
        model.forward_infer_step(
            token,
            pos,
            &mut state,
            &mut scratch,
            &mut h_in,
            &mut h_out,
            &mut logits,
        );
    }

    // 2. Autoregressively generate new tokens
    let mut current_pos = prompt_bytes.len();
    for _ in 0..max_new_tokens {
        softmax_temperature(&logits, temperature, &mut probs);
        let next_token = sample_from_probs(&probs, &mut rng);
        generated_bytes.push(next_token as u8);

        model.forward_infer_step(
            next_token,
            current_pos,
            &mut state,
            &mut scratch,
            &mut h_in,
            &mut h_out,
            &mut logits,
        );
        current_pos += 1;
    }

    String::from_utf8_lossy(&generated_bytes).to_string()
}

/// Autoregressive text generation using multi-layer stacked AXIOM model with O(1) memory.
pub fn generate_text_stacked(
    model: &AxiomModel,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
) -> String {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_layers = model.config.num_layers;
    let d = model.config.d_model;
    let v = model.config.vocab_size;

    let mut states: Vec<LayerState> = (0..num_layers).map(|_| LayerState::new(&model.config)).collect();
    let mut scratches: Vec<LayerScratch> = (0..num_layers).map(|_| LayerScratch::new(&model.config)).collect();
    let mut h_buffers: Vec<Vec<f32>> = (0..=num_layers).map(|_| vec![0.0f32; d]).collect();
    let mut logits = vec![0.0f32; v];
    let mut probs = vec![0.0f32; v];

    let prompt_bytes = prompt.as_bytes();
    let mut generated_bytes = prompt_bytes.to_vec();

    // 1. Prime the model with prompt tokens
    for (pos, &b) in prompt_bytes.iter().enumerate() {
        let token = b as usize;
        model.forward_infer_step(
            token,
            pos,
            &mut states,
            &mut scratches,
            &mut h_buffers,
            &mut logits,
        );
    }

    // 2. Autoregressively generate new tokens
    let mut current_pos = prompt_bytes.len();
    for _ in 0..max_new_tokens {
        softmax_temperature(&logits, temperature, &mut probs);
        let next_token = sample_from_probs(&probs, &mut rng);
        generated_bytes.push(next_token as u8);

        model.forward_infer_step(
            next_token,
            current_pos,
            &mut states,
            &mut scratches,
            &mut h_buffers,
            &mut logits,
        );
        current_pos += 1;
    }

    String::from_utf8_lossy(&generated_bytes).to_string()
}
