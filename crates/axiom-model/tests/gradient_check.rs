use axiom_model::layer::LayerScratch;
use axiom_model::model::{AxiomSingleLayerModel, ModelGrads};
use axiom_model::{AxiomConfig, LayerState};

const EPS: f32 = 1e-3;
const TOL: f32 = 5e-2; // Tolerance for finite differences

#[test]
fn test_single_layer_gradient_check() {
    let config = AxiomConfig {
        vocab_size: 16,
        d_model: 32,
        num_layers: 1,
        num_experts: 4,
        active_experts: 2,
        buffer_capacity: 64,
        d_ffn: 64,
        hebbian_decay: 0.999,
        hebbian_lr: 1e-4,
    };

    let mut model = AxiomSingleLayerModel::new(config.clone(), 64, 123);
    let mut state = LayerState::new(&config);
    let mut scratch = LayerScratch::new(&config);
    let mut grads = ModelGrads::new(&config, 64);

    let token_x = 3;
    let pos = 0;
    let target_y = 7;
    let mut h_in = vec![0.0f32; config.d_model];
    let mut h_out = vec![0.0f32; config.d_model];
    let mut grad_h_in = vec![0.0f32; config.d_model];

    let state_prev = state.recurrent_state.clone();

    // 1. Analytical Forward & Backward
    let (_loss_pred, _loss_recon) = model.forward_train_step(
        token_x,
        pos,
        target_y,
        &mut state,
        &mut scratch,
        &mut h_in,
        &mut h_out,
    );
    let state_curr = state.recurrent_state.clone();

    let lambda_pred = 1.0f32;
    let lambda_recon = 0.5f32;
    let lambda_res = 0.01f32;

    model.backward_train_step(
        token_x,
        pos,
        &h_in,
        &state_prev,
        &state_curr,
        &h_out,
        lambda_pred,
        lambda_recon,
        lambda_res,
        &mut scratch,
        &mut grads,
        &mut grad_h_in,
    );

    // 2. Numerical Gradient Check for W_pred
    let eval_loss = |m: &AxiomSingleLayerModel| -> f32 {
        let mut st = LayerState::new(&config);
        let mut sc = LayerScratch::new(&config);
        let mut hi = vec![0.0f32; config.d_model];
        let mut ho = vec![0.0f32; config.d_model];
        let (lp, lr) = m.forward_train_step(token_x, pos, target_y, &mut st, &mut sc, &mut hi, &mut ho);
        let d = config.d_model as f32;
        let mut res = 0.0f32;
        for (&a, &b) in ho.iter().zip(hi.iter()) {
            res += (a - b) * (a - b);
        }
        res /= d;
        lambda_pred * lp + lambda_recon * lr + lambda_res * res
    };

    // Check sample indices of W_pred
    for idx in [0, 5, 12, 50, 100, 200].iter().cloned() {
        if idx < model.layer.w_pred.len() {
            let orig = model.layer.w_pred[idx];
            model.layer.w_pred[idx] = orig + EPS;
            let l_plus = eval_loss(&model);
            model.layer.w_pred[idx] = orig - EPS;
            let l_minus = eval_loss(&model);
            model.layer.w_pred[idx] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * EPS);
            let ana_grad = grads.layer_grads.grad_w_pred[idx];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < TOL || diff / (num_grad.abs() + ana_grad.abs() + 1e-5) < TOL,
                "W_pred grad mismatch at idx {}: num={}, ana={}, diff={}",
                idx, num_grad, ana_grad, diff
            );
        }
    }

    // Check sample indices of W_decode
    for idx in [0, 10, 50, 100].iter().cloned() {
        if idx < model.layer.w_decode.len() {
            let orig = model.layer.w_decode[idx];
            model.layer.w_decode[idx] = orig + EPS;
            let l_plus = eval_loss(&model);
            model.layer.w_decode[idx] = orig - EPS;
            let l_minus = eval_loss(&model);
            model.layer.w_decode[idx] = orig;

            let num_grad = (l_plus - l_minus) / (2.0 * EPS);
            let ana_grad = grads.layer_grads.grad_w_decode[idx];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < TOL || diff / (num_grad.abs() + ana_grad.abs() + 1e-5) < TOL,
                "W_decode grad mismatch at idx {}: num={}, ana={}, diff={}",
                idx, num_grad, ana_grad, diff
            );
        }
    }
}
