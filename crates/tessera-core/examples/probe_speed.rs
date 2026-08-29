use std::time::Instant;
use tessera_core::tessera_model::{TesseraConfig, TesseraModel, TesseraModelGrads};
use axiom_train::dataset::CharDataset;

fn main() {
    let ds = CharDataset::from_file("data/tiny_shakespeare.txt").expect("load data");
    let (train, _val) = ds.split(0.9);

    let cfg = TesseraConfig {
        d_model: 672,
        d_ff: 2688,
        n_heads: 4,
        n_stages: 6,
        adapter_rank: 16,
        use_mrm_v2: true,
        k_fine_slots: 128,
        k_coarse_slots: 16,
        use_meridian: false,
    };
    let seq_len = 128usize;
    let mut model = TesseraModel::new(256, seq_len, cfg, 42);
    let (total, _active, _dram, _param_bytes, _mem_footprint) = model.parameter_metrics();
    println!("Total params: {:.3} M", total as f32 / 1e6);

    let mut grads = TesseraModelGrads::new(model.vocab_size, model.d_model, model.max_seq_len, &model.stages);

    let n_probe = 8usize;
    let start = Instant::now();
    for i in 0..n_probe {
        let s = (i * 97) % (train.data.len() - seq_len - 1);
        let seq = &train.data[s..s + seq_len + 1];
        let x: Vec<usize> = seq[..seq_len].iter().map(|&b| b as usize).collect();
        let y: Vec<usize> = seq[1..seq_len + 1].iter().map(|&b| b as usize).collect();
        grads.zero();
        let loss = model.forward_backward_sequence(&x, &y, &mut grads);
        println!("step {} loss {:.4}", i, loss / seq_len as f32);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let toks = n_probe * seq_len;
    println!("Elapsed: {:.2}s for {} tokens => {:.1} tok/s (single-threaded fwd+bwd)", elapsed, toks, toks as f64 / elapsed);
}
