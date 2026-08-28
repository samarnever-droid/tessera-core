//! Standalone Production Inference CLI for TESSERA-Q in Native Rust.
//! Loads exported binary checkpoints from Python / Triton / Kaggle and executes AVX-accelerated text generation.

use std::env;
use std::time::Instant;
use tessera_core::tessera_model::TesseraModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("==========================================================================");
        println!("  TESSERA-Q: NATIVE RUST HIGH-PERFORMANCE INFERENCE ENGINE");
        println!("==========================================================================");
        println!("Usage: tessera-infer <path_to_checkpoint.bin> [prompt] [max_tokens] [temp]");
        println!("Example: tessera-infer tessera_qwen_1.5b.bin \"What is a computer?\" 100 0.7\n");
        return;
    }

    let checkpoint_path = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("What is a computer and how does its CPU process data?");
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(120);
    let temperature: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.7);

    println!("==========================================================================");
    println!("  TESSERA-Q NATIVE RUST ENGINE");
    println!("==========================================================================");
    println!("Loading checkpoint: {}", checkpoint_path);

    let t0 = Instant::now();
    let mut model = match TesseraModel::load_binary(checkpoint_path) {
        Ok(m) => {
            println!("✓ Checkpoint loaded successfully in {:.2}s!", t0.elapsed().as_secs_f32());
            println!("  Vocab Size: {} | Dim: {} | FFN: {} | Stages: {}", m.vocab_size, m.d_model, m.config.d_ff, m.stages.len());
            m
        }
        Err(e) => {
            eprintln!("Error loading binary checkpoint: {}", e);
            return;
        }
    };

    println!("\n💬 Prompt: \"{}\"", prompt);
    println!("--------------------------------------------------------------------------");
    print!("{}", prompt);

    let t_gen = Instant::now();
    let generated = model.generate_text(prompt, max_tokens, temperature, 20, 42);

    if generated.len() > prompt.len() {
        println!("{}", &generated[prompt.len()..]);
    } else {
        println!("{}", generated);
    }

    let gen_time = t_gen.elapsed().as_secs_f32();
    let tok_per_sec = max_tokens as f32 / gen_time.max(1e-4);
    println!("--------------------------------------------------------------------------");
    println!("⚡ Generation Time: {:.2}s ({:.1} tok/s on CPU with AVX2!)", gen_time, tok_per_sec);
    println!("==========================================================================\n");
}
