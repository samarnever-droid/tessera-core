//! Interactive Text Generation & Prompt Completion CLI for TESSERA-Q.

use std::env;
use std::time::Instant;
use tessera_core::tessera_model::{TesseraConfig, TesseraModel};
use tessera_core::tessera_trainer::train_tessera;
use axiom_train::dataset::CharDataset;

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("The computer is an electronic device that");
    let train_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    println!("==========================================================================");
    println!("  TESSERA-Q: INTERACTIVE GENERATION & COMPLETION DEMO");
    println!("==========================================================================");

    let config = TesseraConfig::nano_default();
    let mut model = TesseraModel::new(256, 128, config, 42);

    let dataset_path = "data/enwik8";
    println!("Loading dataset ({}) ...", dataset_path);
    let raw_dataset = CharDataset::from_file(dataset_path).expect("Failed to load dataset");
    let (train_data, val_data) = raw_dataset.split(0.9);

    println!("Quick-training TESSERA-Q (0.73M params) for {} steps on Wikipedia text...", train_steps);
    let t0 = Instant::now();
    let history = train_tessera(
        &mut model,
        &train_data.data,
        &val_data.data,
        32,
        64,
        120,
        train_steps,
        3.0e-3,
        "Interactive Demo",
    );

    if let Some(&(step, train_loss, val_bpc, _)) = history.last() {
        println!("  Training Complete: Step {:3} | Train Loss: {:.4} | Val BPC: {:.4} | Time: {:.1}s", step, train_loss, val_bpc, t0.elapsed().as_secs_f32());
    }

    println!("\nPrompt: \"{}\"", prompt);
    println!("--------------------------------------------------------------------------");
    print!("{}", prompt);
    let generated = model.generate_text(prompt, 150, 0.7, 30, 42);
    // Print the continuation after the prompt
    if generated.len() > prompt.len() {
        println!("{}", &generated[prompt.len()..]);
    } else {
        println!("{}", generated);
    }
    println!("--------------------------------------------------------------------------\n");
}
