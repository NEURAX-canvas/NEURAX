//! Hyperparameter sweep over real models, one per family.
//!
//! Demonstrates `neurax_core::sweep::sweep_hyperparameters` — reusing the
//! existing analytical pipeline as a cheap black-box evaluator to search
//! batch_size × zero_stage × gpu_count × precision for the best feasible
//! (VRAM-fitting) point under three different objectives.

use neurax_core::sweep::{sweep_hyperparameters, SweepCandidates, SweepObjective};
use neurax_parser::parse_model_config;
use std::fs;

fn run_for(path: &str) {
    let json = fs::read_to_string(path).expect("fixture must exist");
    let config = parse_model_config(&json).expect("fixture must parse");
    let name = std::path::Path::new(path)
        .file_stem()
        .unwrap()
        .to_string_lossy();

    let candidates = SweepCandidates {
        batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
        zero_stages: vec![0, 1, 2, 3],
        gpu_counts: vec![config.hardware.gpus.first().map(|g| g.count).unwrap_or(1)],
        precisions: vec![config.training.precision.clone()],
    };

    println!("\n=== {name} ({} points to evaluate) ===", {
        candidates.batch_sizes.len() * candidates.zero_stages.len()
    });

    for objective in [
        SweepObjective::MaxThroughput,
        SweepObjective::MinCost,
        SweepObjective::MaxBatchSize,
    ] {
        let result =
            sweep_hyperparameters(&config, &candidates, objective).expect("sweep must not error");
        let feasible_count = result.points.iter().filter(|p| p.feasible).count();
        match &result.best {
            Some(p) => println!(
                "  {objective:?}: batch={:<4} zero={} -> {:.1} tok/s, ${:.2}, {:.2} GB/{} feasible/{total}",
                p.batch_size,
                p.zero_stage,
                p.throughput_tokens_per_s,
                p.training_cost_usd,
                p.peak_vram_gb,
                feasible_count,
                total = result.points.len(),
            ),
            None => println!("  {objective:?}: NO FEASIBLE POINT (model doesn't fit at any swept batch size)"),
        }
    }

    // Sanity check: VRAM should grow monotonically with batch_size at fixed
    // zero_stage — activation memory scales with batch, nothing in the model
    // should shrink it. A violation here would mean a real bug, not just an
    // uninteresting result.
    let result = sweep_hyperparameters(
        &config,
        &SweepCandidates {
            batch_sizes: candidates.batch_sizes.clone(),
            zero_stages: vec![0],
            gpu_counts: candidates.gpu_counts.clone(),
            precisions: candidates.precisions.clone(),
        },
        SweepObjective::MaxThroughput,
    )
    .unwrap();
    let mut by_batch = result.points.clone();
    by_batch.sort_by_key(|p| p.batch_size);
    for w in by_batch.windows(2) {
        assert!(
            w[1].peak_vram_gb >= w[0].peak_vram_gb - 1e-6,
            "{name}: VRAM should not shrink as batch grows (batch {} -> {}: {:.3} GB -> {:.3} GB)",
            w[0].batch_size,
            w[1].batch_size,
            w[0].peak_vram_gb,
            w[1].peak_vram_gb
        );
    }
    println!("  [ok] peak VRAM is monotonically non-decreasing in batch_size (zero_stage=0)");
}

fn main() {
    // One model per family that already ships under examples/models/.
    for path in [
        "examples/models/gpt3_175b.json",      // transformer
        "examples/models/mixtral_8x7b.json",   // moe
        "examples/models/resnet50.json",       // cnn
        "examples/models/mamba_2.8b.json",     // ssm
        "examples/models/sdxl_1.0.json",       // diffusion
        "examples/models/dcgan.json",          // gan
        "examples/models/awd_lstm_ptb.json",   // rnn
        "examples/models/gcn_ogbn_arxiv.json", // gnn
    ] {
        run_for(path);
    }
}
