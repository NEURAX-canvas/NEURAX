//! Demonstrates the memoization claim, corrected to the boundary verified in
//! `src/lib.rs`'s module doc: only Architecture and Graph are genuinely
//! invariant across batch_size/zero_stage/precision/gpu_count.
//! Tensor/Operator/Compute depend on batch_size (via shape inference) and
//! are recomputed per `View` — a smaller, but real and honestly measured,
//! win over `neurax_core::sweep::sweep_hyperparameters`, which reruns every
//! one of the 9 phases per candidate.
//!
//! This does not claim a dramatic wall-clock win for these small reference
//! models — a full analysis is already sub-millisecond. What's measured
//! here is real: how many times Architecture/Graph actually run.

use neurax_core::sweep::{sweep_hyperparameters, SweepCandidates, SweepObjective};
use neurax_ir_poc::{StructuralFacts, View};
use std::time::Instant;

fn main() {
    let json = std::fs::read_to_string("examples/models/gpt3_175b.json")
        .expect("gpt3_175b.json must exist");
    let config = neurax_parser::parse_model_config(&json).expect("must parse");

    let candidates = SweepCandidates {
        batch_sizes: vec![1, 2, 4, 8, 16, 32],
        zero_stages: vec![0, 1, 2, 3],
        gpu_counts: vec![config.hardware.gpus.first().map(|g| g.count).unwrap_or(1)],
        precisions: vec![config.training.precision.clone()],
    };
    let grid_size =
        candidates.batch_sizes.len() * candidates.zero_stages.len() * candidates.gpu_counts.len();
    println!("Sweeping {grid_size} configurations for gpt3_175b.json\n");

    // ── Current: sweep_hyperparameters reruns all 9 phases per point ──
    let t0 = Instant::now();
    let old_result = sweep_hyperparameters(&config, &candidates, SweepObjective::MaxThroughput)
        .expect("sweep must not error");
    let old_elapsed = t0.elapsed();
    println!(
        "Current sweep_hyperparameters: {} points, {:?} total ({:?}/point average)",
        old_result.points.len(),
        old_elapsed,
        old_elapsed / old_result.points.len() as u32
    );

    // ── This PoC: Architecture+Graph computed once, Tensor/Operator/Compute
    // recomputed per View (they depend on batch_size), Memory/Parallelism/
    // Hardware/Cost per View as before.
    let t0 = Instant::now();
    let facts = StructuralFacts::compute(&config).expect("structural facts");
    let structural_elapsed = t0.elapsed();
    println!("\nStructuralFacts::compute() (Architecture+Graph only) called ONCE: {structural_elapsed:?}");

    let t0 = Instant::now();
    let mut new_points = Vec::new();
    for &batch_size in &candidates.batch_sizes {
        for &zero_stage in &candidates.zero_stages {
            let mut c = config.clone();
            c.training.batch_size = batch_size;
            c.training.zero_stage = zero_stage;
            let view = View::new(&facts, c);
            let vram = view.peak_vram_bytes().expect("vram");
            let comm = view.communication_overhead().expect("comm");
            new_points.push((batch_size, zero_stage, vram, comm));
        }
    }
    let views_elapsed = t0.elapsed();
    println!(
        "{} Views built + queried (Tensor/Operator/Compute/Memory/Parallelism run per view): {:?} total ({:?}/point average)",
        new_points.len(),
        views_elapsed,
        views_elapsed / new_points.len() as u32
    );
    println!(
        "\nTotal (Architecture+Graph once + {} views): {:?}  vs.  current: {:?}",
        new_points.len(),
        structural_elapsed + views_elapsed,
        old_elapsed
    );

    // ── Sanity: VRAM must vary with batch_size (it does now — this exact
    // assertion is what caught the first version's bug, sharing Tensor
    // across batch sizes and silently freezing this value).
    let vram_batch1 = new_points
        .iter()
        .find(|(b, z, _, _)| *b == 1 && *z == 0)
        .unwrap()
        .2;
    let vram_batch32 = new_points
        .iter()
        .find(|(b, z, _, _)| *b == 32 && *z == 0)
        .unwrap()
        .2;
    assert!(
        vram_batch32 > vram_batch1,
        "VRAM must grow with batch_size — got batch=1: {vram_batch1}, batch=32: {vram_batch32}"
    );
    println!(
        "\n[ok] VRAM correctly varies with batch_size (batch=1: {:.3} GB, batch=32: {:.3} GB)",
        vram_batch1 as f64 / 1e9,
        vram_batch32 as f64 / 1e9
    );

    // ── Sanity: communication_overhead must vary with zero_stage (the exact
    // metric the fbf867c ZeRO bug corrupted).
    let comm_zero0 = new_points
        .iter()
        .find(|(b, z, _, _)| *b == 1 && *z == 0)
        .unwrap()
        .3;
    let comm_zero3 = new_points
        .iter()
        .find(|(b, z, _, _)| *b == 1 && *z == 3)
        .unwrap()
        .3;
    assert!(
        (comm_zero0 - comm_zero3).abs() > 1e-9,
        "communication_overhead should respond to zero_stage"
    );
    println!(
        "[ok] communication_overhead correctly varies with zero_stage (stage 0: {:.2}%, stage 3: {:.2}%)",
        comm_zero0 * 100.0,
        comm_zero3 * 100.0
    );

    println!(
        "\nCall count for Architecture/Graph (the phases verified batch/zero/precision-invariant):"
    );
    println!(
        "  current sweep_hyperparameters: {} (once per point, all 9 phases)",
        old_result.points.len()
    );
    println!("  this PoC:                      1 (Architecture+Graph only; Tensor/Operator/Compute still run {} times, once per point, because they genuinely depend on batch_size)", new_points.len());
}
