//! Compares the current compiler (`neurax_core::analyze_json`) against this
//! PoC's query-graph orchestration (`neurax_ir_poc::StructuralFacts`/`View`)
//! on one real reference model per family, for the 5 target metrics.
//!
//! Both sides call the exact same, unmodified passes — this only tests
//! whether restructuring how they're wired changes any result.

use neurax_ir_poc::{StructuralFacts, View};
use std::fs;

struct Row {
    family: &'static str,
    model: &'static str,
    old_params: u64,
    new_params: u64,
    old_flops: f64,
    new_flops: f64,
    old_vram: u64,
    new_vram: u64,
    old_tflops: f64,
    new_tflops: f64,
    old_comm: f64,
    new_comm: f64,
}

fn rel_diff(a: f64, b: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        return 0.0;
    }
    (a - b).abs() / a.abs().max(b.abs()).max(1.0)
}

fn main() {
    let models: &[(&str, &str)] = &[
        ("transformer", "examples/models/gpt3_175b.json"),
        ("moe", "examples/models/mixtral_8x7b.json"),
        ("cnn", "examples/models/resnet50.json"),
        ("ssm", "examples/models/mamba_2.8b.json"),
        ("diffusion", "examples/models/sdxl_1.0.json"),
        ("gan", "examples/models/dcgan.json"),
        ("rnn", "examples/models/awd_lstm_ptb.json"),
        ("gnn", "examples/models/gcn_ogbn_arxiv.json"),
    ];

    let mut rows = Vec::new();
    let mut any_mismatch = false;

    for (family, path) in models {
        let json = fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));

        // ── Current compiler ──
        let old = neurax_core::analyze_json(&json)
            .unwrap_or_else(|e| panic!("{family}: neurax_core::analyze_json failed: {e}"));

        // ── This PoC ──
        let config = neurax_parser::parse_model_config(&json)
            .unwrap_or_else(|e| panic!("{family}: parse_model_config failed: {e}"));
        let facts = StructuralFacts::compute(&config)
            .unwrap_or_else(|e| panic!("{family}: StructuralFacts::compute failed: {e}"));
        let view = View::new(&facts, config);

        let row = Row {
            family,
            model: path.rsplit('/').next().unwrap_or(path),
            old_params: old.arch.metrics.total_parameters,
            new_params: view.total_parameters(),
            old_flops: old.compute.metrics.total_flops,
            new_flops: view.total_flops().expect("total_flops"),
            old_vram: old.memory.metrics.peak_vram_bytes,
            new_vram: view.peak_vram_bytes().expect("peak_vram_bytes"),
            old_tflops: old.hardware.metrics.effective_tflops,
            new_tflops: view.effective_tflops().expect("effective_tflops"),
            old_comm: old.parallelism.metrics.communication_overhead,
            new_comm: view
                .communication_overhead()
                .expect("communication_overhead"),
        };

        let mismatch = rel_diff(row.old_params as f64, row.new_params as f64) > 1e-9
            || rel_diff(row.old_flops, row.new_flops) > 1e-9
            || rel_diff(row.old_vram as f64, row.new_vram as f64) > 1e-9
            || rel_diff(row.old_tflops, row.new_tflops) > 1e-9
            || rel_diff(row.old_comm, row.new_comm) > 1e-9;
        any_mismatch |= mismatch;

        rows.push((row, mismatch));
    }

    println!(
        "{:<12} {:<22} {:>8} {:>16} {:>10} {:>10} {:>10}",
        "family", "model", "params", "flops", "vram_GB", "tflops", "comm%"
    );
    for (row, mismatch) in &rows {
        let flag = if *mismatch { "  <-- MISMATCH" } else { "" };
        println!(
            "{:<12} {:<22} old={:>10} old={:>12.3e} old={:>7.3} old={:>7.1} old={:>6.2}{}",
            row.family,
            row.model,
            row.old_params,
            row.old_flops,
            row.old_vram as f64 / 1e9,
            row.old_tflops,
            row.old_comm * 100.0,
            flag
        );
        println!(
            "{:<12} {:<22} new={:>10} new={:>12.3e} new={:>7.3} new={:>7.1} new={:>6.2}",
            "",
            "",
            row.new_params,
            row.new_flops,
            row.new_vram as f64 / 1e9,
            row.new_tflops,
            row.new_comm * 100.0
        );
    }

    println!();
    if any_mismatch {
        println!("RESULT: at least one metric diverged between the current compiler and this PoC.");
        std::process::exit(1);
    } else {
        println!(
            "RESULT: all 5 metrics match exactly across all {} families — the orchestration change alone produced zero difference in output.",
            rows.len()
        );
    }
}
