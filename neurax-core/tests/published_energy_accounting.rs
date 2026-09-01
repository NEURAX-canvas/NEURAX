//! Third external "Simulate" reference, covering a piece of the pipeline
//! with zero prior external validation: `CostPass`'s energy accounting
//! (`neurax-ir/src/cost/pass.rs`). Every existing cost test
//! (`internal_coherence.rs` group 5, `metric_realism.rs`) only checks that
//! `energy_kwh` is positive/finite — never against a real measured figure.
//!
//! Ground truth: Luccioni et al. 2023, "Estimating the Carbon Footprint of
//! BLOOM, a 176B Parameter Language Model" (JMLR; arXiv:2211.02001),
//! Table 1: BLOOM (176B parameters, 70 layers, hidden size 14336, 112
//! heads, vocab 250,680 — bigscience/bloom's own published architecture)
//! trained on **384 NVIDIA A100 80GB GPUs** for **118 days, 5h, 41min**
//! (2,837.68 hours), consuming **433,196 kWh** (dynamic GPU power only —
//! the paper's own headline energy number, distinct from its separate
//! "all processes including manufacturing" figure).
//!
//! What this test does and does not prove:
//!
//! NEURAX cannot predict BLOOM's real 118-day training time from its
//! architecture alone — that would require modeling the exact 3D
//! (pipeline + tensor + data) parallelism BLOOM actually used, which is a
//! materially larger feature than this compiler currently has (the
//! hardware pass models a single effective GPU count with a data-parallel
//! all-reduce term — see `published_hardware_scaling.rs` for what that
//! does and doesn't capture). So this test does not ask NEURAX to predict
//! the duration. Instead it supplies the real, published duration as an
//! input (by solving for the `max_steps` that makes NEURAX's own
//! per-step latency multiply out to the real 2,837.68 hours) and checks
//! the part that actually is fully specified and testable in isolation:
//! given a correct training duration, does `CostPass`'s energy formula
//! (`hours * gpu_tdp_watts * num_gpus / 1000 * pue_factor`) reproduce a
//! real, independently reported energy figure? `pue_factor` is set to 1.0
//! to match the paper's own "dynamic power only" accounting (no
//! datacenter-overhead multiplier) — NEURAX's own default is 1.2, a
//! reasonable general PUE assumption this specific comparison isn't
//! trying to reproduce.
//!
//! CO2 is deliberately not checked here: BLOOM trained on Jean Zay
//! (France's largely nuclear grid, ~57 g CO2/kWh by the paper's own
//! numbers — 24.7 tonnes / 433,196 kWh), while NEURAX's `co2_per_kwh`
//! (0.233 kg/kWh) is a generic, location-independent default. Comparing
//! them would penalize NEURAX for not knowing which country's grid a run
//! used, which is a genuinely different, unrelated gap from the energy
//! arithmetic this test targets.

fn bloom_json(num_gpus: u32, max_steps: u64) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{
                "name": "BLOOM-176B",
                "type": "transformer",
                "global_params": {{
                    "num_layers": 70, "hidden_size": 14336, "num_heads": 112,
                    "intermediate_size": 57344, "vocab_size": 250680, "sequence_length": 2048
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 250680, "hidden_size": 14336}}}},
                    {{"id": "layer_0", "layer_type": "attention", "params": {{"hidden_size": 14336, "num_attention_heads": 112, "intermediate_size": 57344}}}},
                    {{"id": "mlp_0", "layer_type": "mlp", "params": {{"hidden_size": 14336, "intermediate_size": 57344}}}}
                ]
            }},
            "training": {{
                "batch_size": 2048, "sequence_length": 2048, "precision": "bf16",
                "learning_rate": 0.00006, "max_steps": {max_steps}
            }},
            "hardware": {{
                "gpus": [{{"name": "A100-SXM", "memory_gb": 80, "count": {num_gpus}}}],
                "interconnect_bandwidth_gb_s": 200.0
            }},
            "cost_config": {{ "pue_factor": 1.0 }},
            "data": {{"dataset_size": 366000000000, "vocab_size": 250680, "num_classes": 0}}
        }}"#
    )
}

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

const REAL_TRAINING_HOURS: f64 = 118.0 * 24.0 + 5.0 + 41.0 / 60.0; // 2,837.68h
const REAL_ENERGY_KWH: f64 = 433_196.0;

#[test]
fn model_reproduces_the_real_176b_architecture() {
    let probe = analyze(&bloom_json(384, 1));
    let error = (probe.arch.metrics.total_parameters as f64 - 176.0e9).abs() / 176.0e9;
    assert!(
        error < 0.01,
        "expected ~176B params (BLOOM's real published size), got {}",
        probe.arch.metrics.total_parameters
    );
}

#[test]
fn energy_formula_reproduces_the_real_measured_figure_given_the_real_duration() {
    // Pass 1: read NEURAX's own predicted per-step latency for this config.
    let probe = analyze(&bloom_json(384, 1));
    let latency_ms = probe.hardware.metrics.latency_ms;
    assert!(latency_ms > 0.0, "latency should be positive");

    // Solve for the max_steps that makes max_steps * latency_ms equal the
    // real measured training duration — supplying the real duration as an
    // input, not asking NEURAX to predict it (see module doc comment).
    let real_seconds = REAL_TRAINING_HOURS * 3600.0;
    let max_steps = (real_seconds * 1000.0 / latency_ms).round() as u64;

    // Pass 2: with training_time_hours now equal to the real duration by
    // construction, check the part actually under test — the energy
    // formula itself.
    let r = analyze(&bloom_json(384, max_steps));
    let hours_error =
        (r.cost.metrics.training_time_hours - REAL_TRAINING_HOURS).abs() / REAL_TRAINING_HOURS;
    assert!(
        hours_error < 0.01,
        "sanity check: training_time_hours should match the real duration by \
         construction, got {} vs {REAL_TRAINING_HOURS}",
        r.cost.metrics.training_time_hours
    );

    let energy_error = (r.cost.metrics.energy_kwh - REAL_ENERGY_KWH).abs() / REAL_ENERGY_KWH;
    assert!(
        energy_error < 0.02,
        "predicted {:.0} kWh vs BLOOM's real published 433,196 kWh (dynamic power \
         only, PUE=1.0) — {:.1}% off, expected the energy formula (hours * TDP * \
         gpu_count) to reproduce this closely given a correct duration and the \
         real A100-80GB TDP (400W)",
        r.cost.metrics.energy_kwh,
        energy_error * 100.0
    );
}
