//! Guards on the numbers the compiler reports.
//!
//! Every case here corresponds to a figure that used to come out zero,
//! placeholder-derived, or inconsistent between passes for the shipped GPT-4
//! example — a config that names a real GPU and states a training budget.

use neurax_core::{run_analysis, AnalysisResult};

fn gpt4() -> AnalysisResult {
    let json = std::fs::read_to_string("../examples/models/gpt4.json")
        .expect("examples/models/gpt4.json should exist");
    let config = neurax_parser::parse_model_config(&json).expect("gpt4 example should parse");
    run_analysis(config).expect("gpt4 example should analyze")
}

#[test]
fn named_gpu_resolves_to_its_real_specs() {
    let r = gpt4();
    // The config says `"name": "H100"` and states no TFLOPS of its own, so the
    // hardware database must supply them. A generic fallback profile (100
    // TFLOPS / 1000 GB/s) means the lookup missed.
    assert_eq!(r.hardware.gpu_profile.name, "H100-SXM");
    assert!(
        r.hardware.gpu_profile.peak_tflops > 500.0,
        "expected real H100 throughput, got {} TFLOPS",
        r.hardware.gpu_profile.peak_tflops
    );
    assert!(
        r.hardware.gpu_profile.memory_bandwidth > 3000.0,
        "expected real H100 bandwidth, got {} GB/s",
        r.hardware.gpu_profile.memory_bandwidth
    );
}

#[test]
fn parameter_memory_matches_the_parameter_count_and_dtype() {
    let r = gpt4();
    let params = r.arch.metrics.total_parameters;
    let bytes = r.memory.metrics.parameter_memory_bytes;
    assert!(params > 0, "parameter count should not be zero");

    // The example trains in bf16, so parameter storage is exactly 2 bytes each.
    // The architecture and memory passes used to scale the listed layers
    // differently, which showed up here as ~46 bytes per parameter.
    let bytes_per_param = bytes as f64 / params as f64;
    assert!(
        (bytes_per_param - 2.0).abs() < 0.01,
        "expected 2 bytes per parameter in bf16, got {bytes_per_param:.2}"
    );
}

#[test]
fn training_cost_energy_and_carbon_are_populated() {
    let r = gpt4();
    // The example sets `num_epochs` and `dataset_size` but no `max_steps`.
    // Without deriving the step count these were all structurally zero.
    for (label, value) in [
        ("training_time_hours", r.cost.metrics.training_time_hours),
        ("training_cost_usd", r.cost.metrics.training_cost_usd),
        ("energy_kwh", r.cost.metrics.energy_kwh),
        ("co2_kg", r.cost.metrics.co2_kg),
        ("gpu_hours_total", r.cost.metrics.gpu_hours_total),
    ] {
        assert!(
            value > 0.0 && value.is_finite(),
            "{label} should be a positive finite number, got {value}"
        );
    }
}

#[test]
fn step_latency_is_derived_from_the_model_not_a_constant() {
    let r = gpt4();
    let latency = r.hardware.metrics.latency_ms;
    assert!(
        latency > 0.0 && latency.is_finite(),
        "latency should be positive and finite, got {latency}"
    );
    // The cost pass used to assume a flat 100 ms per step regardless of model
    // and GPU; a real analysis of this model lands nowhere near that.
    assert!(
        (latency - 100.0).abs() > 1.0,
        "latency looks like the old hard-coded 100 ms placeholder"
    );
}

#[test]
fn throughput_follows_from_latency() {
    let r = gpt4();
    assert!(
        r.hardware.metrics.throughput_tokens_per_s > 0.0,
        "throughput should be positive once latency is real"
    );
}

#[test]
fn reported_depth_and_parameter_count_describe_the_same_model() {
    let r = gpt4();
    // The example lists a handful of representative blocks and sets
    // `num_layers: 120`. Reporting that depth alongside the parameter count of
    // only the listed blocks understated the model by more than an order of
    // magnitude.
    assert_eq!(r.arch.metrics.num_layers, 120);
    assert!(
        r.arch.metrics.total_parameters > 50_000_000_000,
        "a 120-layer, 12288-wide model should be tens of billions of parameters, got {}",
        r.arch.metrics.total_parameters
    );
}
