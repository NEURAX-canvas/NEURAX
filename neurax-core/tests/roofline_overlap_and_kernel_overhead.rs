//! Regression test for a second "Simulate" bug found in the same audit as
//! `memory_time_reflects_real_bytes.rs`: `HardwarePass` already builds an
//! "Industrial" roofline model (`overlap_factor: 0.3`,
//! `kernel_launch_overhead_us: 5.0`) and already computes
//! `kernel_launch_count`, but neither ever reached `latency_ms` — the
//! aggregate combination was a plain
//! `compute_time.max(memory_time) + communication_overhead`, which
//! implicitly assumes compute and memory overlap perfectly (no real GPU
//! does) and adds zero cost for kernel launches no matter how many small
//! kernels the model has. `hardware_corrections.rs`'s F05 tests claimed to
//! check exactly this ("overhead significant for small batch") without
//! ever calling the real pipeline.
//!
//! Fixed: latency now adds `min(compute, memory) * (1 - overlap_factor)`
//! on top of the max, plus `kernel_launch_overhead_us * kernel_launch_count`.

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

const MANY_TINY_LAYERS: &str = r#"{
    "schema_version": "1.0",
    "model": { "name": "many-layers", "type": "transformer",
        "global_params": { "num_layers": 1, "hidden_size": 512, "sequence_length": 128 },
        "layers": [
            {"id": "l1", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 512}},
            {"id": "l2", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 512}},
            {"id": "l3", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 512}},
            {"id": "l4", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 512}}
        ]
    },
    "training": {"batch_size": 1, "max_steps": 100, "precision": "fp16"},
    "hardware": {"gpus": [{"name": "A100-SXM", "count": 1}]}
}"#;

#[test]
fn latency_exceeds_the_pre_fix_plain_max_formula() {
    let r = analyze(MANY_TINY_LAYERS);

    // Replicates the exact pre-fix aggregate formula so the comparison
    // isolates the overlap/kernel-overhead fix specifically, rather than
    // picking up the (pre-existing, unrelated) gpu_efficiency division —
    // this model is pure MLP layers, so that division alone would already
    // make a naive sum of `per_layer_timings` smaller than the internal
    // compute_time_ms, which would make a naive "no fix" comparison pass
    // for the wrong reason.
    let mlp_efficiency_fp16 = 0.85; // hardware/pass.rs's own documented table
    let old_compute_time_ms: f64 = r
        .hardware
        .per_layer_timings
        .iter()
        .map(|t| t.compute_time_ms / mlp_efficiency_fp16)
        .sum();
    let old_memory_time_ms: f64 = r
        .hardware
        .per_layer_timings
        .iter()
        .map(|t| t.memory_time_ms) // no attention layers: FlashAttention factor is 1.0
        .sum();
    let old_style_latency_ms = old_compute_time_ms.max(old_memory_time_ms); // single GPU: no comm overhead

    assert!(
        r.hardware.metrics.latency_ms > old_style_latency_ms,
        "latency ({}) should exceed the pre-fix plain-max formula ({old_style_latency_ms}) \
         now that overlap (<100%) and kernel launch overhead are actually applied",
        r.hardware.metrics.latency_ms
    );
}

#[test]
fn kernel_launch_overhead_is_a_real_component_of_latency() {
    let r = analyze(MANY_TINY_LAYERS);

    let kernel_count = r.hardware.metrics.kernel_launch_count;
    assert!(kernel_count > 0, "this model should report kernel launches");

    let expected_kernel_overhead_ms = 5.0 * kernel_count as f64 / 1000.0;
    assert!(
        r.hardware.metrics.latency_ms >= expected_kernel_overhead_ms,
        "latency ({}) should be at least the kernel launch overhead alone \
         ({expected_kernel_overhead_ms}ms for {kernel_count} kernels)",
        r.hardware.metrics.latency_ms
    );
}
