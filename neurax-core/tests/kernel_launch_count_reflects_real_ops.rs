//! Regression test for the fourth "Simulate" bug found in this audit:
//! `kernel_launch_count` was a flat `layers.len() * 2` guess, unrelated to
//! how many atomic operations a layer actually decomposes into (an
//! attention layer is 2 real ops — attention + RoPE — while an MLP layer
//! is 1). Once kernel launch overhead started actually affecting latency
//! (see `roofline_overlap_and_kernel_overhead.rs`), that guess mattered
//! more than a cosmetic reported number. Fixed to use
//! `HardwareIR::per_layer_timings.len()`, which already carries one entry
//! per real atomic op from the operator/compute passes.

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

#[test]
fn kernel_count_matches_the_real_op_count_not_a_flat_per_layer_guess() {
    // 2 JSON layers: attention (decomposes into 2 ops: attention + RoPE)
    // and mlp (decomposes into 1 op). Real total: 3. The old flat guess
    // (layers.len() * 2) would have reported 4.
    let json = r#"{
        "schema_version": "1.0",
        "model": { "name": "kernel-count-test", "type": "transformer",
            "global_params": { "num_layers": 1, "hidden_size": 512, "sequence_length": 128, "num_heads": 8 },
            "layers": [
                {"id": "attn", "layer_type": "attention", "params": {"hidden_size": 512, "num_heads": 8}},
                {"id": "mlp", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 2048}}
            ]
        },
        "training": {"batch_size": 1, "max_steps": 100, "precision": "fp16"},
        "hardware": {"gpus": [{"name": "A100-SXM", "count": 1}]}
    }"#;
    let r = analyze(json);

    assert_eq!(
        r.hardware.metrics.kernel_launch_count, 3,
        "expected the real op count (attention=2 ops + mlp=1 op = 3), \
         not the old flat layers.len()*2 guess (which would be 4)"
    );
}
