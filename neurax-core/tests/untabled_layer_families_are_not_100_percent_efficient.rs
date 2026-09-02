//! Real bug found via the "over- vs under-estimate" audit: a layer family
//! with no entry in `hardware/pass.rs`'s efficiency table (every SSM, RNN,
//! GNN, and GAN type) defaulted to `gpu_efficiency = 1.0` — no real GPU
//! runs any kernel at 100% of peak throughput, so this silently
//! under-estimated compute time (and everything latency/cost derive from
//! it) for entire families NEURAX supports. Fixed by reusing
//! `conv_efficiency` (the lowest tabled tier, and the one with a real
//! published calibration point) instead of `1.0` when nothing is
//! recognized.

use neurax_core::analyze_json;

fn single_layer_json(layer_type: &str, params: &str) -> String {
    format!(
        r#"{{
            "schema_version": "1.0",
            "model": {{
                "name": "single-layer",
                "type": "transformer",
                "global_params": {{ "hidden_size": 2048, "sequence_length": 2048 }},
                "layers": [
                    {{"id": "l", "layer_type": "{layer_type}", "params": {params}}}
                ]
            }},
            "training": {{"batch_size": 1, "precision": "fp16"}},
            "hardware": {{"gpus": [{{"name": "A100", "count": 1}}]}}
        }}"#
    )
}

#[test]
fn a_pure_ssm_model_reports_meaningfully_lower_effective_tflops_than_a_pure_mlp_one() {
    // Same hidden_size, so roughly comparable FLOPs magnitude — the
    // difference in reported `effective_tflops` should come from the
    // efficiency factor, not a wildly different amount of raw work.
    let mlp = analyze_json(&single_layer_json(
        "mlp",
        r#"{"hidden_size": 2048, "intermediate_size": 8192}"#,
    ))
    .expect("analyzes");
    let mamba = analyze_json(&single_layer_json(
        "mamba_block",
        r#"{"hidden_size": 2048, "state_dim": 16, "expansion_factor": 2}"#,
    ))
    .expect("analyzes");

    assert!(mlp.hardware.metrics.effective_tflops > 0.0);
    assert!(mamba.hardware.metrics.effective_tflops > 0.0);
    // Before this fix, an untabled family (SSM) defaulted to the *same*
    // 1.0 "no penalty" efficiency an MLP layer only approaches at its best
    // tabled precision tier (0.92 at fp8) — an SSM layer reporting
    // efficiency on par with or above a real MLP layer's tabled value
    // would mean the bug is back.
    assert!(
        mamba.hardware.metrics.effective_tflops < mlp.hardware.metrics.effective_tflops,
        "an untabled family (SSM) must not report equal-or-better effective throughput \
         than a tabled one (MLP): mamba={}, mlp={}",
        mamba.hardware.metrics.effective_tflops,
        mlp.hardware.metrics.effective_tflops
    );
}
