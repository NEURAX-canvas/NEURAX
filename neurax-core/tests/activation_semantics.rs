//! Checks that the compiler understands what an activation *does*, not just
//! that it accepts the name.
//!
//! An activation affects analysis in two ways: its element-wise cost, and —
//! for the gated family — the number of weight matrices in the feed-forward
//! block. Both must show up in the reported numbers.

use neurax_core::{run_analysis, AnalysisResult};
use serde_json::json;

fn analyze_mlp_with(activation: &str, gated_flag: Option<bool>) -> AnalysisResult {
    let mut params = json!({
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "activation": activation
    });
    if let Some(g) = gated_flag {
        params["gated"] = json!(g);
    }
    let config = json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "mlp-probe", "type": "transformer",
            "global_params": {"num_layers": 1, "hidden_size": 1024, "sequence_length": 512,
                              "vocab_size": 32000},
            "layers": [{"id": "ffn", "layer_type": "mlp", "params": params}]
        },
        "training": {"batch_size": 4, "sequence_length": 512, "precision": "bf16",
                     "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "count": 1}]},
        "data": {"dataset_size": 1000000, "vocab_size": 32000}
    });
    let parsed = neurax_parser::parse_model_config(&config.to_string())
        .unwrap_or_else(|e| panic!("{activation} config should parse: {e}"));
    run_analysis(parsed).unwrap_or_else(|e| panic!("{activation} config should analyze: {e}"))
}

fn ffn_params(r: &AnalysisResult) -> u64 {
    *r.arch
        .metrics
        .params_per_layer
        .get("ffn")
        .expect("ffn layer should be reported")
}

#[test]
fn gated_activation_implies_a_three_matrix_feed_forward() {
    // SwiGLU splits the input across a gate and an up projection, so the block
    // is three matrices rather than two — 1.5x the parameters. Asking for
    // SwiGLU must be enough; a separate `gated` flag should not be required.
    let plain = ffn_params(&analyze_mlp_with("silu", None));
    let swiglu = ffn_params(&analyze_mlp_with("swiglu", None));

    let ratio = swiglu as f64 / plain as f64;
    assert!(
        (ratio - 1.5).abs() < 0.01,
        "swiglu should have 1.5x the parameters of a plain MLP, got {ratio:.3}x \
         ({swiglu} vs {plain})"
    );
}

#[test]
fn gated_activation_and_explicit_flag_agree() {
    let by_activation = ffn_params(&analyze_mlp_with("swiglu", None));
    let by_flag = ffn_params(&analyze_mlp_with("silu", Some(true)));
    assert_eq!(
        by_activation, by_flag,
        "stating the gate via the activation or via the flag must give the same model"
    );
}

#[test]
fn activation_cost_is_reflected_in_flops() {
    // GELU costs an order of magnitude more per element than ReLU, so the same
    // block must not report identical FLOPs for both.
    let relu = analyze_mlp_with("relu", None).compute.metrics.forward_flops;
    let gelu = analyze_mlp_with("gelu", None).compute.metrics.forward_flops;
    let none = analyze_mlp_with("none", None).compute.metrics.forward_flops;

    assert!(
        gelu > relu,
        "gelu ({gelu:e}) should cost more than relu ({relu:e})"
    );
    assert!(
        relu > none,
        "relu ({relu:e}) should cost more than no activation ({none:e})"
    );
}

#[test]
fn every_activation_offered_by_the_ui_is_understood() {
    // The picker must not offer a name the cost model cannot price, or the
    // choice would silently fall back to a generic estimate.
    for name in [
        "relu", "leaky_relu", "relu6", "sigmoid", "tanh", "silu", "swish", "gelu", "gelu_tanh",
        "gelu_new", "softplus", "mish", "elu", "selu", "hard_swish", "hard_sigmoid", "glu",
        "swiglu", "geglu", "reglu", "none",
    ] {
        assert!(
            neurax_formulas::activation::activation_spec(name).is_some(),
            "{name} should be known to the cost model"
        );
    }
}

#[test]
fn analysis_succeeds_for_every_known_activation() {
    for name in ["relu", "gelu", "silu", "tanh", "mish", "swiglu", "geglu", "reglu", "none"] {
        let r = analyze_mlp_with(name, None);
        assert!(
            r.compute.metrics.forward_flops > 0.0,
            "{name} should produce positive FLOPs"
        );
    }
}
