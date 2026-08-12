//! A researcher must be able to describe an operator NEURAX has no built-in
//! formula for, and have it compile like any other block.
//!
//! Custom layers carry their own equations over the documented variables
//! (`B` batch, `S` sequence, `H` hidden, `D` head dim, `I` intermediate,
//! `V` vocab, `N` heads). Those equations used to be parsed with
//! `str::parse::<f64>()`, which succeeds only for a bare number, so any real
//! formula silently produced a block costing zero FLOPs and zero parameters —
//! and a model built only from such blocks failed outright.

use serde_json::{json, Value};

fn model_with_custom_block(equations: Value, params: Value) -> String {
    json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "novel-architecture", "type": "transformer",
            "global_params": {"num_layers": 1, "hidden_size": 1024,
                              "sequence_length": 512, "vocab_size": 32000},
            "layers": [{
                "id": "my_new_block",
                "layer_type": "custom",
                "params": params,
                "custom_equations": equations
            }]
        },
        "training": {"batch_size": 4, "sequence_length": 512, "precision": "bf16",
                     "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "count": 1}]},
        "data": {"dataset_size": 1000000}
    })
    .to_string()
}

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("custom block should parse");
    neurax_core::run_analysis(config).expect("custom block should analyze")
}

#[test]
fn custom_flop_equation_is_evaluated() {
    let r = analyze(&model_with_custom_block(
        json!({"flops_forward": "2 * B * S * H * H", "params": "4 * H * H"}),
        json!({"hidden_size": 1024}),
    ));
    // 2 x 4 x 512 x 1024 x 1024
    let expected = 2.0 * 4.0 * 512.0 * 1024.0 * 1024.0;
    let got = r.compute.metrics.forward_flops;
    assert!(
        (got - expected).abs() / expected < 1e-9,
        "expected {expected:e} FLOPs from the equation, got {got:e}"
    );
}

#[test]
fn custom_parameter_equation_is_evaluated() {
    let r = analyze(&model_with_custom_block(
        json!({"flops_forward": "2 * B * S * H", "params": "4 * H * H"}),
        json!({"hidden_size": 1024}),
    ));
    assert_eq!(r.arch.metrics.total_parameters, 4 * 1024 * 1024);
}

#[test]
fn an_explicit_parameter_count_wins_over_the_equation() {
    let r = analyze(&model_with_custom_block(
        json!({"flops_forward": "2 * B * S * H", "params": "4 * H * H"}),
        json!({"hidden_size": 1024, "param_count": 12345}),
    ));
    assert_eq!(r.arch.metrics.total_parameters, 12345);
}

#[test]
fn equations_can_use_every_documented_variable() {
    for equation in [
        "B * S * H", "D * N", "I * H", "V * H", "2 * B * S * I",
    ] {
        let r = analyze(&model_with_custom_block(
            json!({"flops_forward": equation, "params": "H * H"}),
            json!({"hidden_size": 1024, "num_heads": 16, "intermediate_size": 4096,
                   "vocab_size": 32000}),
        ));
        assert!(
            r.compute.metrics.forward_flops > 0.0,
            "equation `{equation}` should evaluate to a positive cost"
        );
    }
}

#[test]
fn a_malformed_equation_is_rejected_with_a_message_naming_it() {
    // A broken formula used to surface far downstream as an unexplained
    // "total FLOPs is zero"; it must be reported where it can be acted on.
    let json = model_with_custom_block(
        json!({"flops_forward": "2 * * B +", "params": "H * H"}),
        json!({"hidden_size": 1024}),
    );
    let error = neurax_parser::parse_model_config(&json)
        .expect_err("a malformed equation should be rejected")
        .to_string();

    assert!(
        error.contains("my_new_block") && error.contains("flops_forward"),
        "the error should name the layer and the field, got: {error}"
    );
    assert!(
        error.contains("2 * * B +"),
        "the error should quote the offending expression, got: {error}"
    );
}

#[test]
fn a_model_built_only_from_custom_blocks_compiles() {
    // The point of the feature: an architecture with no built-in counterpart.
    let json = json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "entirely-new", "type": "transformer",
            "global_params": {"num_layers": 4, "hidden_size": 768, "sequence_length": 256},
            "layers": [
                {"id": "novel_mix", "layer_type": "custom",
                 "params": {"hidden_size": 768},
                 "custom_equations": {"flops_forward": "6 * B * S * H * H", "params": "3 * H * H"}},
                {"id": "novel_gate", "layer_type": "custom",
                 "params": {"hidden_size": 768},
                 "custom_equations": {"flops_forward": "B * S * H", "params": "H"}}
            ]
        },
        "training": {"batch_size": 2, "sequence_length": 256, "precision": "bf16",
                     "learning_rate": 0.0003, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "count": 1}]},
        "data": {"dataset_size": 500000}
    })
    .to_string();

    let r = analyze(&json);
    assert!(r.arch.metrics.total_parameters > 0, "custom model should have parameters");
    assert!(r.compute.metrics.forward_flops > 0.0, "custom model should have a cost");
    assert!(r.memory.metrics.peak_vram_gb() > 0.0, "custom model should have a memory footprint");
    assert_eq!(r.operator.metrics.custom_op_count, 2, "both blocks should be marked custom");
}
