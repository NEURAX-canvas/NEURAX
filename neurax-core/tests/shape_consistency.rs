//! Cross-layer shape consistency diagnostic (W007).
//!
//! Regression coverage for a real bug: an agent-built transformer had a
//! positional_encoding/output_head layer whose declared input shape did not
//! match what the layer immediately before it actually produced. Before this
//! check, that inconsistency was only ever a `tracing::warn!` line in the
//! service's own log — invisible both to the UI's diagnostics panel and to
//! the agent's own pre-flight self-check (`measure_and_check`), so nothing
//! downstream of the mismatch was ever flagged.

use neurax_core::analyze_json;

const MISMATCHED_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Mismatched-Chain",
        "type": "transformer",
        "layers": [
            {
                "id": "attn_0",
                "layer_type": "attention",
                "input_shape": [4, 128, 768],
                "output_shape": [4, 128, 768],
                "params": { "hidden_size": 768, "num_attention_heads": 12 }
            },
            {
                "id": "mlp_0",
                "layer_type": "mlp",
                "input_shape": [4, 128, 512],
                "output_shape": [4, 128, 512],
                "params": { "hidden_size": 512 }
            }
        ],
        "global_params": { "hidden_size": 768, "num_layers": 2, "vocab_size": 32000 }
    },
    "training": { "batch_size": 4, "precision": "fp16" },
    "hardware": { "gpus": [{ "name": "A100-80GB", "count": 1 }] },
    "data": { "dtype": "fp16" }
}
"#;

const COHERENT_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Coherent-Chain",
        "type": "transformer",
        "layers": [
            {
                "id": "attn_0",
                "layer_type": "attention",
                "input_shape": [4, 128, 768],
                "output_shape": [4, 128, 768],
                "params": { "hidden_size": 768, "num_attention_heads": 12 }
            },
            {
                "id": "mlp_0",
                "layer_type": "mlp",
                "input_shape": [4, 128, 768],
                "output_shape": [4, 128, 768],
                "params": { "hidden_size": 768 }
            }
        ],
        "global_params": { "hidden_size": 768, "num_layers": 2, "vocab_size": 32000 }
    },
    "training": { "batch_size": 4, "precision": "fp16" },
    "hardware": { "gpus": [{ "name": "A100-80GB", "count": 1 }] },
    "data": { "dtype": "fp16" }
}
"#;

#[test]
fn flags_a_declared_input_shape_that_does_not_match_the_previous_layers_output() {
    let result = analyze_json(MISMATCHED_JSON).expect("analysis should succeed");
    let hit = result
        .report
        .diagnostics
        .iter()
        .find(|d| matches!(d.code, neurax_ir::DiagnosticCode::W007));

    let d = hit.expect("expected a W007 shape-mismatch diagnostic");
    assert_eq!(d.layer_id.as_deref(), Some("mlp_0"));
    assert!(
        d.message.contains("attn_0") && d.message.contains("mlp_0"),
        "message should name both layers: {}",
        d.message
    );
}

#[test]
fn stays_silent_on_a_chain_whose_shapes_actually_line_up() {
    let result = analyze_json(COHERENT_JSON).expect("analysis should succeed");
    let hit = result
        .report
        .diagnostics
        .iter()
        .any(|d| matches!(d.code, neurax_ir::DiagnosticCode::W007));
    assert!(
        !hit,
        "a coherent chain should not raise a shape-mismatch diagnostic"
    );
}
