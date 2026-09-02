//! A block dragged onto the canvas but never wired to anything else used to
//! be silently costed like any other layer — its params/FLOPs/memory count
//! toward every metric even though it contributes nothing to the model's
//! actual data flow. GraphPass now flags it as W008. Idea ported from
//! LIFT's dead-code-elimination pass (`lift-opt::dce.rs`'s "result never
//! consumed" check) and adapted to flag rather than silently delete —
//! NEURAX never rewrites a user's design, only reports on it.

use neurax_core::analyze_json;
use neurax_ir::DiagnosticCode;

const ORPHANED_LAYER_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "orphaned-layer-test",
        "type": "transformer",
        "global_params": { "num_layers": 1, "sequence_length": 64, "hidden_size": 128 },
        "layers": [
            { "id": "emb", "layer_type": "embedding", "params": { "vocab_size": 1000, "embedding_dim": 128 } },
            { "id": "attn", "layer_type": "attention", "params": {} },
            { "id": "orphan", "layer_type": "dense", "params": { "out_features": 128 } }
        ],
        "connections": [{ "from": "emb", "to": "attn" }]
    },
    "training": { "batch_size": 1, "max_steps": 100 },
    "hardware": { "gpus": [{ "name": "A100-SXM", "count": 1 }] }
}
"#;

const FULLY_CONNECTED_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "fully-connected-test",
        "type": "transformer",
        "global_params": { "num_layers": 1, "sequence_length": 64, "hidden_size": 128 },
        "layers": [
            { "id": "emb", "layer_type": "embedding", "params": { "vocab_size": 1000, "embedding_dim": 128 } },
            { "id": "attn", "layer_type": "attention", "params": {} }
        ],
        "connections": [{ "from": "emb", "to": "attn" }]
    },
    "training": { "batch_size": 1, "max_steps": 100 },
    "hardware": { "gpus": [{ "name": "A100-SXM", "count": 1 }] }
}
"#;

#[test]
fn a_disconnected_layer_produces_a_w008_warning_naming_it() {
    let result = analyze_json(ORPHANED_LAYER_JSON).expect("analysis should succeed");

    let w008 = result
        .report
        .diagnostics
        .iter()
        .find(|d| matches!(d.code, DiagnosticCode::W008));

    let w008 = w008.unwrap_or_else(|| {
        panic!(
            "expected W008 (orphaned layer) in the report's diagnostics, got: {:?}",
            result
                .report
                .diagnostics
                .iter()
                .map(|d| d.code)
                .collect::<Vec<_>>()
        )
    });

    assert_eq!(w008.layer_id.as_deref(), Some("orphan"));
}

#[test]
fn a_fully_connected_design_produces_no_w008_warning() {
    let result = analyze_json(FULLY_CONNECTED_JSON).expect("analysis should succeed");

    let has_w008 = result
        .report
        .diagnostics
        .iter()
        .any(|d| matches!(d.code, DiagnosticCode::W008));

    assert!(
        !has_w008,
        "a design with every layer connected should not trigger W008, got: {:?}",
        result
            .report
            .diagnostics
            .iter()
            .map(|d| d.code)
            .collect::<Vec<_>>()
    );
}
