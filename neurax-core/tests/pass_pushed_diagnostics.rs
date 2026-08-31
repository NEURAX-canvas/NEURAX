//! Regression test for a real bug: diagnostics a pass pushes directly via
//! `ctx.add_diagnostic()` (as opposed to the ones `generate_diagnostics()`
//! derives from final metrics) used to be written into
//! `NeuraxContext::diagnostics` and never read back out anywhere in the
//! codebase — a single `.lock()` call existed on that field, and it was the
//! write side. A custom layer with no FLOP equation, or one whose equation
//! failed to evaluate, produced no warning in the report at all despite the
//! code appearing to generate one (W001 / E003 respectively).
//!
//! Fixed by merging `ctx.diagnostics` into `report.diagnostics` in
//! `ReportPass::build_report()`. This test exercises the W001 path with a
//! real model through the real public API, not a unit test against the
//! pass in isolation, so it fails the same way a real user's report would
//! have.

use neurax_core::analyze_json;

#[test]
fn a_custom_layer_with_no_formula_produces_a_w001_warning_in_the_report() {
    let json = r#"{
        "schema_version": "1.0",
        "model": {
            "name": "custom-layer-test",
            "type": "transformer",
            "global_params": { "num_layers": 1, "sequence_length": 128, "hidden_size": 64 },
            "layers": [
                { "id": "emb", "layer_type": "embedding", "params": { "vocab_size": 1000, "embedding_dim": 64 } },
                { "id": "l1", "layer_type": "custom", "params": { "hidden_size": 64 } }
            ]
        },
        "training": { "batch_size": 1, "max_steps": 100 },
        "hardware": { "gpus": [{ "name": "A100-SXM", "count": 1 }] }
    }"#;

    let result = analyze_json(json).expect("analysis should succeed");

    let has_w001 = result
        .report
        .diagnostics
        .iter()
        .any(|d| matches!(d.code, neurax_ir::DiagnosticCode::W001));

    assert!(
        has_w001,
        "expected W001 (custom layer without a formula) in the report's diagnostics, got: {:?}",
        result
            .report
            .diagnostics
            .iter()
            .map(|d| d.code)
            .collect::<Vec<_>>()
    );
}
