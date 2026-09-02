//! Real bug found auditing 44+ real-model scenarios: with no
//! `training.max_steps` and no `training.num_epochs` + `data.dataset_size`
//! to derive one from, `training_cost_usd`/`energy_kwh`/`co2_kg` all report
//! exactly `$0.00` — a confident-looking computed answer, not "unknown".
//! Every one of the audit's 44 scenarios hit this, since none of them set
//! a training budget. Fixed via `ReportMetrics::training_budget_stated`
//! (and its JSON twin, `CostMetricsOutput::training_budget_stated`), which
//! a renderer must check before presenting those zeros as real.

use neurax_core::analyze_json;

fn gpt2_small_json(training_extra: &str, data_extra: &str) -> String {
    format!(
        r#"{{
            "schema_version": "1.0",
            "model": {{
                "name": "gpt2-small",
                "type": "transformer",
                "global_params": {{ "num_layers": 12, "hidden_size": 768, "sequence_length": 1024, "num_heads": 12 }},
                "layers": [
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 768, "num_heads": 12}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 768, "intermediate_size": 3072}}}}
                ]
            }},
            "training": {{"batch_size": 1{training_extra}}},
            "data": {{"input_shape": [1, 1024], "dtype": "fp16"{data_extra}}},
            "hardware": {{"gpus": [{{"name": "A100", "count": 1}}]}}
        }}"#
    )
}

#[test]
fn no_training_budget_is_reported_as_not_stated_not_free() {
    let result = analyze_json(&gpt2_small_json("", "")).expect("analyzes");
    assert_eq!(result.cost.effective_steps, 0);
    assert!(!result.report.metrics.training_budget_stated);
    // The underlying numbers stay 0.0 (nothing computed them) — the fix is
    // the flag, not a different number, so a renderer can tell "$0.00 is
    // the real answer" from "there is no answer yet" instead of guessing.
    assert_eq!(result.report.metrics.training_cost_usd, 0.0);

    let markdown = neurax_ir::report::format_markdown(&result.report);
    assert!(
        markdown.contains("N/A — no training budget stated"),
        "the report must not present a fabricated $0.00 as a real answer:\n{markdown}"
    );
    assert!(
        !markdown.contains("$0.00"),
        "a bare $0.00 with no budget stated reads as a computed answer:\n{markdown}"
    );
}

#[test]
fn an_explicit_max_steps_is_reported_as_a_real_cost() {
    let result = analyze_json(&gpt2_small_json(r#", "max_steps": 10000"#, "")).expect("analyzes");
    assert_eq!(result.cost.effective_steps, 10000);
    assert!(result.report.metrics.training_budget_stated);
    assert!(result.report.metrics.training_cost_usd > 0.0);

    let markdown = neurax_ir::report::format_markdown(&result.report);
    assert!(
        !markdown.contains("N/A — no training budget stated"),
        "a real, stated training budget must not be reported as absent:\n{markdown}"
    );
}

#[test]
fn epochs_and_dataset_size_derive_a_real_budget_too() {
    let result = analyze_json(&gpt2_small_json(
        r#", "num_epochs": 3"#,
        r#", "dataset_size": 1000000000"#,
    ))
    .expect("analyzes");
    assert!(result.cost.effective_steps > 0);
    assert!(result.report.metrics.training_budget_stated);
    assert!(result.report.metrics.training_cost_usd > 0.0);
}
