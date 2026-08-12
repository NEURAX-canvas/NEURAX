//! Regression tests for malformed / adversarial model configurations.
//!
//! `/analyze` accepts model JSON straight from the browser, so every input
//! below was reachable from a request. Each case previously either panicked
//! (taking down the request handler) or was silently accepted and reported
//! wrapped, badly wrong numbers.

use serde_json::{json, Value};

/// A minimal well-formed transformer config, used as the base for mutations.
fn base() -> Value {
    json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "probe",
            "type": "transformer",
            "global_params": {
                "num_layers": 2, "hidden_size": 768, "num_heads": 12,
                "intermediate_size": 3072, "vocab_size": 50000, "sequence_length": 1024
            },
            "layers": [
                {"id": "embed", "layer_type": "embedding",
                 "params": {"vocab_size": 50000, "hidden_size": 768}},
                {"id": "l0", "layer_type": "attention",
                 "params": {"hidden_size": 768, "num_heads": 12, "intermediate_size": 3072}}
            ]
        },
        "training": {"batch_size": 8, "sequence_length": 1024, "precision": "bf16",
                     "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "memory_gb": 80, "count": 8}]},
        "data": {"dataset_size": 1000000, "vocab_size": 50000, "num_classes": 0}
    })
}

/// Assert the config is rejected with an error naming `expected_field`, and
/// that it never panics on the way there.
fn assert_rejected(case: &str, config: Value, expected_field: &str) {
    let parsed = neurax_parser::parse_model_config(&config.to_string());

    let err = match parsed {
        Err(e) => e.to_string(),
        // Rejection at parse time is the goal, but if a future change lets the
        // config through, it must still complete without panicking.
        Ok(cfg) => {
            let analysis = std::panic::catch_unwind(move || neurax_core::run_analysis(cfg));
            panic!(
                "{case}: expected rejection naming `{expected_field}`, but the config was \
                 accepted (analysis panicked: {})",
                analysis.is_err()
            );
        }
    };

    assert!(
        err.contains(expected_field),
        "{case}: expected the error to name `{expected_field}`, got: {err}"
    );
}

#[test]
fn baseline_config_is_accepted() {
    let cfg = neurax_parser::parse_model_config(&base().to_string())
        .expect("baseline config should parse");
    neurax_core::run_analysis(cfg).expect("baseline config should analyze");
}

#[test]
fn zero_head_count_is_rejected() {
    // Was: `hidden / heads` divided by zero and panicked, in debug and release.
    let mut v = base();
    v["model"]["global_params"]["num_heads"] = json!(0);
    v["model"]["layers"][1]["params"]["num_heads"] = json!(0);
    assert_rejected("num_heads = 0", v, "num_heads");
}

#[test]
fn oversized_hidden_size_is_rejected() {
    // Was: silently accepted in release, reporting a wrapped parameter count
    // roughly ten orders of magnitude below the true value.
    let mut v = base();
    v["model"]["global_params"]["hidden_size"] = json!(u64::MAX);
    v["model"]["layers"][1]["params"]["hidden_size"] = json!(u64::MAX);
    assert_rejected("hidden_size = u64::MAX", v, "hidden_size");
}

#[test]
fn oversized_sequence_length_is_rejected() {
    let mut v = base();
    v["model"]["global_params"]["sequence_length"] = json!(u64::MAX);
    assert_rejected("sequence_length = u64::MAX", v, "sequence_length");
}

#[test]
fn oversized_vocab_size_is_rejected() {
    let mut v = base();
    v["model"]["global_params"]["vocab_size"] = json!(u64::MAX);
    assert_rejected("vocab_size = u64::MAX", v, "vocab_size");
}

#[test]
fn oversized_layer_count_is_rejected() {
    let mut v = base();
    v["model"]["global_params"]["num_layers"] = json!(u32::MAX);
    assert_rejected("num_layers = u32::MAX", v, "num_layers");
}

#[test]
fn kv_heads_exceeding_head_count_is_rejected() {
    let mut v = base();
    v["model"]["layers"][1]["params"]["num_kv_heads"] = json!(99999);
    assert_rejected("num_kv_heads > num_heads", v, "num_kv_heads");
}

#[test]
fn head_count_not_dividing_hidden_size_is_rejected() {
    // 768 / 7 truncates, silently changing the modelled head dimension.
    let mut v = base();
    v["model"]["layers"][1]["params"]["num_heads"] = json!(7);
    assert_rejected("hidden_size not divisible by num_heads", v, "num_heads");
}

#[test]
fn zero_vocab_size_is_allowed_for_vision_models() {
    // `vocab_size: 0` is the "not applicable" sentinel for models without a
    // vocabulary; it must not be caught by the overflow bounds.
    let mut v = base();
    v["model"]["global_params"]["vocab_size"] = json!(0);
    let cfg = neurax_parser::parse_model_config(&v.to_string())
        .expect("vocab_size = 0 should remain valid");
    neurax_core::run_analysis(cfg).expect("vocab_size = 0 should analyze");
}

#[test]
fn shipped_example_models_still_parse() {
    let mut checked = 0;
    for entry in std::fs::read_dir("../examples/models").expect("examples/models should exist") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let json = std::fs::read_to_string(&path).expect("readable model file");
        neurax_parser::parse_model_config(&json)
            .unwrap_or_else(|e| panic!("{} should still parse: {e}", path.display()));
        checked += 1;
    }
    assert!(checked > 0, "expected at least one example model");
}
