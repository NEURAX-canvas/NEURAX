//! A config may carry hyperparameters beyond the built-in set.
//!
//! Architectures differ enough that no fixed list covers them, so the schema
//! keeps a flattened catch-all: unknown keys must survive parsing and analysis
//! rather than being rejected or silently dropped.

use serde_json::json;

fn config_with_extra_globals(extra: serde_json::Value) -> String {
    let mut global = json!({
        "num_layers": 2, "hidden_size": 768, "num_heads": 12,
        "intermediate_size": 3072, "vocab_size": 50000, "sequence_length": 1024
    });
    for (k, v) in extra.as_object().unwrap() {
        global[k] = v.clone();
    }
    json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "custom", "type": "transformer",
            "global_params": global,
            "layers": [
                {"id": "embed", "layer_type": "embedding",
                 "params": {"vocab_size": 50000, "hidden_size": 768}},
                {"id": "l0", "layer_type": "attention",
                 "params": {"hidden_size": 768, "num_heads": 12}}
            ]
        },
        "training": {"batch_size": 8, "sequence_length": 1024, "precision": "bf16",
                     "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "count": 8}]},
        "data": {"dataset_size": 1000000, "vocab_size": 50000}
    })
    .to_string()
}

#[test]
fn arbitrary_extra_hyperparameters_are_accepted_and_kept() {
    let json = config_with_extra_globals(json!({
        "weight_decay": 0.1,
        "lr_scheduler": "cosine",
        "layer_norm_eps": 1e-5,
        "my_own_knob": 42,
        "another_one": "some-value",
        "a_flag": true
    }));

    let config = neurax_parser::parse_model_config(&json).expect("extra keys should be accepted");

    // They must still be readable afterwards, not dropped on the floor.
    let extra = &config.model.global_params.extra;
    assert_eq!(extra.get("my_own_knob").and_then(|v| v.as_u64()), Some(42));
    assert_eq!(
        extra.get("another_one").and_then(|v| v.as_str()),
        Some("some-value")
    );
    assert_eq!(extra.get("a_flag").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        extra.get("lr_scheduler").and_then(|v| v.as_str()),
        Some("cosine")
    );

    neurax_core::run_analysis(config).expect("extra keys should not break analysis");
}

#[test]
fn many_extra_hyperparameters_are_supported() {
    // There is no cap on how many a user may add.
    let mut extra = serde_json::Map::new();
    for i in 0..200 {
        extra.insert(format!("knob_{i}"), json!(i));
    }
    let json = config_with_extra_globals(serde_json::Value::Object(extra));
    let config = neurax_parser::parse_model_config(&json).expect("200 extra keys should parse");
    // The catch-all also holds built-ins that are not declared fields of
    // `RawGlobalParams` (hidden_size, num_heads, ...), so check the user's own
    // keys rather than the map's total size.
    let extra = &config.model.global_params.extra;
    for i in 0..200 {
        assert_eq!(
            extra.get(&format!("knob_{i}")).and_then(|v| v.as_u64()),
            Some(i),
            "knob_{i} should survive parsing"
        );
    }
    neurax_core::run_analysis(config).expect("200 extra keys should analyze");
}

#[test]
fn extra_layer_parameters_are_also_kept() {
    let json = json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "custom-layer", "type": "transformer",
            "global_params": {"num_layers": 1, "hidden_size": 768, "sequence_length": 512},
            "layers": [{
                "id": "l0", "layer_type": "attention",
                "params": {"hidden_size": 768, "num_heads": 12, "my_layer_knob": 7}
            }]
        },
        "training": {"batch_size": 4, "sequence_length": 512, "precision": "bf16",
                     "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "A100", "count": 1}]},
        "data": {"dataset_size": 100000}
    })
    .to_string();

    let config = neurax_parser::parse_model_config(&json).expect("extra layer key should parse");
    let layer = &config.model.layers[0];
    assert_eq!(layer.params.extra.get("my_layer_knob").and_then(|v| v.as_u64()), Some(7));
    neurax_core::run_analysis(config).expect("extra layer key should analyze");
}
