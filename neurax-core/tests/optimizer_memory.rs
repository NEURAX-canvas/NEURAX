//! "Adafactor" and "Lion" are real options in the optimizer dropdown —
//! both exist specifically to use less memory than AdamW — but any
//! optimizer other than "sgd" fell into the same `parameter_memory_bytes *
//! 2` bucket, reporting no benefit at all for choosing them.

use neurax_core::analyze_json;

fn model_json(optimizer: &str) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "Optimizer-Memory-Check",
                "type": "transformer",
                "layers": [
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_heads": 32}}}}
                ]
            }},
            "training": {{ "batch_size": 1, "precision": "fp16", "optimizer": "{optimizer}" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp16" }}
        }}
        "#
    )
}

#[test]
fn lion_costs_the_same_as_sgd_not_adamw() {
    let sgd = analyze_json(&model_json("sgd")).unwrap().memory.metrics.optimizer_state_bytes;
    let lion = analyze_json(&model_json("lion")).unwrap().memory.metrics.optimizer_state_bytes;
    let adamw = analyze_json(&model_json("adamw")).unwrap().memory.metrics.optimizer_state_bytes;

    assert_eq!(lion, sgd, "Lion has one momentum buffer, same as SGD");
    assert!(lion < adamw, "Lion must report less memory than AdamW's two states");
}

#[test]
fn adafactor_costs_less_than_adamw() {
    let adamw = analyze_json(&model_json("adamw")).unwrap().memory.metrics.optimizer_state_bytes;
    let adafactor = analyze_json(&model_json("adafactor"))
        .unwrap()
        .memory
        .metrics
        .optimizer_state_bytes;

    assert!(
        adafactor < adamw,
        "Adafactor's factored second moment should report less memory than AdamW's full states"
    );
}
