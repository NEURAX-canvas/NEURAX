//! int4 quantization was reported at the same memory footprint as int8 —
//! `dtype_bytes("int4")` returned 1 (a whole byte) instead of 0.5 (two
//! values packed per byte, as GPTQ/AWQ/QLoRA-NF4/GGUF Q4 — the most-cited
//! LLM quantization scheme in practice — actually store it). A model
//! quantized to int4 specifically to fit smaller hardware was reported as
//! needing twice the memory it actually does.

use neurax_core::analyze_json;

fn model_json(precision: &str) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "Int4-Check",
                "type": "transformer",
                "layers": [
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_heads": 32}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 4096, "intermediate_size": 11008}}}}
                ],
                "global_params": {{ "hidden_size": 4096 }}
            }},
            "training": {{ "batch_size": 1, "precision": "{precision}", "optimizer": "none" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "{precision}" }}
        }}
        "#
    )
}

#[test]
fn int4_is_half_the_memory_of_int8() {
    let int8 = analyze_json(&model_json("int8")).expect("analysis should succeed");
    let int4 = analyze_json(&model_json("int4")).expect("analysis should succeed");

    let int8_bytes = int8.memory.metrics.parameter_memory_bytes;
    let int4_bytes = int4.memory.metrics.parameter_memory_bytes;

    assert!(int8_bytes > 0);
    assert_eq!(
        int4_bytes,
        int8_bytes / 2,
        "int4 should store exactly half the bytes int8 does for the same parameter count"
    );
}
