//! A config that lists every layer must be taken at face value.
//!
//! The repeat-scaling heuristic exists for configs that describe a deep model by
//! listing one representative block and setting `num_layers` to the real depth.
//! It used to default that depth to the number of listed layers, so a fully
//! written-out architecture was mistaken for a partial one: a five-layer design
//! with a single attention block was analysed as five attention blocks.

use serde_json::json;

fn tiny_multimodal(num_layers: Option<u64>) -> String {
    let mut global = json!({"hidden_size": 64, "sequence_length": 64, "vocab_size": 2000});
    if let Some(n) = num_layers {
        global["num_layers"] = json!(n);
    }
    json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "TinyPhoneVLM", "type": "transformer",
            "global_params": global,
            "layers": [
                {"id": "img_patch", "layer_type": "conv",
                 "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 8, "stride": 8}},
                {"id": "txt_embed", "layer_type": "embedding",
                 "params": {"vocab_size": 2000, "hidden_size": 64}},
                {"id": "fusion_attn", "layer_type": "attention",
                 "params": {"hidden_size": 64, "num_heads": 4}},
                {"id": "ffn", "layer_type": "mlp",
                 "params": {"hidden_size": 64, "intermediate_size": 128, "activation": "relu"}},
                {"id": "head", "layer_type": "dense",
                 "params": {"in_features": 64, "out_features": 10}}
            ]
        },
        "training": {"batch_size": 1, "sequence_length": 64, "precision": "int8"},
        "hardware": {"gpus": [{"name": "T4", "count": 1}]},
        "data": {"input_shape": [1, 64]}
    })
    .to_string()
}

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("config should parse");
    neurax_core::run_analysis(config).expect("config should analyze")
}

#[test]
fn a_fully_listed_architecture_is_not_scaled_up() {
    let r = analyze(&tiny_multimodal(None));
    // Conv 6,176 + embedding 128,000 + attention 16,384 + MLP 16,576 + head 650.
    let params = r.arch.metrics.total_parameters;
    assert!(
        (160_000..180_000).contains(&params),
        "expected roughly 168k parameters for the layers as listed, got {params}"
    );
}

#[test]
fn model_size_follows_the_declared_precision() {
    let r = analyze(&tiny_multimodal(None));
    let params = r.arch.metrics.total_parameters;
    let bytes = r.memory.metrics.parameter_memory_bytes;
    // int8 stores one byte per parameter.
    assert_eq!(
        bytes, params,
        "int8 storage should be exactly one byte per parameter"
    );
    assert!(
        bytes < 1024 * 1024,
        "this design should fit an on-device 1 MB budget, got {bytes} bytes"
    );
}

#[test]
fn an_explicit_depth_still_scales_the_repeated_blocks() {
    // The heuristic must keep working when the author does declare a depth.
    let listed = analyze(&tiny_multimodal(None)).arch.metrics.total_parameters;
    let declared = analyze(&tiny_multimodal(Some(12))).arch.metrics.total_parameters;
    assert!(
        declared > listed,
        "declaring 12 layers should scale the repeated blocks up ({declared} vs {listed})"
    );
}
