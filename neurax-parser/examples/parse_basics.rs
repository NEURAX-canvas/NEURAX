//! NEURAX-parser examples.
//!
//! Parse, validate and absorb a NEURAX model JSON.

use neurax_parser::{parse_model_config, AbsorbedModel, ModelValidator};

const MODEL_JSON: &str = r#"
{
  "schema_version": "1.0",
  "model": {
    "name": "tiny-gpt",
    "type": "transformer",
    "layers": [
      {
        "id": "attn_0",
        "layer_type": "attention",
        "input_shape": [128, 768],
        "output_shape": [128, 768],
        "params": { "num_heads": 12 }
      }
    ],
    "global_params": { "hidden_size": 768, "num_layers": 1 }
  },
  "training": { "batch_size": 32, "optimizer": "adamw", "precision": "bf16" },
  "hardware": {
    "gpus": [
      {
        "name": "A100-80GB",
        "count": 1,
        "memory_gb": 80,
        "tflops_fp16": 312,
        "tflops_fp32": 19.5,
        "memory_bandwidth_gb_s": 2039,
        "tensor_cores": true
      }
    ],
    "interconnect": "None",
    "interconnect_bandwidth_gb_s": 0
  }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NEURAX parser ===\n");

    // 1. Parse
    let config = parse_model_config(MODEL_JSON)?;
    println!(
        "Parsed: {} ({:?}), {} layer(s)",
        config.model.name.as_deref().unwrap_or("unnamed"),
        config.model.model_type,
        config.model.layers.len()
    );

    // 2. Validate
    let validation = ModelValidator::new().validate(MODEL_JSON);
    println!(
        "Valid: {} | layer_count: {} | gpu_count: {}",
        validation.is_valid, validation.metrics.layer_count, validation.metrics.gpu_count
    );

    // 3. Absorb → resolve symbolic dims
    let absorbed = AbsorbedModel::absorb(config);
    let grc = &absorbed.resolution_context;

    println!("\n-- Resolution context --");
    println!("hidden_size: {:?}", grc.hidden_size);
    println!("dtype_bytes: {}", grc.dtype_bytes);
    println!("optimizer_bytes_per_param: {}", grc.optimizer_bytes_per_param);
    println!("confidence: {:.0}%", grc.confidence_score * 100.0);
    if !grc.missing_fields.is_empty() {
        println!("missing fields: {:?}", grc.missing_fields);
    }

    Ok(())
}
