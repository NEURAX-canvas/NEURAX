//! NEURAX-parser examples.
//!
//! Parse and validate a NEURAX model JSON.

use neurax_parser::parse_model_config;

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

    // parse_model_config() both parses and validates (validate_model_config()
    // runs internally) — there is no separate validate-only entry point in
    // this crate; use neurax_core::validate_json() for that.
    let config = parse_model_config(MODEL_JSON)?;
    println!(
        "Parsed: {} ({:?}), {} layer(s)",
        config.model.name.as_deref().unwrap_or("unnamed"),
        config.model.model_type,
        config.model.layers.len()
    );

    Ok(())
}
