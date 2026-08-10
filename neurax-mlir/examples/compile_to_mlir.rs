//! NEURAX-mlir examples.
//!
//! Compile a parsed model to textual MLIR using the NEURAX dialects.

use neurax_mlir::{compile_model_to_mlir, NeuraxContext};
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
    println!("=== NEURAX MLIR code generation ===\n");

    let config = parse_model_config(MODEL_JSON)?;
    let ctx = NeuraxContext::new();

    let mlir = compile_model_to_mlir(ctx.as_context(), &config)
        .map_err(|e| format!("MLIR compilation failed: {e}"))?;

    println!("Generated {} bytes of MLIR:\n", mlir.len());
    for line in mlir.lines().take(30) {
        println!("{}", line);
    }

    Ok(())
}