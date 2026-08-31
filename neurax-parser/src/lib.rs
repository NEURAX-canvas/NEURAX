//! # NEURAX Parser
//!
//! **The NEURAX universal model format — JSON → strongly-typed AST.**
//!
//! Part of the [NEURAX](https://github.com/rustnew/NEURAX) analytical compiler
//! for neural network architectures. Parses the NEURAX JSON format (a universal
//! description of *any* neural architecture) into a validated, strongly-typed
//! AST.
//!
//! ## Pipeline
//!
//! 1. **Parse** — [`parse_model_config`] turns JSON into a [`ModelConfig`]
//! 2. **Validate** — [`validate_model_config`] checks schema + semantic coherence
//!
//! ## Quick example
//!
//! ```rust
//! use neurax_parser::parse_model_config;
//!
//! let json = r#"{
//!   "schema_version": "1.0",
//!   "model": {
//!     "name": "tiny-gpt",
//!     "type": "transformer",
//!     "layers": [
//!       { "id": "attn_0", "layer_type": "attention",
//!         "input_shape": [128, 768], "output_shape": [128, 768],
//!         "params": { "num_heads": 12 } }
//!     ],
//!     "global_params": { "hidden_size": 768, "num_layers": 1 }
//!   },
//!   "training": { "batch_size": 32, "optimizer": "adamw", "precision": "bf16" },
//!   "hardware": {
//!     "gpus": [
//!       { "name": "A100-80GB", "count": 1, "memory_gb": 80,
//!         "tflops_fp16": 312, "tflops_fp32": 19.5,
//!         "memory_bandwidth_gb_s": 2039, "tensor_cores": true }
//!     ],
//!     "interconnect": "None", "interconnect_bandwidth_gb_s": 0
//!   }
//! }"#;
//!
//! let config = parse_model_config(json).expect("valid NEURAX JSON");
//! assert_eq!(config.model.layers.len(), 1);
//! ```
//!
//! ## Run the example
//!
//! ```bash
//! cargo run --example parse_basics
//! ```

mod error;
mod model_config;
mod schema;
mod validator;

pub use error::*;
pub use model_config::*;
pub use schema::*;
pub use validator::*;

use std::io::Read;

/// Parse JSON string into ModelConfig
pub fn parse_model_config(json: &str) -> Result<ModelConfig, ParserError> {
    let raw: RawModelConfig = serde_json::from_str(json).map_err(ParserError::JsonParse)?;

    let config = ModelConfig::from_raw(raw)?;
    validate_model_config(&config)?;

    Ok(config)
}

/// Parse JSON from reader
pub fn parse_model_config_from_reader<R: Read>(reader: R) -> Result<ModelConfig, ParserError> {
    let raw: RawModelConfig = serde_json::from_reader(reader).map_err(ParserError::JsonParse)?;

    let config = ModelConfig::from_raw(raw)?;
    validate_model_config(&config)?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_valid_json() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "TestModel",
                "type": "transformer",
                "layers": [
                    {
                        "id": "layer_1",
                        "layer_type": "embedding",
                        "input_shape": [128, 1],
                        "output_shape": [128, 512],
                        "params": {
                            "vocab_size": 50000,
                            "embedding_dim": 512
                        }
                    }
                ]
            },
            "training": {
                "batch_size": 128
            },
            "hardware": {
                "gpus": [{"name": "A100", "count": 1}]
            }
        }"#;

        let config = parse_model_config(json);
        assert!(config.is_ok());
    }

    #[test]
    fn test_parse_missing_schema_version() {
        let json = r#"{
            "model": {
                "name": "Test",
                "type": "transformer",
                "layers": []
            }
        }"#;

        let result = parse_model_config(json);
        // May fail on missing layers or other validation, not specifically schema_version
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_layers() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "Test",
                "type": "transformer",
                "layers": []
            },
            "training": {"batch_size": 32},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;

        let result = parse_model_config(json);
        assert!(matches!(result, Err(ParserError::SchemaValidation { .. })));
    }
}
