# neurax-parser

**The NEURAX universal model format — JSON → strongly-typed AST.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural network architectures. This crate parses the NEURAX JSON format (a universal description of any neural architecture) into a strongly-typed, validated, dimension-resolved AST.

> **The vision:** one JSON format to describe *any* neural architecture — Transformer, CNN, MoE, SSM, Diffusion, GNN, GAN, RL, SNN, RNN — so that analysis, cost prediction and code generation all speak the same language.

## What it does

1. **Parse** — NEURAX JSON → strongly-typed `ModelConfig` (`parse_model_config`)
2. **Validate** — schema + semantic validation (coherence checks, dimension consistency) (`ModelValidator`)
3. **Absorb** — resolve symbolic dimensions into concrete values using layer context and global propagation (`AbsorbedModel::absorb`)

## Installation

```toml
[dependencies]
neurax-parser = { git = "https://github.com/rustnew/NEURAX", package = "neurax-parser" }
```

## Quick start

```rust
use neurax_parser::{parse_model_config, AbsorbedModel};

// Minimal NEURAX model JSON
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

// 1. Parse → strongly-typed config
let config = parse_model_config(MODEL_JSON)?;
assert_eq!(config.model.layers.len(), 1);

// 2. Validate (schema + coherence)
use neurax_parser::ModelValidator;
let validation = ModelValidator::new().validate(MODEL_JSON);
assert!(validation.is_valid);
println!("layer_count: {}", validation.metrics.layer_count);

// 3. Absorb → resolve symbolic dimensions
let absorbed = AbsorbedModel::absorb(config);
let grc = &absorbed.resolution_context;
println!("hidden_size: {:?}", grc.hidden_size);
println!("dtype bytes: {}", grc.dtype_bytes);
```

## The NEURAX format

```json
{
  "schema_version": "1.0",
  "model":       { "name", "type", "layers[]", "global_params" },
  "training":    { "batch_size", "optimizer", "learning_rate", "precision",
                   "gradient_checkpointing", "zero_stage", "parallelism" },
  "hardware":    { "gpus[]", "interconnect", "interconnect_bandwidth_gb_s" },
  "data":        { "input_shape", "dtype" },
  "cost_config": { "provider", "gpu_hour_usd", "energy_kwh_usd", "pue_factor" }
}
```

- **11 architecture families** supported: transformer, moe, cnn, ssm, diffusion, gnn, gan, rl, snn, rnn, experimental
- **208 configurable blocks** via `layer_type` + `params`
- **88 reference templates** in the [NEURAX repo](https://github.com/rustnew/NEURAX/tree/main/examples/models) (GPT-4, LLaMA-2, Mixtral, DeepSeek-V3, SDXL...)

## API overview

| Item | Description |
|------|-------------|
| `parse_model_config(json) -> Result<ModelConfig, ParserError>` | Parse JSON into `ModelConfig` |
| `ModelConfig::validate()` | Schema + semantic validation |
| `AbsorbedModel::absorb(config)` | Dimension resolution & propagation |
| `GlobalResolutionContext` | Resolved symbols: `hidden_size`, `intermediate_size`, `dtype_bytes`, `confidence_score`, `missing_fields`, `d_inner()`, ... |
| `ModelValidator` / `CoherenceValidator` | Standalone validation entry points |

## Using it in the pipeline

`neurax-parser` is the front-end of the [NEURAX ecosystem](https://github.com/rustnew/NEURAX). For end-to-end analysis use [`neurax-core`](../neurax-core):

```rust
use neurax_core::analyze_json;

let result = analyze_json(MODEL_JSON)?; // parse → validate → 10-pass IR → report
```

## License

Proprietary — closed-source, commercial software. All rights reserved.