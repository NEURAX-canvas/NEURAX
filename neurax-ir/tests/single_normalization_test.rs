//! Test compilation of a single normalization block
//! Shows detailed absorption results for normalization layers (LayerNorm, RMSNorm)

/// Single normalization block JSON (RMSNorm style)
const SINGLE_NORM_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "SingleNorm-Block",
        "type": "transformer",
        "layers": [
            {
                "id": "rms_norm_0",
                "layer_type": "normalization",
                "input_shape": [2048, 4096],
                "output_shape": [2048, 4096],
                "params": {
                    "hidden_size": 4096,
                    "norm_type": "rms",
                    "epsilon": 1e-6
                }
            }
        ],
        "global_params": {
            "hidden_size": 4096,
            "num_layers": 1,
            "vocab_size": 32000
        }
    },
    "training": {
        "batch_size": 32,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "precision": "bf16",
        "gradient_checkpointing": false,
        "zero_stage": 0,
        "max_steps": 1000,
        "warmup_steps": 100,
        "parallelism": {
            "data_parallel": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "A100-80GB",
                "count": 1,
                "memory_gb": 80,
                "tflops_fp16": 312,
                "tflops_fp32": 19.5,
                "tflops_fp8": 624,
                "memory_bandwidth_gb_s": 2039,
                "tensor_cores": true,
                "nvlink": false
            }
        ],
        "interconnect": "None",
        "interconnect_bandwidth_gb_s": 0
    },
    "data": {
        "input_shape": [2048, 4096],
        "dtype": "bf16",
        "vocab_size": 32000
    },
    "cost_config": {
        "provider": "local",
        "gpu_hour_usd": 0.0,
        "energy_kwh_usd": 0.0,
        "pue_factor": 1.0
    }
}
"#;

#[test]
fn test_norm_types_comparison() {
    println!("\n=== Normalization Types Comparison ===\n");

    let h = 4096f64;

    println!("┌───────────────────┬───────────────┬───────────────────────┐");
    println!("│ Normalization     │ Parameters    │ Formula               │");
    println!("├───────────────────┼───────────────┼───────────────────────┤");
    println!(
        "│ RMSNorm           │ {:>10.0}    │ γ * x / √(mean(x²)+ε) │",
        h
    );
    println!(
        "│ LayerNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │",
        h * 2.0
    );
    println!(
        "│ BatchNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │",
        h * 2.0
    );
    println!(
        "│ GroupNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │",
        h * 2.0
    );
    println!("└───────────────────┴───────────────┴───────────────────────┘\n");

    println!("RMSNorm (used in LLaMA):");
    println!("  - No bias term (β)");
    println!("  - No mean subtraction");
    println!("  - Simpler computation: x * γ / √(mean(x²) + ε)");
    println!("  - Parameters: {} (weight only)\n", h as u64);

    println!("LayerNorm (used in BERT/GPT):");
    println!("  - Has both weight (γ) and bias (β)");
    println!("  - Full normalization: (x - μ) / σ");
    println!("  - Parameters: {} (weight + bias)\n", (h * 2.0) as u64);
}
