//! NEURAX-core examples.
//!
//! End-to-end: JSON -> full analytical report in one call.

use neurax_core::{analyze_json, get_model_summary, validate_json};
use neurax_ir::report::format_markdown;

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
    println!("=== NEURAX unified analysis ===\n");

    // Validate only (fast path)
    let config = validate_json(MODEL_JSON)?;
    let summary = get_model_summary(&config);
    println!(
        "Model: {} | type: {} | layers: {} | batch: {} | precision: {} | gpus: {}",
        summary.name, summary.model_type, summary.num_layers, summary.batch_size,
        summary.precision, summary.gpu_count
    );

    // Full analysis
    let result = analyze_json(MODEL_JSON)?;
    println!("Analysis time: {} ms\n", result.analysis_time_ms);

    println!("-- Compute --");
    println!("  total FLOPs: {:.2e}", result.compute.metrics.total_flops);
    println!("  MACs:        {:.2e}", result.compute.metrics.macs);
    println!("  FLOPs/token: {:.2e}", result.compute.metrics.flops_per_token);

    println!("-- Memory --");
    println!(
        "  peak VRAM: {:.2} GB",
        result.memory.metrics.peak_vram_bytes as f64 / 1e9
    );

    println!("-- Hardware --");
    println!(
        "  GPU utilization: {:.2} | throughput: {:.0} tok/s",
        result.hardware.metrics.gpu_utilization, result.hardware.metrics.throughput_tokens_per_s
    );

    println!("-- Cost --");
    println!(
        "  training: {:.2} h | ${:.2} | {:.1} kWh | {:.2} kg CO2",
        result.cost.metrics.training_time_hours,
        result.cost.metrics.training_cost_usd,
        result.cost.metrics.energy_kwh,
        result.cost.metrics.co2_kg
    );

    // Full markdown report
    println!("\n-- Markdown report (first 40 lines) --");
    let md = format_markdown(&result.report);
    for line in md.lines().take(40) {
        println!("{}", line);
    }

    Ok(())
}