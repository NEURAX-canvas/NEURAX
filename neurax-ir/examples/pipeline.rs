//! NEURAX-ir examples.
//!
//! Run the 10-pass IR pipeline manually and inspect each stage's IR.
//! Each pass implements `IrPass` with `build` / `compute_metrics` / `validate`.
//! (The convenience `run` helper lives in `neurax-core::IrPassExt`.)

use neurax_ir::architecture::ArchitecturePass;
use neurax_ir::compute::ComputePass;
use neurax_ir::cost::CostPass;
use neurax_ir::graph::GraphPass;
use neurax_ir::hardware::HardwarePass;
use neurax_ir::memory::MemoryPass;
use neurax_ir::operator::OperatorPass;
use neurax_ir::parallelism::ParallelismPass;
use neurax_ir::report::ReportPass;
use neurax_ir::tensor::TensorPass;
use neurax_ir::traits::IrPass;
use neurax_ir::NeuraxContext;
use neurax_parser::parse_model_config;

/// Run a pass and return its output IR.
fn run_pass<P: IrPass>(
    pass: P,
    input: &P::Input,
    ctx: &NeuraxContext,
) -> Result<P::Output, neurax_ir::error::NeuraxError> {
    let mut output = pass.build(input, ctx).map_err(Into::into)?;
    let metrics = pass.compute_metrics(&mut output, ctx).map_err(Into::into)?;
    pass.validate(&output, &metrics).map_err(Into::into)?;
    Ok(output)
}

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
    println!("=== NEURAX 10-pass IR pipeline ===\n");

    let config = parse_model_config(MODEL_JSON)?;
    let ctx = NeuraxContext::new(config.clone());

    // Pass 1: Architecture — ModelConfig -> ArchitectureIR
    let arch = run_pass(ArchitecturePass, &config, &ctx)?;
    println!("[1] Architecture: {} layer(s)", arch.layers.len());

    // Pass 2: Graph
    let graph = run_pass(GraphPass, &arch, &ctx)?;
    println!("[2] Graph: depth={}, ops={}", graph.metrics.graph_depth, graph.metrics.total_operations);

    // Pass 3: Tensor
    let tensor = run_pass(TensorPass, &graph, &ctx)?;
    println!("[3] Tensor: {} tensor(s)", tensor.tensors.len());

    // Pass 4: Operator
    let operator = run_pass(OperatorPass, &(tensor.clone(), arch.clone()), &ctx)?;
    println!("[4] Operator: {} op(s)", operator.operations.len());

    // Pass 5: Compute
    let compute = run_pass(ComputePass, &operator, &ctx)?;
    println!(
        "[5] Compute: {:.2e} FLOPs, intensity={:.1}",
        compute.metrics.total_flops, compute.metrics.arithmetic_intensity
    );

    // Pass 6: Memory
    let memory = run_pass(MemoryPass, &(compute.clone(), tensor.clone(), arch.clone()), &ctx)?;
    println!("[6] Memory: peak={:.2} GB", memory.metrics.peak_vram_bytes as f64 / 1e9);

    // Pass 7: Parallelism
    let parallelism = run_pass(ParallelismPass, &(memory.clone(), graph.clone()), &ctx)?;
    println!("[7] Parallelism: optimal={:?}", parallelism.optimal_strategy);

    // Pass 8: Hardware
    let hardware = run_pass(
        HardwarePass,
        &(compute.clone(), memory.clone(), parallelism.clone()),
        &ctx,
    )?;
    println!("[8] Hardware: gpu_utilization={:.2}", hardware.metrics.gpu_utilization);

    // Pass 9: Cost
    let cost = run_pass(CostPass, &(hardware.clone(), parallelism.clone()), &ctx)?;
    println!("[9] Cost: ${:.2} total", cost.metrics.training_cost_usd);

    Ok(())
}
