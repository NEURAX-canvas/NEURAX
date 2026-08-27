//! High-level NEURAX → MLIR compilation entry point.
//!
//! Exposes [`compile_model_to_mlir`], which lowers a parsed
//! [`neurax_parser::ModelConfig`] into textual MLIR using the NEURAX
//! dialects.
//!
//! This is a library API with no command behind it. `neurax-cli` used to call
//! it through a `compile` subcommand; that crate has been removed, so the MLIR
//! backend is reached by depending on this crate — see `examples/` — rather
//! than by running a binary. Nothing else in the workspace pulls it in, which
//! is worth knowing before changing it: its 119 tests are the only thing
//! exercising it.

use melior::ir::Location;

use crate::dialects::{
    ArchitectureDialect, ComputeDialect, CostDialect, HardwareDialect, MemoryDialect,
    OperatorDialect, ParallelismDialect, ReportDialect, TensorDialect,
};
use neurax_ir::architecture::LayerDef;
use neurax_ir::NeuraxContext;
use neurax_parser::{LayerType, ModelConfig};

/// `batch`, `seq` and `dtype` as the analytical pipeline (`neurax-ir`)
/// derives them, so a layer costed here and the same layer costed by the
/// application's actual metrics pipeline start from the same inputs.
fn training_shape(ctx: &NeuraxContext) -> (usize, usize, String) {
    let batch = ctx.config.training.batch_size;
    let seq = ctx
        .config
        .training
        .sequence_length
        .or(ctx.config.model.global_params.sequence_length)
        .unwrap_or(512);
    (batch, seq, ctx.config.training.precision.clone())
}

/// Compile a parsed model configuration to textual MLIR.
///
/// Produces one MLIR operation per architectural element (model, layers,
/// operators, hardware, parallelism, compute, memory, cost, report) using
/// the NEURAX MLIR dialects, then concatenates their textual forms.
pub fn compile_model_to_mlir(
    context: &melior::Context,
    config: &ModelConfig,
) -> Result<String, String> {
    let mut operations = Vec::new();
    let location = Location::unknown(context);

    // 1. Architecture model operation
    let model_name = config.model.name.as_deref().unwrap_or("unnamed_model");
    let model_type = config.model.model_type.as_str();

    let model_op = ArchitectureDialect::model(context, model_name, model_type, location)
        .map_err(|e| format!("Failed to create model op: {e:?}"))?;
    operations.push(model_op.to_string());

    // 2. Global params
    let global_params = &config.model.global_params;
    let num_layers = global_params
        .num_layers
        .unwrap_or(config.model.layers.len() as u64) as i64;

    let mut params_vec: Vec<(&str, i64)> = Vec::new();
    if let Some(seq_len) = global_params.sequence_length {
        params_vec.push(("sequence_length", seq_len as i64));
    }
    if let Some(vocab) = global_params.vocab_size {
        params_vec.push(("vocab_size", vocab as i64));
    }
    if let Some(embed_dim) = global_params.embedding_dim {
        params_vec.push(("embedding_dim", embed_dim as i64));
    }

    if !params_vec.is_empty() {
        let global_op = ArchitectureDialect::global_params(context, &params_vec, location)
            .map_err(|e| format!("Failed to create global_params op: {e:?}"))?;
        operations.push(global_op.to_string());
    }

    // 3. Layer operations
    //
    // Every layer's `param_count` and `flops` below come from the same
    // formulas the application's actual metrics pipeline uses
    // (`neurax_ir::calculate_layer_params`, `neurax_ir::layer_flops`) instead
    // of a separate approximation kept only here. Previously this match
    // covered five layer types (Attention, Mlp, MoE, Embedding, Dense) with
    // its own cruder formulas — e.g. attention's param count was `hidden²`,
    // missing the ×4 for the Q/K/V/O projections a real attention block has
    // — and silently emitted nothing at all (`_ => {}`) for every CNN, SSM,
    // GAN, RNN, diffusion, GNN and SNN layer type. Every layer type now gets
    // a real `param_count`/`flops` pair; the specific `op.*` dialect kind
    // below still only distinguishes the families that already had one
    // (matching a richer dialect op per family — `op.conv2d`, `op.ssm`,
    // `op.generator`, ... — to every remaining `LayerType` is further,
    // separate work; `op.custom` carries the real numbers honestly under
    // the layer's own type name until then).
    let mlir_ctx = NeuraxContext::new(config.clone());
    let (batch, seq, dtype) = training_shape(&mlir_ctx);

    for layer in &config.model.layers {
        let layer_op =
            ArchitectureDialect::layer(context, &layer.id, layer.layer_type.as_str(), location)
                .map_err(|e| format!("Failed to create layer op for {}: {e:?}", layer.id))?;
        operations.push(layer_op.to_string());

        let param_count = neurax_ir::calculate_layer_params(layer) as i64;
        let layer_def = LayerDef::from(layer);
        let flops = neurax_ir::layer_flops(&layer_def, batch, seq, &dtype, &mlir_ctx);

        match layer.layer_type {
            LayerType::Attention => {
                // Same defaults `neurax_ir::calculate_layer_params` uses for
                // this layer type — otherwise a layer that omits
                // `hidden_size`/`num_heads` would show one pair of numbers
                // here while `param_count` above was computed from another.
                let hidden_size = layer.params.hidden_size.unwrap_or(512) as i64;
                let num_heads = layer.params.num_heads.unwrap_or(8) as i64;

                let attn_op = OperatorDialect::attention(
                    context,
                    hidden_size,
                    num_heads,
                    param_count,
                    flops,
                    location,
                )
                .map_err(|e| format!("Failed to create attention op: {e:?}"))?;
                operations.push(attn_op.to_string());
            }
            LayerType::Mlp | LayerType::Dense => {
                let mlp_op = OperatorDialect::matmul(context, param_count, flops, location)
                    .map_err(|e| format!("Failed to create matmul op: {e:?}"))?;
                operations.push(mlp_op.to_string());
            }
            LayerType::MoE => {
                // Same defaults `neurax_ir`'s MoE param/FLOPs formulas use.
                let hidden_size = layer.params.hidden_size.unwrap_or(512) as i64;
                let num_experts = layer.params.num_experts.unwrap_or(8) as i64;
                let top_k = layer.params.top_k.unwrap_or(2) as i64;

                let moe_op = OperatorDialect::moe(
                    context,
                    hidden_size,
                    num_experts,
                    top_k,
                    param_count,
                    flops,
                    location,
                )
                .map_err(|e| format!("Failed to create moe op: {e:?}"))?;
                operations.push(moe_op.to_string());
            }
            LayerType::Embedding => {
                // Same fallback chain `neurax_ir`'s Embedding formula uses:
                // an explicit `embedding_dim`, then the layer's own
                // `hidden_size`, then 512 — not a flat 768 default that
                // ignores whether `hidden_size` was actually set.
                let vocab_size = layer.params.vocab_size.unwrap_or(50000) as i64;
                let embed_dim = layer
                    .params
                    .embedding_dim
                    .unwrap_or(layer.params.hidden_size.unwrap_or(512))
                    as i64;

                let tensor_shape = vec![vocab_size, embed_dim];
                let tensor_op = TensorDialect::tensor_info(
                    context,
                    &format!("{}_weights", layer.id),
                    &tensor_shape,
                    &dtype,
                    (param_count as f64 * neurax_formulas::dtype_bytes(&dtype)).round() as i64,
                    &layer.id,
                    location,
                )
                .map_err(|e| format!("Failed to create tensor op: {e:?}"))?;
                operations.push(tensor_op.to_string());
            }
            _ => {
                let custom_op = OperatorDialect::custom(
                    context,
                    layer.layer_type.as_str(),
                    param_count,
                    flops,
                    None,
                    None,
                    location,
                )
                .map_err(|e| format!("Failed to create op for {}: {e:?}", layer.id))?;
                operations.push(custom_op.to_string());
            }
        }
    }

    // 4. Architecture metrics
    let total_params = calculate_total_params(config);
    let metrics_op = ArchitectureDialect::metrics(context, total_params, num_layers, location)
        .map_err(|e| format!("Failed to create metrics op: {e:?}"))?;
    operations.push(metrics_op.to_string());

    // 5. Hardware configuration
    if let Some(gpu) = config.hardware.primary_gpu() {
        let hw_op = HardwareDialect::gpu(
            context,
            &gpu.name,
            gpu.memory_gb.unwrap_or(40) as i64,
            gpu.tflops_fp16.unwrap_or(100.0),
            gpu.memory_bandwidth_gbs.unwrap_or(1000.0),
            location,
        )
        .map_err(|e| format!("Failed to create hw op: {e:?}"))?;
        operations.push(hw_op.to_string());
    }

    // 6. Parallelism configuration
    let par = &config.training.parallelism;
    if par.data_parallel > 0 || par.tensor_parallel > 0 || par.pipeline_parallel > 0 {
        let par_op = ParallelismDialect::hybrid(
            context,
            par.data_parallel as i64,
            par.tensor_parallel as i64,
            par.pipeline_parallel as i64,
            location,
        )
        .map_err(|e| format!("Failed to create parallelism op: {e:?}"))?;
        operations.push(par_op.to_string());
    }

    // 7. Compute metrics
    let total_flops = calculate_total_flops(config);
    let compute_op = ComputeDialect::flops(context, total_flops, total_flops * 2.0, location)
        .map_err(|e| format!("Failed to create compute op: {e:?}"))?;
    operations.push(compute_op.to_string());

    // 8. Memory metrics (approximate)
    let param_memory = total_params * 4; // FP32
    let activation_memory = calculate_activation_memory(config);
    let gradient_memory = param_memory;
    let optimizer_memory = param_memory * 2; // Adam states
    let peak_vram = param_memory + activation_memory + gradient_memory + optimizer_memory;

    let mem_op = MemoryDialect::metrics(
        context,
        param_memory,
        activation_memory,
        gradient_memory,
        optimizer_memory,
        peak_vram,
        1,
        location,
    )
    .map_err(|e| format!("Failed to create memory op: {e:?}"))?;
    operations.push(mem_op.to_string());

    // 9. Cost analysis
    let training_hours = estimate_training_hours(config);
    let gpu_hours = training_hours * config.hardware.total_gpu_count() as f64;
    let training_cost = gpu_hours * config.cost_config.gpu_hour_usd;

    let cost_op =
        CostDialect::training_cost(context, training_hours, training_cost, gpu_hours, location)
            .map_err(|e| format!("Failed to create cost op: {e:?}"))?;
    operations.push(cost_op.to_string());

    // 10. Final report
    let report_op = ReportDialect::report(context, model_name, model_type, "1.0", 0, location)
        .map_err(|e| format!("Failed to create report op: {e:?}"))?;
    operations.push(report_op.to_string());

    // Format output
    let mut output = String::new();
    output.push_str(&format!("// MLIR for model: {model_name}\n"));
    output.push_str(&format!("// Type: {model_type}\n"));
    output.push_str(&format!("// Layers: {num_layers}\n"));
    output.push_str(&format!("// Parameters: {total_params}\n\n"));

    for op in operations {
        output.push_str(&op);
        output.push('\n');
    }

    Ok(output)
}

/// Total parameters across every layer, real formula per layer type, scaled
/// by [`neurax_ir::repeat_scale_for`] exactly as the application's own
/// metrics pipeline scales a JSON that lists one representative block per
/// kind rather than every real layer.
///
/// This used to assume every model was a uniform transformer stack — one
/// hidden size, one intermediate size, one attention shape, read out of
/// `global_params.extra` and repeated `num_layers` times — which was wrong
/// for the same reason the MLIR backend's per-layer loop was: a CNN's,
/// SSM's or diffusion model's layers aren't attention/MLP blocks at all, and
/// even for an actual transformer, a config with several genuinely different
/// layers (GQA in some blocks, MHA in others; a wider or narrower MLP on
/// specific layers) was flattened into "the same block, `num_layers` times".
fn calculate_total_params(config: &ModelConfig) -> i64 {
    neurax_ir::scaled_total_parameters(config) as i64
}

/// Total FLOPs across every layer, real per-layer-type formula
/// ([`neurax_ir::layer_flops`]) scaled the same way as parameters above.
fn calculate_total_flops(config: &ModelConfig) -> f64 {
    let ctx = NeuraxContext::new(config.clone());
    let (batch, seq, dtype) = training_shape(&ctx);

    config
        .model
        .layers
        .iter()
        .map(|layer| {
            let layer_def = LayerDef::from(layer);
            let scale = neurax_ir::repeat_scale_for(config, layer);
            neurax_ir::layer_flops(&layer_def, batch, seq, &dtype, &ctx) * scale
        })
        .sum()
}

fn calculate_activation_memory(config: &ModelConfig) -> i64 {
    let hidden = config.model.global_params.embedding_dim.unwrap_or(768) as i64;
    let seq = config.model.global_params.sequence_length.unwrap_or(2048) as i64;
    let batch = config.training.batch_size as i64;
    let num_layers = config
        .model
        .global_params
        .num_layers
        .unwrap_or(config.model.layers.len() as u64) as i64;

    batch * seq * hidden * num_layers * 2 * 2
}

fn estimate_training_hours(config: &ModelConfig) -> f64 {
    let total_params = calculate_total_params(config) as f64;
    let steps = config.training.max_steps as f64;
    let batch = config.training.batch_size as f64;

    let tokens_per_step = batch * config.model.global_params.sequence_length.unwrap_or(2048) as f64;
    let total_training_flops = 6.0 * total_params * tokens_per_step * steps;

    let gpu_tflops = config
        .hardware
        .primary_gpu()
        .and_then(|g| g.tflops_fp16)
        .unwrap_or(100.0)
        * 1e12;

    let num_gpus = config.hardware.total_gpu_count() as f64;
    let efficiency = 0.3;

    if gpu_tflops <= 0.0 || num_gpus <= 0.0 {
        return 0.0;
    }

    let time_seconds = total_training_flops / (gpu_tflops * num_gpus * efficiency);
    time_seconds / 3600.0
}
