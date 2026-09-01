//! Compute IR pass

use super::{ComplexityClass, ComputeIR, ComputeMetrics, OpFlops};
use crate::error::ComputeError;
use crate::operator::OperatorIR;
use crate::traits::IrPass;
use crate::NeuraxContext;

/// How many full U-Net forward passes one sample actually costs — 1.0 for
/// every non-diffusion family, and for a diffusion model without a stated
/// `diffusion_timesteps` (nothing to scale by, so it stays a single pass
/// rather than guessing a step count).
pub(crate) fn diffusion_sample_factor(ctx: &NeuraxContext) -> f64 {
    if ctx.config.model.model_type != neurax_parser::ModelType::Diffusion {
        return 1.0;
    }
    let Some(steps) = ctx.config.model.global_params.diffusion_timesteps else {
        return 1.0;
    };
    let guidance = if ctx.config.model.global_params.guidance_scale.is_some() {
        2.0
    } else {
        1.0
    };
    steps as f64 * guidance
}

/// Compute pass implementation
pub struct ComputePass;

impl IrPass for ComputePass {
    type Input = OperatorIR;
    type Output = ComputeIR;
    type Metrics = ComputeMetrics;
    type PassError = ComputeError;

    fn name(&self) -> &'static str {
        "ComputeIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let mut compute_ir = ComputeIR::default();

        // A diffusion model's layers describe one U-Net forward pass, but
        // generating a sample means running that pass repeatedly — the
        // JSON's `layers` list was never multiplied by `diffusion_timesteps`
        // anywhere in the pipeline, so every diffusion model's reported
        // FLOPs/latency/cost were the price of one denoising step, not the
        // 20-1000 a real sample takes. Classifier-free guidance (Ho &
        // Salimans, 2022 — the default in Stable Diffusion, SDXL, DALL-E)
        // doubles that again: the U-Net runs once conditioned, once not,
        // every step. Scaling every op's FLOPs here — rather than only the
        // aggregate metric below — matters because `HardwarePass` derives
        // `latency_ms` (and everything the cost pass computes from it) by
        // re-reading these same per-op numbers, not the aggregate.
        let sample_factor = diffusion_sample_factor(ctx);

        let dtype_bytes = neurax_formulas::dtype_bytes(&ctx.config.training.precision);

        // Convert operator FLOPs to compute FLOPs
        for op in &input.operations {
            let flops = op.flops * sample_factor;
            // This op's own weight bytes (in the run's real dtype) plus its
            // own activation buffer — both already computed per-op by the
            // operator pass (`AtomOp::param_count`/`activation_memory`) and
            // previously dropped here, leaving nothing but a flat ratio for
            // HardwarePass to estimate memory time from.
            // Scaled by the same `sample_factor` as `flops`: a diffusion
            // sample re-runs this op (and re-reads its weights, re-writes
            // its activations) once per effective denoising pass, not once
            // total — leaving this unscaled made memory time (and so
            // latency) stay flat across a 1-step vs 1000-step CFG sample
            // while FLOPs correctly scaled 2000x, decoupling latency from
            // the model's real cost.
            let param_bytes = (op.param_count as f64 * dtype_bytes * sample_factor).round() as u64;
            let bytes_accessed =
                param_bytes.saturating_add((op.activation_memory as f64 * sample_factor) as u64);
            compute_ir.op_flops.push(OpFlops {
                op_id: op.id,
                layer_id: op.layer_id.clone(),
                forward_flops: flops,
                backward_flops: flops * 2.0, // Standard approximation
                bytes_accessed,
            });
        }

        Ok(compute_ir)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let batch = ctx.config.training.batch_size;
        let _ = batch; // Reserved for future batch-aware compute metrics
                       // The universal schema stores sequence length in `training`; older model
                       // descriptions can still provide it under `model.global_params`.
        let seq = ctx
            .config
            .training
            .sequence_length
            .or(ctx.config.model.global_params.sequence_length)
            .unwrap_or(512);

        // Calculate forward FLOPs
        let forward_flops: f64 = output.op_flops.iter().map(|op| op.forward_flops).sum();

        // Published for ParallelismPass, which runs concurrently with
        // HardwarePass (rayon::join) and has no typed path to ComputeIR.
        ctx.set_metric("total_flops", forward_flops);

        // Calculate backward FLOPs (approximately 2× forward)
        let backward_flops: f64 = output.op_flops.iter().map(|op| op.backward_flops).sum();

        // Optimizer FLOPs based on optimizer type
        let optimizer_flops = match ctx.config.training.optimizer.to_lowercase().as_str() {
            "adam" | "adamw" => forward_flops * 0.1, // Adam: momentum + variance
            "sgd" => forward_flops * 0.02,           // SGD: minimal overhead
            "lamb" => forward_flops * 0.15,          // LAMB: layer-wise adaptation
            "adafactor" => forward_flops * 0.08,     // Adafactor: factorized states
            _ => forward_flops * 0.1,                // Default: Adam-like
        };

        // Total per step
        let total_step_flops = forward_flops + backward_flops + optimizer_flops;

        // MACs
        let macs = forward_flops / 2.0;

        // FLOPs per token
        // CORRECTION F01: flops_per_token = forward_flops / seq_len (pas batch×seq)
        // INVARIANT: flops_per_token × seq_len ≈ forward_flops (±0.1%)
        let flops_per_token = if seq > 0 {
            forward_flops / seq as f64
        } else {
            0.0
        };

        // FLOPs incremental decode (LLM avec KV cache)
        // Quand le KV cache est plein, l'attention passe de O(S²) à O(S)
        // Les projections QKV+O restent constantes → ~60% des FLOPs par token
        let has_attention = ctx
            .config
            .model
            .layers
            .iter()
            .any(|l| l.layer_type == neurax_parser::LayerType::Attention);
        let flops_incremental_decode = if has_attention && seq > 0 {
            flops_per_token * 0.6
        } else {
            0.0
        };

        // Vérification de cohérence interne (debug_assert pour attraper les erreurs)
        if seq > 0 && flops_per_token > 0.0 {
            let reconstructed = flops_per_token * seq as f64;
            let relative_err = (reconstructed - forward_flops).abs() / forward_flops;
            debug_assert!(
                relative_err < 0.001,
                "INCOHÉRENCE F01: flops_per_token({:.3e}) × seq_len({}) = {:.3e} ≠ forward_flops({:.3e})",
                flops_per_token, seq, reconstructed, forward_flops
            );
        }

        // FLOPs per layer
        let mut flops_per_layer = std::collections::HashMap::new();
        for op in &output.op_flops {
            let entry = flops_per_layer.entry(op.layer_id.clone()).or_insert(0.0);
            *entry += op.forward_flops;
        }

        // Estimate bytes accessed (simplified)
        let bytes_accessed = estimate_bytes_accessed(ctx);

        // Arithmetic intensity
        let arithmetic_intensity = if bytes_accessed > 0 {
            forward_flops / bytes_accessed as f64
        } else {
            0.0
        };

        // Determine complexity class
        let complexity_class = determine_complexity_class(ctx);

        let metrics = ComputeMetrics {
            total_flops: forward_flops,
            macs,
            flops_per_layer,
            flops_per_token,
            arithmetic_intensity,
            complexity_class,
            forward_flops,
            backward_flops,
            optimizer_flops,
            total_step_flops,
            flops_per_batch: forward_flops,
            bytes_accessed,
            flops_incremental_decode,
        };

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(
        &self,
        _output: &Self::Output,
        metrics: &Self::Metrics,
    ) -> Result<(), Self::PassError> {
        if metrics.forward_flops <= 0.0 {
            return Err(ComputeError::ZeroFlops);
        }
        if metrics.backward_flops <= 0.0 {
            return Err(ComputeError::ZeroFlops);
        }
        Ok(())
    }
}

fn estimate_bytes_accessed(ctx: &NeuraxContext) -> u64 {
    // Simplified estimation: parameters + activations
    let param_bytes = ctx
        .config
        .model
        .layers
        .iter()
        .map(|l| {
            let params = crate::architecture::calculate_layer_params(&neurax_parser::Layer {
                id: l.id.clone(),
                layer_type: l.layer_type,
                input_shape: l.input_shape.clone(),
                output_shape: l.output_shape.clone(),
                params: l.params.clone(),
                custom_equations: l.custom_equations.clone(),
            });
            (params as f64 * neurax_formulas::dtype_bytes(&ctx.config.training.precision)).round()
                as u64
        })
        .sum::<u64>();

    // Activation bytes (rough estimate)
    let batch = ctx.config.training.batch_size;
    let seq = ctx
        .config
        .training
        .sequence_length
        .or(ctx.config.model.global_params.sequence_length)
        .unwrap_or(512);
    let hidden = ctx.config.model.global_params.embedding_dim.unwrap_or(512);
    let activation_bytes = ((batch * seq * hidden) as f64
        * neurax_formulas::dtype_bytes(&ctx.config.training.precision))
    .round() as u64;

    param_bytes + activation_bytes * ctx.config.model.layers.len() as u64
}

fn determine_complexity_class(ctx: &NeuraxContext) -> ComplexityClass {
    // Check if model has attention layers
    let has_attention = ctx
        .config
        .model
        .layers
        .iter()
        .any(|l| l.layer_type == neurax_parser::LayerType::Attention);

    if has_attention {
        ComplexityClass::Quadratic // Attention is O(n²)
    } else {
        ComplexityClass::Linear
    }
}
