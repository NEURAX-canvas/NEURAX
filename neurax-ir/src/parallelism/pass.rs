//! Parallelism IR pass

use super::{ParallelStrategy, ParallelismIR, ParallelismMetrics};
use crate::error::ParallelismError;
use crate::graph::GraphIR;
use crate::memory::MemoryIR;
use crate::traits::IrPass;
use crate::NeuraxContext;

/// Parallelism pass implementation
pub struct ParallelismPass;

impl IrPass for ParallelismPass {
    type Input = (MemoryIR, GraphIR);
    type Output = ParallelismIR;
    type Metrics = ParallelismMetrics;
    type PassError = ParallelismError;

    fn name(&self) -> &'static str {
        "ParallelismIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let (memory_ir, graph_ir) = input;
        // True, un-sharded byte counts. compute_metrics()'s own per-ZeRO-stage
        // arithmetic (see `memory_per_gpu` below) already divides these down
        // by num_gpus itself; feeding it MemoryPass's already-sharded
        // per-GPU figures here would double-shard the result. Same reason
        // hardware/pass.rs::build() uses the true total for its all-reduce
        // sizing.
        let mut parallel_ir = ParallelismIR {
            parameter_bytes: memory_ir.metrics.total_parameter_bytes,
            gradient_bytes: memory_ir.metrics.total_gradient_bytes,
            optimizer_bytes: memory_ir.metrics.total_optimizer_bytes,
            ..Default::default()
        };

        // Analyze available strategies
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let gpu_vram = ctx.primary_gpu_vram_bytes();

        // Data parallel
        let dp_efficiency = calculate_dp_efficiency(ctx, memory_ir.metrics.parameter_memory_bytes);
        parallel_ir.strategies.push(ParallelStrategy::DataParallel {
            num_gpus,
            efficiency: dp_efficiency,
        });

        // The real minimum GPU count this model needs to fit at all — real
        // bug found auditing 100B+ models: `optimal_gpu_count` used to just
        // echo the configured GPU count back (`num_gpus`), so a 405B model
        // configured with 1 GPU reported "1 GPU is optimal" in the same
        // breath as `oom_risk: Overflow`. This was already computed here
        // and thrown away (`_num_splits`) — now it's kept and actually used
        // in `compute_metrics()` below.
        parallel_ir.minimum_gpus_to_fit = if gpu_vram > 0 {
            (memory_ir.metrics.peak_vram_bytes as f64 / gpu_vram as f64)
                .ceil()
                .max(1.0) as u32
        } else {
            num_gpus.max(1)
        };

        // Model parallel if model doesn't fit on one GPU
        if memory_ir.metrics.peak_vram_bytes > gpu_vram {
            parallel_ir
                .strategies
                .push(ParallelStrategy::ModelParallel {
                    splits: vec![], // Would be computed properly
                });
            parallel_ir.metrics.model_parallel_feasible = true;
        } else {
            parallel_ir.metrics.model_parallel_feasible = false;
        }

        // Pipeline parallel
        if num_gpus > 1 && graph_ir.metrics.graph_depth > 2 {
            parallel_ir
                .strategies
                .push(ParallelStrategy::PipelineParallel {
                    stages: num_gpus,
                    micro_batches: num_gpus * 2,
                    bubble_ratio: 1.0 / (num_gpus * 2) as f64,
                });
            parallel_ir.metrics.pipeline_stages = Some(num_gpus);
        }

        // ZeRO
        if ctx.config.training.zero_stage > 0 {
            let zero_memory =
                calculate_zero_memory(memory_ir, ctx.config.training.zero_stage, num_gpus);
            parallel_ir.strategies.push(ParallelStrategy::ZeRO {
                stage: ctx.config.training.zero_stage,
                memory_per_gpu: zero_memory,
            });
        }

        // Hybrid (3D parallelism)
        if num_gpus >= 8 {
            let dp = ctx.config.training.parallelism.data_parallel;
            let tp = ctx.config.training.parallelism.tensor_parallel;
            let pp = ctx.config.training.parallelism.pipeline_parallel;
            if dp > 1 || tp > 1 || pp > 1 {
                parallel_ir
                    .strategies
                    .push(ParallelStrategy::Hybrid { dp, tp, pp });
            }
        }

        // Select optimal strategy
        parallel_ir.optimal_strategy =
            select_optimal_strategy(&parallel_ir.strategies, memory_ir, ctx);

        Ok(parallel_ir)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let _gpu_vram = ctx.primary_gpu_vram_bytes();

        // Memory-pass figures carried over on the IR from `build()`.
        let param_bytes = output.parameter_bytes;
        let gradient_bytes = output.gradient_bytes;
        let optimizer_bytes = output.optimizer_bytes;

        // Calculate communication overhead
        let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9;

        // ZeRO-1/2 keep the baseline 2Ψ communication volume (Rajbhandari et
        // al. 2020, "ZeRO") — only memory changes. ZeRO-3 additionally
        // gathers its sharded parameters back on demand: the same paper puts
        // its total at 3Ψ, i.e. 1.5× the baseline. Same fix as
        // hardware/pass.rs's identical formula.
        let zero_comm_multiplier = if ctx.config.training.zero_stage >= 3 {
            1.5
        } else {
            1.0
        };
        let allreduce_time_ms = if num_gpus > 1 && interconnect_bw > 0.0 {
            // Ring All-Reduce: 2 * (N-1)/N * data_size / bandwidth
            let factor = 2.0 * (num_gpus - 1) as f64 / num_gpus as f64 * zero_comm_multiplier;
            (param_bytes as f64 * factor / interconnect_bw) * 1000.0
        } else {
            0.0
        };

        // Compute time for one step. This pass runs concurrently with
        // HardwarePass (rayon::join) and its typed `Input` is only
        // `(MemoryIR, GraphIR)`, so it can't reach ComputeIR's FLOPs
        // directly — `ctx.get_metric` is the one channel available for data
        // a pass needs but doesn't own, and `ComputePass` (which runs
        // earlier, before the join) publishes it there.
        let total_flops = ctx.get_metric("total_flops").unwrap_or(0.0);
        let gpu_name = ctx
            .config
            .hardware
            .gpus
            .first()
            .map(|g| g.name.as_str())
            .unwrap_or("Generic-GPU");
        let compute_time_ms = ctx
            .gpu_db
            .get_gpu(gpu_name)
            .map(|spec| spec.compute_time_ms(total_flops, &ctx.config.training.precision))
            .unwrap_or(0.0);

        let communication_overhead = if compute_time_ms > 0.0 {
            allreduce_time_ms / (compute_time_ms + allreduce_time_ms)
        } else {
            0.0
        };

        // Data parallel efficiency - use improved model
        let data_parallel_efficiency = calculate_dp_efficiency(ctx, param_bytes);

        // Memory per GPU
        let memory_per_gpu = match &output.optimal_strategy {
            ParallelStrategy::ZeRO { stage, .. } => {
                let base = param_bytes + gradient_bytes + optimizer_bytes;
                match stage {
                    1 => base - optimizer_bytes / num_gpus as u64,
                    2 => base / 2,
                    3 => base / num_gpus as u64,
                    _ => base,
                }
            }
            ParallelStrategy::Hybrid { tp, .. } if *tp > 1 => param_bytes / *tp as u64,
            _ => param_bytes,
        };

        // Scaling efficiency curve
        let scaling_efficiency_curve = calculate_scaling_curve(ctx, param_bytes);

        let metrics = ParallelismMetrics {
            data_parallel_efficiency,
            model_parallel_feasible: output.metrics.model_parallel_feasible,
            pipeline_stages: output.metrics.pipeline_stages,
            communication_overhead,
            // Never recommend fewer GPUs than the model actually needs to
            // fit, even if that's more than the caller configured — see
            // `minimum_gpus_to_fit`'s own doc for the bug this fixes.
            optimal_gpu_count: num_gpus.max(output.minimum_gpus_to_fit),
            memory_per_gpu_bytes: memory_per_gpu,
            scaling_efficiency_curve,
            allreduce_time_ms,
            compute_time_ms,
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
        if metrics.optimal_gpu_count == 0 {
            return Err(ParallelismError::InvalidConfiguration(
                "GPU count is zero".to_string(),
            ));
        }
        Ok(())
    }
}

fn calculate_dp_efficiency(ctx: &NeuraxContext, param_bytes: u64) -> f64 {
    let num_gpus = ctx.config.hardware.total_gpu_count();
    if num_gpus <= 1 {
        return 1.0;
    }

    let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9;

    // Real-world scaling efficiency factors (empirical data from large-scale training)
    // Based on: GPT-3, Megatron-LM, DeepSpeed benchmarks
    let base_efficiency: f64 = match num_gpus {
        1 => 1.0,
        2 => 0.95,   // 95% efficiency with 2 GPUs
        4 => 0.92,   // 92% efficiency with 4 GPUs
        8 => 0.88,   // 88% efficiency with 8 GPUs
        16 => 0.85,  // 85% efficiency with 16 GPUs
        32 => 0.82,  // 82% efficiency with 32 GPUs
        64 => 0.78,  // 78% efficiency with 64 GPUs
        128 => 0.75, // 75% efficiency with 128 GPUs
        256 => 0.72, // 72% efficiency with 256 GPUs
        512 => 0.68, // 68% efficiency with 512 GPUs
        _ => 0.65,   // 65% efficiency for 1024+ GPUs
    };

    // Adjust for interconnect bandwidth and type
    // Use both bandwidth and interconnect type from config
    let interconnect_type = ctx.config.hardware.interconnect.to_lowercase();
    let interconnect_factor = if interconnect_bw >= 600e9 || interconnect_type.contains("nvlink") {
        1.0 // NVLink
    } else if interconnect_bw >= 200e9
        || interconnect_type.contains("infiniband")
        || interconnect_type.contains("ib")
    {
        0.90 // InfiniBand HDR
    } else if interconnect_bw >= 100e9
        || interconnect_type.contains("ethernet")
        || interconnect_type.contains("roce")
    {
        0.80 // InfiniBand EDR or RoCE
    } else if interconnect_bw > 0.0 {
        0.70 // Slower interconnect
    } else {
        0.50 // No fast interconnect specified
    };

    // Communication overhead increases with model size
    // Larger models have more gradient synchronization overhead
    let param_gb = param_bytes as f64 / 1e9;
    let size_penalty = if param_gb > 100.0 {
        0.95 // 5% penalty for very large models (>100GB params)
    } else if param_gb > 10.0 {
        0.98 // 2% penalty for large models (>10GB params)
    } else {
        1.0 // No penalty for smaller models
    };

    (base_efficiency * interconnect_factor * size_penalty).clamp(0.5_f64, 1.0_f64)
}

fn calculate_zero_memory(memory_ir: &MemoryIR, stage: u8, num_gpus: u32) -> u64 {
    let base = memory_ir.metrics.parameter_memory_bytes
        + memory_ir.metrics.gradient_memory_bytes
        + memory_ir.metrics.optimizer_state_bytes;

    match stage {
        1 => base - memory_ir.metrics.optimizer_state_bytes / num_gpus as u64,
        2 => base / 2,
        3 => base / num_gpus as u64,
        _ => base,
    }
}

fn select_optimal_strategy(
    strategies: &[ParallelStrategy],
    memory_ir: &MemoryIR,
    ctx: &NeuraxContext,
) -> ParallelStrategy {
    let gpu_vram = ctx.primary_gpu_vram_bytes();

    // If model fits in single GPU, use data parallel
    if memory_ir.metrics.peak_vram_bytes <= gpu_vram {
        return strategies.first().cloned().unwrap_or_default();
    }

    // Otherwise, prefer ZeRO-3 or model parallel
    strategies
        .iter()
        .find(|s| matches!(s, ParallelStrategy::ZeRO { stage: 3, .. }))
        .or_else(|| {
            strategies
                .iter()
                .find(|s| matches!(s, ParallelStrategy::ModelParallel { .. }))
        })
        .cloned()
        .unwrap_or_default()
}

fn calculate_scaling_curve(ctx: &NeuraxContext, param_bytes: u64) -> Vec<(u32, f64)> {
    let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9;
    let mut curve = Vec::new();

    for n in [1, 2, 4, 8, 16, 32, 64, 128] {
        if n == 1 {
            curve.push((n, 1.0));
        } else if interconnect_bw > 0.0 {
            let compute_time = 100.0;
            let comm_time =
                (param_bytes as f64 * 2.0 * (n - 1) as f64 / n as f64 / interconnect_bw) * 1000.0;
            let efficiency = compute_time / (compute_time + comm_time);
            curve.push((n, efficiency));
        }
    }

    curve
}
