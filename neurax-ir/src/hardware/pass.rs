//! Hardware IR pass

use super::{Bottleneck, GpuProfile, HardwareIR, HardwareMetrics, LayerTiming, RooflineModel};
use crate::compute::ComputeIR;
use crate::error::HardwareError;
use crate::memory::MemoryIR;
use crate::parallelism::ParallelismIR;
use crate::traits::IrPass;
use crate::NeuraxContext;

/// Hardware pass implementation

/// True for the 16-bit float formats that use tensor cores.
///
/// Configs, papers and the UI write `bf16`; only the longer `bfloat16` spelling
/// was matched before, so bf16 runs silently fell through to the fp32 branch and
/// were costed with fp32 throughput and efficiency.
fn is_half_precision(precision: &str) -> bool {
    matches!(precision, "fp16" | "float16" | "bf16" | "bfloat16")
}

pub struct HardwarePass;

impl IrPass for HardwarePass {
    type Input = (ComputeIR, MemoryIR, ParallelismIR);
    type Output = HardwareIR;
    type Metrics = HardwareMetrics;
    type PassError = HardwareError;

    fn name(&self) -> &'static str {
        "HardwareIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let (compute_ir, memory_ir, _parallel_ir) = input;
        let mut hw_ir = HardwareIR::default();
        // The true, un-sharded parameter count — a gradient all-reduce needs
        // to know the real buffer size, not this GPU's ZeRO-reduced share of
        // it (using the sharded figure made higher ZeRO stages look like
        // they reduce communication, when ZeRO-3 specifically increases it).
        hw_ir.parameter_bytes = memory_ir.metrics.total_parameter_bytes;
        hw_ir.total_flops = compute_ir.metrics.total_flops;

        // Get GPU profile from JSON config or fallback to database
        let gpu_config = ctx.config.hardware.gpus.first();
        let gpu_name = gpu_config.map(|g| g.name.as_str()).unwrap_or("Generic-GPU");

        // Try to get base profile from database, then override with JSON values
        let mut gpu_profile = ctx
            .gpu_db
            .get_gpu(gpu_name)
            .map(GpuProfile::from)
            .unwrap_or_else(|| {
                // GPU not in database - create profile from JSON config
                let mut profile = GpuProfile::default();
                if let Some(gpu) = gpu_config {
                    profile.name = gpu.name.clone();
                    if let Some(memory_gb) = gpu.memory_gb {
                        profile.vram_gb = memory_gb;
                    }
                    if let Some(tflops) = gpu.tflops_fp16.or(gpu.tflops_fp32) {
                        profile.peak_tflops = tflops;
                    }
                    if let Some(bandwidth) = gpu.memory_bandwidth_gbs {
                        profile.memory_bandwidth = bandwidth;
                    }
                }
                profile
            });

        // Point `peak_tflops` at the precision the run actually uses. The
        // database profile lands on the fp16 figure by default, which understates
        // int8/fp8 runs and misses bf16 on parts where the two differ.
        {
            let precision = ctx.config.training.precision.as_str();
            let by_precision = match precision {
                "fp32" | "float32" => gpu_profile.tflops_fp32,
                "fp16" | "float16" => gpu_profile.tflops_fp16,
                "bf16" | "bfloat16" => gpu_profile.tflops_bf16,
                "int8" => gpu_profile.tflops_int8,
                "fp8" | "float8" => gpu_profile.tflops_fp8,
                _ => 0.0,
            };
            if by_precision > 0.0 {
                gpu_profile.peak_tflops = by_precision;
            }
        }

        // Override the database profile only where the config states a value.
        //
        // These fields are `Option` precisely so that "unspecified" stays
        // distinguishable from "specified as some number": treating an absent
        // spec as a real one used to replace the database's true figures for a
        // named GPU with placeholder constants.
        if let Some(gpu) = gpu_config {
            let precision = &ctx.config.training.precision;
            if precision == "fp8" {
                if let Some(tflops) = gpu.tflops_fp8 {
                    gpu_profile.peak_tflops = tflops;
                } else if let Some(tflops) = gpu.tflops_fp16 {
                    gpu_profile.peak_tflops = tflops * 2.0; // fp8 is 2x fp16
                }
            } else if is_half_precision(precision) {
                if let Some(tflops) = gpu.tflops_fp16 {
                    gpu_profile.peak_tflops = tflops;
                }
            } else if let Some(tflops) = gpu.tflops_fp32 {
                gpu_profile.peak_tflops = tflops;
            }

            if let Some(bandwidth) = gpu.memory_bandwidth_gbs {
                gpu_profile.memory_bandwidth = bandwidth;
            }

            // Tensor cores enable higher TFLOPs for FP16/BF16
            if gpu.tensor_cores.unwrap_or(true) && is_half_precision(precision) {
                gpu_profile.tensor_core_tflops = gpu_profile.peak_tflops * 2.0;
            }
        }

        hw_ir.gpu_profile = gpu_profile;

        // Build roofline model (Industrial level)
        hw_ir.roofline = RooflineModel {
            compute_roof: hw_ir.gpu_profile.peak_tflops * hw_ir.gpu_profile.efficiency_factor,
            memory_roof: hw_ir.gpu_profile.memory_bandwidth,
            ridge_point: calculate_ridge_point(&hw_ir.gpu_profile, &ctx.config.training.precision),
            level: crate::hardware::RooflineLevel::Industrial,
            l2_roof: Some(hw_ir.gpu_profile.l2_bandwidth_tb_s),
            sram_roof: Some(hw_ir.gpu_profile.sram_bandwidth_tb_s),
            overlap_factor: 0.3,
            kernel_launch_overhead_us: 5.0,
        };

        // Calculate per-layer timings
        hw_ir.per_layer_timings = calculate_layer_timings(&compute_ir, &hw_ir.gpu_profile);

        Ok(hw_ir)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let batch = ctx.config.training.batch_size;
        let seq = ctx
            .config
            .model
            .global_params
            .sequence_length
            .unwrap_or(512);
        let precision = &ctx.config.training.precision;

        // Count attention vs MLP layers for efficiency estimation
        let attention_count = ctx
            .config
            .model
            .layers
            .iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Attention)
            .count();
        let mlp_count = ctx
            .config
            .model
            .layers
            .iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Mlp)
            .count();
        // Convolution-family ops: many small kernels (conv + batchnorm +
        // activation) rather than one large GEMM — real throughput on tensor
        // cores lands far below what a transformer-shaped efficiency table
        // would predict. Before this was counted, a pure-CNN model had zero
        // recognized layers (Conv wasn't in the attention/mlp weighting) and
        // fell back to the 1.0 "no penalty" neutral default below, which
        // predicted ResNet-50 inference throughput on a single A100 at
        // ~23,300 img/s — about 7x NVIDIA's own published real number
        // (~3,229 img/s, mixed precision, batch 256; see
        // `published_hardware_scaling.rs`'s companion CNN reference).
        // Includes every layer type whose own param/FLOPs formula
        // (architecture/mod.rs, operator/pass.rs) already calls
        // `conv::conv2d_params`/`cnn_blocks::resnet_basic_block_*` — i.e.
        // types the compiler already treats as "this is a convolution" at
        // the formula level. GAN's GeneratorBlock/DiscriminatorBlock (plain
        // conv2d_params) and diffusion's UnetBlock/ResnetBlock/DownBlock/
        // UpBlock/MidBlock/VaeEncoder/VaeDecoder/NoisePredictor (the same
        // resnet_basic_block_params/conv2d_params as CNN's own blocks) used
        // to fall outside this count and get the 1.0 neutral fallback
        // despite running the literal same kernels as a recognized Conv
        // layer — not a new guess, a real formula-equivalence gap.
        let conv_count = ctx
            .config
            .model
            .layers
            .iter()
            .filter(|l| {
                matches!(
                    l.layer_type,
                    neurax_parser::LayerType::Conv
                        | neurax_parser::LayerType::ResidualBlock
                        | neurax_parser::LayerType::ResnetBottleneck
                        | neurax_parser::LayerType::Mbconv
                        | neurax_parser::LayerType::Inception
                        | neurax_parser::LayerType::DenseBlock
                        | neurax_parser::LayerType::ConvnextBlock
                        | neurax_parser::LayerType::ShuffleUnit
                        | neurax_parser::LayerType::C2f
                        | neurax_parser::LayerType::Transition
                        | neurax_parser::LayerType::Detection
                        | neurax_parser::LayerType::ProgressiveBlock
                        | neurax_parser::LayerType::GeneratorBlock
                        | neurax_parser::LayerType::DiscriminatorBlock
                        | neurax_parser::LayerType::UnetBlock
                        | neurax_parser::LayerType::ResnetBlock
                        | neurax_parser::LayerType::DownBlock
                        | neurax_parser::LayerType::UpBlock
                        | neurax_parser::LayerType::MidBlock
                        | neurax_parser::LayerType::VaeEncoder
                        | neurax_parser::LayerType::VaeDecoder
                        | neurax_parser::LayerType::NoisePredictor
                )
            })
            .count();
        // Same formula-equivalence principle for MLP: MoE's experts and
        // router are `mlp::gated_mlp_params`/a dense gating matrix under
        // the hood (architecture/mod.rs's MoE/MoeRouter/MoeSharedExpert
        // arms) — mechanically the same operation as a recognized Mlp
        // layer, so they get the same efficiency treatment rather than the
        // 1.0 fallback.
        let mlp_shaped_count = ctx
            .config
            .model
            .layers
            .iter()
            .filter(|l| {
                matches!(
                    l.layer_type,
                    neurax_parser::LayerType::MoE
                        | neurax_parser::LayerType::MoeRouter
                        | neurax_parser::LayerType::MoeSharedExpert
                )
            })
            .count();

        // Real-world GPU efficiency factors based on operation type
        // Attention is memory-bound (lower efficiency)
        // MLP is compute-bound (higher efficiency)
        let attention_efficiency = match precision.as_str() {
            "fp32" => 0.45,
            "fp16" | "bf16" | "bfloat16" => 0.55, // FlashAttention helps
            "fp8" => 0.65,
            _ => 0.45,
        };
        let mlp_efficiency = match precision.as_str() {
            "fp32" => 0.75,
            "fp16" | "bf16" | "bfloat16" => 0.85, // Tensor cores shine
            "fp8" => 0.92,
            _ => 0.75,
        };
        // The fp16/bf16 value is directly calibrated against NVIDIA's own
        // published ResNet-50 v1.5 mixed-precision inference number on a
        // single A100 40GB (NGC "ResNet v1.5 for TensorFlow" performance
        // page): matching NEURAX's own forward-FLOPs figure for that model
        // to the real 3,229 img/s implies an effective efficiency of
        // ~0.139. The other precisions follow the same shape as the
        // attention/mlp tables (no independent reference for them yet) and
        // should be treated as reasoned estimates, not individually
        // verified figures.
        let conv_efficiency = match precision.as_str() {
            "fp32" => 0.12,
            "fp16" | "bf16" | "bfloat16" => 0.14,
            "fp8" => 0.16,
            _ => 0.12,
        };

        // Weighted average efficiency based on layer distribution. Only
        // recognized layer families are counted — a repeated-block
        // architecture (LAYER_STACK, the shape every template on this app's
        // own catalogue compiles to) tags its block as `Custom` instead, so
        // every count can be 0 for essentially every real model. Dividing
        // by the resulting 0.0 efficiency below used to turn
        // every per-layer compute time into +/-Infinity; summing them produced
        // NaN, `f64::max` silently dropped it into `latency_ms`, and NaN then
        // serialized as a bare JSON `null` for gpu_utilization — read by the
        // UI as "no hardware data", contradicting the VRAM/GPU numbers sitting
        // right next to it, which come from a different, unaffected path.
        // With nothing recognized to weight, there is nothing to derate by
        // — 1.0 (no efficiency penalty) is the neutral answer, not a guess.
        let recognized_layers = attention_count + mlp_count + conv_count + mlp_shaped_count;
        let gpu_efficiency = if recognized_layers > 0 {
            (attention_count as f64 * attention_efficiency
                + (mlp_count + mlp_shaped_count) as f64 * mlp_efficiency
                + conv_count as f64 * conv_efficiency)
                / recognized_layers as f64
        } else {
            1.0
        };

        // Attention layer ids — scopes FlashAttention's memory reduction to
        // the layers it actually touches (see below).
        let attention_layer_ids: std::collections::HashSet<&str> = ctx
            .config
            .model
            .layers
            .iter()
            .filter(|l| {
                matches!(
                    l.layer_type,
                    neurax_parser::LayerType::Attention
                        | neurax_parser::LayerType::SelfAttention
                        | neurax_parser::LayerType::CrossAttention
                        | neurax_parser::LayerType::GraphAttentionNet
                )
            })
            .map(|l| l.id.as_str())
            .collect();

        // FlashAttention-family kernels are the standard attention
        // implementation in reduced precision — they exist to feed tensor
        // cores, which is exactly the condition `attention_efficiency` above
        // already credits ("FlashAttention helps"). There's no per-model
        // signal for which attention kernel a config actually used, so this
        // ties the estimate to that same precision fact instead of assuming
        // every model everywhere runs an IO-optimized kernel.
        let flash_attention_enabled =
            is_half_precision(precision) || matches!(precision.as_str(), "fp8" | "float8");

        // Compute time with efficiency factor
        let compute_time_ms: f64 = output
            .per_layer_timings
            .iter()
            .map(|t| t.compute_time_ms / gpu_efficiency)
            .sum();

        // Memory time — FlashAttention's ~4x reduction applies only to the
        // attention layers it optimizes, not to every layer in the model (a
        // Conv/MLP layer's HBM traffic is unaffected by attention elsewhere
        // using a memory-efficient kernel).
        let memory_time_ms: f64 = output
            .per_layer_timings
            .iter()
            .map(|t| {
                let factor = if flash_attention_enabled
                    && attention_layer_ids.contains(t.layer_id.as_str())
                {
                    0.25
                } else {
                    1.0
                };
                t.memory_time_ms * factor
            })
            .sum();

        // Total FLOPs from compute IR with efficiency factor
        let total_flops = output
            .per_layer_timings
            .iter()
            .map(|t| {
                t.compute_time_ms
                    * output.gpu_profile.effective_tflops(precision)
                    * gpu_efficiency
                    * 1e9
                    / 1000.0
            })
            .sum::<f64>();

        // Arithmetic intensity (FLOPs/byte)
        let arithmetic_intensity =
            if memory_time_ms > 0.0 && output.gpu_profile.memory_bandwidth > 0.0 {
                let bytes = memory_time_ms * output.gpu_profile.memory_bandwidth * 1e6;
                total_flops / bytes.max(1.0)
            } else {
                0.0
            };

        // Communication overhead for multi-GPU
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let interconnect_bw = ctx.config.hardware.interconnect_bandwidth_gbs * 1e9; // bytes/s
                                                                                    // A data-parallel all-reduce exchanges the gradients, i.e. one buffer
                                                                                    // the size of the parameters — not the model's whole HBM traffic.
        let param_bytes = output.parameter_bytes;

        // Ring all-reduce moves 2·(N-1)/N × buffer per rank. ZeRO-1/2 keep
        // this baseline (Rajbhandari et al. 2020, "ZeRO": stage 1/2 change
        // memory, not communication volume — still 2Ψ). ZeRO-3 additionally
        // has to gather its sharded parameters back on demand, which the
        // same paper's own analysis puts at 3Ψ total — 1.5× the baseline.
        let zero_comm_multiplier = if ctx.config.training.zero_stage >= 3 {
            1.5
        } else {
            1.0
        };
        let communication_overhead_ms = if num_gpus > 1 && interconnect_bw > 0.0 {
            let factor = 2.0 * (num_gpus - 1) as f64 / num_gpus as f64 * zero_comm_multiplier;
            (param_bytes as f64 * factor / interconnect_bw) * 1000.0
        } else {
            0.0
        };

        // Total latency including communication
        // Divide by total GPU count for parallel execution
        let num_gpus_f64 = num_gpus.max(1) as f64;
        let parallel_compute_time_ms = compute_time_ms / num_gpus_f64;
        let parallel_memory_time_ms = memory_time_ms / num_gpus_f64;

        // Kernel launches (estimate) — moved above `latency_ms` so its own
        // overhead can be added to it below, rather than only reported
        // alongside a latency that never accounted for it. Previously a
        // flat `layers.len() * 2` guess, unrelated to how many atomic
        // operations the model actually decomposes into (an attention
        // layer alone is QKV projection + scores + softmax + output
        // projection — well more than 2, while a plain normalization layer
        // is one). `output.per_layer_timings` has exactly one entry per
        // real atomic op from the operator/compute passes
        // (`calculate_layer_timings` maps 1:1 over `compute_ir.op_flops`),
        // which is what actually gets scheduled as a kernel. Scaled by the
        // same diffusion sample factor as FLOPs/bytes (compute/pass.rs): a
        // 1000-step CFG sample launches every one of the U-Net's kernels
        // 2000 times, not once — leaving this unscaled would have made
        // kernel overhead a fixed floor that swamped the real, correctly
        // scaled FLOPs-driven cost for a small model.
        let kernel_launch_count = (output.per_layer_timings.len() as f64
            * crate::compute::diffusion_sample_factor(ctx))
        .round() as usize;

        // `output.roofline` was already built at "Industrial" level
        // (overlap_factor: 0.3, kernel_launch_overhead_us: 5.0 —
        // hardware/pass.rs::build()) but neither field was ever read here:
        // latency took the plain max of compute/memory time (implicitly
        // assuming 100% overlap between them, which no real GPU achieves)
        // and kernel launch overhead was computed into `kernel_launch_count`
        // only to be reported, never actually added to the latency it
        // should inflate — exactly what `hardware_corrections.rs`'s F05
        // tests claimed to check without ever calling real code.
        let max_t = parallel_compute_time_ms.max(parallel_memory_time_ms);
        let min_t = parallel_compute_time_ms.min(parallel_memory_time_ms);
        let overlap_adjusted_ms = max_t + min_t * (1.0 - output.roofline.overlap_factor);
        let kernel_overhead_ms =
            output.roofline.kernel_launch_overhead_us * kernel_launch_count as f64 / 1000.0;
        let latency_ms = overlap_adjusted_ms + communication_overhead_ms + kernel_overhead_ms;

        // Determine bottleneck
        let bottleneck = if compute_time_ms > memory_time_ms * 1.5 {
            Bottleneck::ComputeBound
        } else if memory_time_ms > compute_time_ms * 1.5 {
            Bottleneck::MemoryBound
        } else {
            Bottleneck::Balanced
        };

        // GPU utilization — must use the same per-GPU (parallel) compute
        // time that actually fed `latency_ms`, not the pre-division,
        // whole-batch `compute_time_ms`. Using the un-divided figure here
        // made utilization scale up with GPU count instead of staying
        // bounded to [0, 1]: an 8-GPU run could — and did, for
        // stable_diffusion_1.5.json — report "486% utilization".
        let gpu_utilization = if latency_ms > 0.0 {
            parallel_compute_time_ms / latency_ms
        } else {
            0.0
        };

        // Throughput
        let samples_per_s = if latency_ms > 0.0 {
            1000.0 / latency_ms * batch as f64
        } else {
            0.0
        };

        let throughput_tokens_per_s = samples_per_s * seq as f64;

        // Tensor core utilization
        let tensor_core_utilization = calculate_tensor_core_utilization(ctx);

        // Effective TFLOPS = total_flops / latency_seconds / 1e12.
        // latency_ms is milliseconds, so that's `total_flops * 1000.0 /
        // latency_ms / 1e12`, i.e. `total_flops / latency_ms / 1e9` — this
        // used to carry an extra `/ 1000.0` (never caught because
        // `total_flops` was hardcoded to 0.0 until now, which made the whole
        // branch return 0 regardless of the formula underneath it).
        let effective_tflops = if latency_ms > 0.0 && batch > 0 && seq > 0 {
            output.total_flops / latency_ms / 1e9
        } else {
            0.0
        };

        let metrics = HardwareMetrics {
            latency_ms,
            throughput_tokens_per_s,
            gpu_utilization,
            tensor_core_utilization,
            kernel_launch_count,
            bottleneck,
            effective_tflops,
            memory_bandwidth_achieved: output.gpu_profile.memory_bandwidth * gpu_utilization,
            samples_per_s,
            roofline_position: calculate_roofline_position(&output.roofline, arithmetic_intensity),
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
        if metrics.latency_ms <= 0.0 {
            return Err(HardwareError::InvalidLatency(
                "Latency is zero or negative".to_string(),
            ));
        }
        if metrics.gpu_utilization <= 0.0 {
            return Err(HardwareError::RooflineFailed(
                "GPU utilization is zero".to_string(),
            ));
        }
        Ok(())
    }
}

fn calculate_ridge_point(gpu: &GpuProfile, _precision: &str) -> f64 {
    let tflops = gpu.peak_tflops * gpu.efficiency_factor;
    let bandwidth = gpu.memory_bandwidth;
    // Ridge point = TFLOPS / Bandwidth (FLOPs/byte)
    tflops * 1e12 / (bandwidth * 1e9)
}

fn calculate_layer_timings(
    compute_ir: &crate::compute::ComputeIR,
    gpu: &GpuProfile,
) -> Vec<LayerTiming> {
    compute_ir
        .op_flops
        .iter()
        .map(|op| {
            // Compute time: FLOPs / achievable FLOP/s, in milliseconds.
            //
            // `peak_tflops` is in TFLOP/s, so it converts with 1e12. The scale
            // here used to be 1e9, which treats teraflops as gigaflops and made
            // every layer time — and the latency, throughput and cost derived
            // from it — a thousand times too large.
            let achievable_flops_per_s = gpu.peak_tflops * gpu.efficiency_factor * 1e12;
            let compute_time_ms = if achievable_flops_per_s > 0.0 {
                op.forward_flops / achievable_flops_per_s * 1000.0
            } else {
                0.0
            };
            // Real memory time: this op's own weight+activation bytes
            // (`OpFlops::bytes_accessed`, carried over from the operator
            // pass) over the GPU's actual HBM bandwidth. This replaces a
            // flat `compute_time_ms * 0.5` guess that ignored the model's
            // real size and the GPU's real bandwidth entirely — two models
            // with identical FLOPs but very different memory footprints
            // (e.g. a wide-and-shallow vs narrow-and-deep network) used to
            // report the exact same memory time.
            let bandwidth_bytes_per_s = gpu.memory_bandwidth * 1e9;
            let memory_time_ms = if bandwidth_bytes_per_s > 0.0 {
                op.bytes_accessed as f64 / bandwidth_bytes_per_s * 1000.0
            } else {
                0.0
            };

            LayerTiming {
                layer_id: op.layer_id.clone(),
                compute_time_ms,
                memory_time_ms,
                total_time_ms: compute_time_ms.max(memory_time_ms),
            }
        })
        .collect()
}

fn calculate_tensor_core_utilization(ctx: &NeuraxContext) -> f64 {
    // Estimate based on layer types
    let total_layers = ctx.config.model.layers.len();
    if total_layers == 0 {
        return 0.0;
    }

    let tc_layers = ctx
        .config
        .model
        .layers
        .iter()
        .filter(|l| {
            matches!(
                l.layer_type,
                neurax_parser::LayerType::Attention
                    | neurax_parser::LayerType::Mlp
                    | neurax_parser::LayerType::Dense
                    | neurax_parser::LayerType::Conv
            )
        })
        .count();

    tc_layers as f64 / total_layers as f64
}

/// Calculate roofline position (0.0 = memory-bound, 1.0 = compute-bound)
fn calculate_roofline_position(
    roofline: &crate::hardware::RooflineModel,
    arithmetic_intensity: f64,
) -> f64 {
    if roofline.ridge_point <= 0.0 {
        return 0.5;
    }
    // Position relative to ridge point
    // < 1.0 = memory-bound, > 1.0 = compute-bound
    let position = arithmetic_intensity / roofline.ridge_point;
    // Clamp to [0.0, 1.0] range
    position.clamp(0.0, 1.0)
}
