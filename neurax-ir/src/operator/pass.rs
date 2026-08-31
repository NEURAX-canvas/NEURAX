//! Operator IR pass

use super::{AtomOp, OpType, OperatorIR, OperatorMetrics};
use crate::architecture::ArchitectureIR;
use crate::error::OperatorError;
use crate::tensor::Shape;
use crate::tensor::TensorIR;
use crate::traits::IrPass;
use crate::NeuraxContext;
use neurax_formulas::{attention, cnn_blocks, conv, embedding, gnn, mlp, moe, normalization};
use neurax_parser::LayerType;

/// Reads a `usize` out of a `global_params.extra` map, falling back when the
/// key is absent or not a plain non-negative integer — the same map
/// `GlobalResolutionContext` reads `num_nodes`/`num_edges` from, so a GNN
/// design's real graph size (if the client supplied one) reaches its FLOPs
/// the same way it reaches everything else this map backs.
fn extra_usize(
    extra: &std::collections::HashMap<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> usize {
    extra
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

/// Operator pass implementation
pub struct OperatorPass;

impl IrPass for OperatorPass {
    type Input = (TensorIR, ArchitectureIR);
    type Output = OperatorIR;
    type Metrics = OperatorMetrics;
    type PassError = OperatorError;

    fn name(&self) -> &'static str {
        "OperatorIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let (_tensor_ir, arch_ir) = input;
        let mut op_ir = OperatorIR::default();
        let batch = ctx.config.training.batch_size;
        let seq = ctx
            .config
            .training
            .sequence_length
            .or(arch_ir.global_params.sequence_length)
            .unwrap_or(512);
        let dtype = &ctx.config.training.precision;

        // Compute block_scale: same logic as ArchitecturePass
        let json_attention_count = arch_ir
            .layers
            .iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Attention)
            .count();
        let json_mlp_count = arch_ir
            .layers
            .iter()
            .filter(|l| l.layer_type == neurax_parser::LayerType::Mlp)
            .count();
        let json_block_count = json_attention_count.max(json_mlp_count).max(1);
        // Same rule as `repeat_scale_for`: only an explicitly declared depth
        // means the listed layers stand in for more than themselves.
        let block_scale = match arch_ir.global_params.num_layers.map(|n| n as usize) {
            Some(global_num_layers) if global_num_layers > json_block_count => {
                global_num_layers as f64 / json_block_count as f64
            }
            _ => 1.0_f64,
        };

        for layer in &arch_ir.layers {
            let mut layer_ops = Vec::new();

            let mut ops = decompose_layer_to_ops(layer, batch, seq, dtype, ctx);

            // Scale FLOPs for repeatable layers
            let is_repeatable = matches!(
                layer.layer_type,
                neurax_parser::LayerType::Attention
                    | neurax_parser::LayerType::Mlp
                    | neurax_parser::LayerType::Normalization
            );
            if is_repeatable && block_scale > 1.0 {
                for op in &mut ops {
                    op.flops *= block_scale;
                }
            }

            for op in ops {
                let op_id = op_ir.operations.len();
                layer_ops.push(op_id);
                op_ir.operations.push(op);
            }

            op_ir.layer_ops.insert(layer.id.clone(), layer_ops);
        }

        Ok(op_ir)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        _ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let mut metrics = OperatorMetrics {
            total_op_count: output.operations.len(),
            ..Default::default()
        };

        for op in &output.operations {
            // FLOPs per layer
            let entry = metrics
                .flops_per_layer
                .entry(op.layer_id.clone())
                .or_insert(0.0);
            *entry += op.flops;
            metrics.total_flops_approx += op.flops;

            // Op type distribution
            let type_str = op.op_type.as_str().to_string();
            *metrics.op_type_distribution.entry(type_str).or_insert(0) += 1;

            if op.is_custom {
                metrics.custom_op_count += 1;
            }
        }

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(
        &self,
        _output: &Self::Output,
        metrics: &Self::Metrics,
    ) -> Result<(), Self::PassError> {
        if metrics.total_op_count == 0 {
            return Err(OperatorError::UnknownOperator(
                "No operations generated".to_string(),
            ));
        }
        Ok(())
    }
}

/// Decompose a layer into atomic operations
fn decompose_layer_to_ops(
    layer: &crate::architecture::LayerDef,
    batch: usize,
    seq: usize,
    dtype: &str,
    ctx: &NeuraxContext,
) -> Vec<AtomOp> {
    let mut ops = Vec::new();

    match layer.layer_type {
        LayerType::Embedding => {
            let vocab = layer.params.vocab_size.unwrap_or(50000);
            let dim = layer
                .params
                .embedding_dim
                .unwrap_or(layer.params.hidden_size.unwrap_or(512));
            ops.push(AtomOp {
                id: 0,
                op_type: OpType::Embedding,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, dim]),
                flops: embedding::embedding_flops(batch, seq, dim),
                param_count: (vocab * dim) as u64,
                activation_memory: ((batch * seq * dim) as f64
                    * neurax_formulas::dtype_bytes(dtype))
                .round() as u64,
                is_custom: false,
            });
        }
        LayerType::Attention => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            // `.max(1)`: the parser rejects a zero head count, but this pass is
            // reachable from the published crate API without going through it,
            // and a zero here would be a division by zero below.
            let heads = layer.params.num_heads.unwrap_or(8).max(1);
            let kv_heads = layer.params.num_kv_heads.unwrap_or(heads).max(1);
            let causal = layer.params.causal;
            let head_dim = hidden / heads;

            // A bounded receptive field (sliding window, block-sparse, or
            // dilated) — sub-quadratic attention's whole efficiency argument
            // (Mistral 7B, arXiv:2310.06825) is that a query attends to this
            // many positions, not the full sequence. Every one of these
            // patterns used to collapse into plain dense attention on the
            // wire, so picking one changed nothing about the reported cost.
            let kv_span = layer
                .params
                .window_size
                .or(layer.params.block_size)
                .or(layer.params.dilation.map(|d| (seq / d.max(1)).max(1)))
                .unwrap_or(seq);

            // Use GQA formula if kv_heads < heads (Multi-Query or Grouped-Query Attention)
            let attn_flops = if kv_span < seq {
                attention::windowed_attention_flops(batch, seq, kv_span, hidden, heads, causal)
            } else if kv_heads < heads {
                attention::gqa_flops(batch, seq, hidden, heads, kv_heads, causal)
            } else {
                attention::attention_flops(batch, seq, hidden, heads, causal)
            };

            // QKV projections (scaled for GQA)
            let qkv_param_count = if kv_heads < heads {
                let kv_dim = kv_heads * head_dim;
                hidden * hidden + 2 * hidden * kv_dim + hidden * hidden // Q + K,V + O
            } else {
                4 * hidden * hidden // Standard: Q,K,V,O
            };

            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Attention,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq, hidden])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops: attn_flops,
                param_count: qkv_param_count as u64,
                activation_memory: ((batch * seq * hidden) as f64
                    * neurax_formulas::dtype_bytes(dtype))
                .round() as u64,
                is_custom: false,
            });

            // Add RoPE FLOPs for transformer models (applied to Q and K)
            let rope_flops = embedding::rope_flops(batch, seq, heads, head_dim);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Mul, // RoPE is element-wise rotation
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops: rope_flops,
                param_count: 0,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::Mlp => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let activation = layer.params.activation.as_deref().unwrap_or("gelu");

            let flops = if crate::architecture::is_gated_mlp(&layer.params) {
                mlp::gated_mlp_flops(batch, seq, hidden, intermediate, activation)
            } else {
                mlp::mlp_flops(batch, seq, hidden, intermediate, activation)
            };

            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq, hidden])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: ((batch * seq * intermediate) as f64
                    * neurax_formulas::dtype_bytes(dtype))
                .round() as u64,
                is_custom: false,
            });
        }
        LayerType::Dense => {
            let in_f = layer
                .params
                .in_features
                .or(layer.params.in_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let out_f = layer
                .params
                .out_features
                .or(layer.params.out_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let outer_dims = layer
                .input_shape
                .iter()
                .take(layer.input_shape.len().saturating_sub(1))
                .copied()
                .product::<usize>()
                .max(1);
            let output_elements = if layer.output_shape.is_empty() {
                outer_dims * out_f
            } else {
                layer.output_shape.iter().copied().product::<usize>()
            };

            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MatMul,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(layer.input_shape.clone())],
                output_shape: crate::tensor::Shape::known(layer.output_shape.clone()),
                flops: 2.0 * outer_dims as f64 * in_f as f64 * out_f as f64,
                param_count: layer.param_count,
                activation_memory: (output_elements as f64 * neurax_formulas::dtype_bytes(dtype))
                    .round() as u64,
                is_custom: false,
            });
        }
        LayerType::LoraLinear | LayerType::DoraLinear => {
            let in_f = layer.params.in_features.unwrap_or(512);
            let out_f = layer.params.out_features.unwrap_or(512);
            let rank = layer.params.rank.unwrap_or(16);
            let outer_dims = layer
                .input_shape
                .iter()
                .take(layer.input_shape.len().saturating_sub(1))
                .copied()
                .product::<usize>()
                .max(1);
            let flops = if matches!(layer.layer_type, LayerType::DoraLinear) {
                neurax_formulas::lora::dora_flops(outer_dims, in_f, out_f, rank)
            } else {
                neurax_formulas::lora::lora_flops(outer_dims, in_f, out_f, rank)
            };
            let output_elements = if layer.output_shape.is_empty() {
                outer_dims * out_f
            } else {
                layer.output_shape.iter().copied().product::<usize>()
            };
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MatMul,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(layer.input_shape.clone())],
                output_shape: crate::tensor::Shape::known(layer.output_shape.clone()),
                flops,
                param_count: layer.param_count,
                activation_memory: (output_elements as f64 * neurax_formulas::dtype_bytes(dtype))
                    .round() as u64,
                is_custom: false,
            });
        }
        LayerType::Normalization => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let is_rms = layer.params.activation.as_deref() == Some("rms");

            let (op_type, flops) = if is_rms {
                (
                    OpType::RMSNorm,
                    normalization::rms_norm_flops(batch, seq, hidden),
                )
            } else {
                (
                    OpType::LayerNorm,
                    normalization::layer_norm_flops(batch, seq, hidden),
                )
            };

            ops.push(AtomOp {
                id: ops.len(),
                op_type,
                layer_id: layer.id.clone(),
                input_shapes: vec![crate::tensor::Shape::known(vec![batch, seq, hidden])],
                output_shape: crate::tensor::Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: if is_rms {
                    hidden as u64
                } else {
                    2 * hidden as u64
                },
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::Conv => {
            // Conv2D FLOPs calculation
            let in_ch = layer
                .params
                .in_channels
                .or_else(|| ctx.config.data.image_channels)
                .unwrap_or(3);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kernel_h = layer.params.kernel_size.unwrap_or(3);
            let kernel_w = layer.params.kernel_w.unwrap_or(kernel_h);
            let stride = layer.params.stride.unwrap_or(1);
            let padding = layer.params.padding.unwrap_or(0);
            let groups = 1; // Standard convolution

            // Calculate output dimensions from input_shape or data config
            let (batch, in_h, in_w) = if layer.input_shape.len() >= 4 {
                (
                    layer.input_shape[0],
                    layer.input_shape[2],
                    layer.input_shape[3],
                )
            } else {
                // Use data config for image dimensions
                let h = ctx.config.data.image_height.unwrap_or(224);
                let w = ctx.config.data.image_width.unwrap_or(224);
                (ctx.config.training.batch_size, h, w)
            };

            let out_h = (in_h + 2 * padding - kernel_h + stride) / stride;
            let out_w = (in_w + 2 * padding - kernel_w + stride) / stride;

            let flops = super::formulas::conv2d_flops(
                batch, out_h, out_w, out_ch, kernel_h, kernel_w, in_ch, groups,
            );

            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::Pooling => {
            // Pooling FLOPs calculation
            let kernel_size = layer.params.kernel_size.unwrap_or(2);

            // Calculate output dimensions from input_shape
            let (batch, channels, in_h, in_w) = if layer.input_shape.len() >= 4 {
                (
                    layer.input_shape[0],
                    layer.input_shape[1],
                    layer.input_shape[2],
                    layer.input_shape[3],
                )
            } else {
                (1, 64, 224, 224)
            };

            let stride = layer.params.stride.unwrap_or(2);
            let out_h = in_h.div_ceil(stride);
            let out_w = in_w.div_ceil(stride);

            let flops = super::formulas::pooling_flops(batch, channels, out_h, out_w, kernel_size);

            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Pooling(super::PoolingType::Max),
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: 0,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::MoE => {
            // MoE: router + expert computation.
            //
            // `moe_flops`'s signature is `(batch, seq_len, hidden_size,
            // num_experts, top_k, expert_flops)` — this used to pass
            // `intermediate` where `num_experts` belongs, `num_experts`
            // where `top_k` belongs, and `top_k as f64` (2.0, 6.0, ...)
            // where `expert_flops` — the cost of one token through one
            // expert, normally in the billions — belongs. Every argument
            // after `hidden` was one position off; every one is `usize` or
            // `f64` like its neighbour, so it type-checked and returned a
            // number four to nine orders of magnitude too small without
            // ever failing to compile.
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let num_experts = layer.params.num_experts.unwrap_or(8);
            let top_k = layer.params.top_k.unwrap_or(2);
            let activation = layer.params.activation.as_deref().unwrap_or("silu");
            // One token through one gated-MLP expert — matches
            // `calculate_layer_params`'s `gated_mlp_params`, which is what
            // this expert's own parameter count assumes it is.
            let expert_flops = mlp::gated_mlp_flops(1, 1, hidden, intermediate, activation);
            let flops = moe::moe_flops(batch, seq, hidden, num_experts, top_k, expert_flops);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MoE,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // The router: a real but tiny cost (a `hidden → num_experts`
        // projection plus a softmax), not an expert-sized one.
        LayerType::MoeRouter => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let num_experts = layer.params.num_experts.unwrap_or(8);
            let flops = moe::moe_router_flops(batch, seq, hidden, num_experts);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MoE,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // Combining top-k experts' outputs by their routing weight: a
        // weighted sum over `top_k` tensors per token, real but small next
        // to routing an expert's full MLP.
        LayerType::MoeCombine => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let top_k = layer.params.top_k.unwrap_or(2);
            let flops = 2.0 * batch as f64 * seq as f64 * top_k as f64 * hidden as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MoE,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // A DeepSeek-style shared expert: always active for every token, no
        // routing/top-k involved — every token pays its full cost, not a
        // top_k-scaled fraction of it.
        LayerType::MoeSharedExpert => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let activation = layer.params.activation.as_deref().unwrap_or("silu");
            let shared_experts = layer.params.num_experts.unwrap_or(0) as f64;
            let per_expert_flops =
                mlp::gated_mlp_flops(batch, seq, hidden, intermediate, activation);
            let flops = shared_experts * per_expert_flops;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::MoE,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // ResNet's two real block shapes have real FLOPs formulas
        // (`neurax_formulas::cnn_blocks`) — used here instead of the generic
        // placeholder below. Nothing in this pass tracks a feature map's
        // actual height/width as it shrinks through the network (no CNN
        // layer type does), so `sqrt(seq)` stands in for a square spatial
        // size, on the same assumption the reference templates already make
        // (resnet50.json's `sequence_length: 196` is 14×14) — an
        // approximation, not the exact per-layer resolution, but a real
        // formula applied to it beats a formula with no structural relation
        // to the block at all.
        LayerType::ResidualBlock | LayerType::ResnetBottleneck => {
            let side = (seq as f64).sqrt().round().max(1.0) as usize;
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(256);
            let stride = layer.params.stride.unwrap_or(1);
            let flops = if matches!(layer.layer_type, LayerType::ResnetBottleneck) {
                let mid_ch = layer.params.mid_channels.unwrap_or(out_ch / 4);
                let cardinality = layer.params.cardinality.unwrap_or(1);
                cnn_blocks::resnet_bottleneck_block_flops(
                    batch,
                    in_ch,
                    mid_ch,
                    out_ch,
                    side,
                    side,
                    stride,
                    cardinality,
                )
            } else {
                cnn_blocks::resnet_basic_block_flops(batch, in_ch, out_ch, side, side, stride)
            };
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // The remaining CNN block types don't yet have their real FLOPs
        // formulas wired in here the way the two ResNet block shapes just
        // were, even though several of those formulas already exist in
        // `neurax_formulas::cnn_blocks` (`mbconv_flops` in particular) — this
        // placeholder is a known, separate gap, not a considered estimate.
        LayerType::Mbconv
        | LayerType::Inception
        | LayerType::DenseBlock
        | LayerType::ConvnextBlock
        | LayerType::ShuffleUnit
        | LayerType::C2f
        | LayerType::Detection
        | LayerType::Transition => {
            // Treat as conv-like operations
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let flops = (batch * seq * hidden * hidden) as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // S4/H3/StateSpace are non-selective: A/B/C are plain learned
        // weights, not the output of Mamba's own input-dependent
        // projections — `mamba_flops`'s `in_proj` term prices exactly that
        // selectivity mechanism, which these three don't have. `s4_flops`/
        // `h3_flops` already existed with the right shape (S4's FFT-based
        // O(S log S) convolution instead of Mamba's O(S) selective scan,
        // H3's two-SSM-layer structure) but were never called from
        // anywhere in the live pipeline.
        LayerType::S4Block | LayerType::StateSpace => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let flops = neurax_formulas::ssm::s4_flops(batch, seq, hidden, state_dim);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::H3Block => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let flops = neurax_formulas::ssm::h3_flops(batch, seq, hidden, state_dim);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // State Space Model layer types
        LayerType::MambaBlock | LayerType::RwkvBlock | LayerType::RetentionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let expand = layer.params.expansion_factor.unwrap_or(2);

            // Use proper Mamba FLOPs formula from neurax_formulas
            let flops = neurax_formulas::ssm::mamba_flops(batch, seq, hidden, state_dim, expand);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // GAN layer types — real formulas per block, matching the shape each
        // already uses on the params side (`architecture/mod.rs`). These used to
        // share one placeholder (`batch*seq*hidden^2`) across all seven kinds —
        // costing a per-channel affine (StyleMod) the same as a full convolution,
        // and a self-attention block by a formula with no relation to attention
        // at all.
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock | LayerType::ProgressiveBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer.params.kernel_size.unwrap_or(3);
            let stride = layer.params.stride.unwrap_or(1);
            let padding = layer.params.padding.unwrap_or(0);
            // No per-layer spatial (H, W) tracking exists for GAN blocks — same
            // sqrt(seq) stand-in for a square feature-map side that
            // ResidualBlock/ResnetBottleneck already use for the same reason.
            let side = (seq as f64).sqrt().round().max(1.0) as usize;
            let flops =
                conv::conv2d_flops(batch, in_ch, out_ch, side, side, kh, kh, stride, padding, 1);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::SelfAttention => {
            let channels = layer.params.out_channels.unwrap_or(512);
            // Same head count `SelfAttention`'s own param formula assumes.
            let heads = (channels / 64).max(1);
            let flops = attention::attention_flops(batch, seq, channels, heads, false);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Attention,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, channels])],
                output_shape: Shape::known(vec![batch, seq, channels]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::StyleMod => {
            // A per-channel scale and bias — exactly the two elementwise ops its
            // own param formula (`channels * 2`) counts, not a convolution's cost.
            let channels = layer.params.out_channels.unwrap_or(512);
            let flops = 2.0 * batch as f64 * seq as f64 * channels as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, channels])],
                output_shape: Shape::known(vec![batch, seq, channels]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::AdaIN
        | LayerType::PixelNorm
        | LayerType::MinibatchStd
        | LayerType::SpectralNorm => {
            // Normalization-weight cost (~RMSNorm's own 3-ops-per-element), not a
            // full layer's worth of compute — none of these four carry a weight
            // matrix of their own.
            let channels = layer.params.out_channels.unwrap_or(512);
            let flops = 3.0 * batch as f64 * seq as f64 * channels as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, channels])],
                output_shape: Shape::known(vec![batch, seq, channels]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // LSTM/RNN layer types
        LayerType::LstmBlock
        | LayerType::GruBlock
        | LayerType::RnnCell
        | LayerType::Bidirectional
        | LayerType::EncoderBlock
        | LayerType::DecoderBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            // LSTM: 4 gates, GRU: 3 gates
            let gates = if matches!(layer.layer_type, LayerType::GruBlock) {
                3
            } else {
                4
            };
            let flops = (batch * seq * hidden * hidden * gates) as f64;
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // Diffusion layer types — real formulas per block, matching the shape
        // each already uses on the params side. These used to share one
        // placeholder (`batch*seq*hidden^2*4`) across a U-Net ResNet block, a
        // cross-attention block and a time-embedding MLP alike — three
        // structurally different operations costed identically.
        LayerType::UnetBlock
        | LayerType::ResnetBlock
        | LayerType::DownBlock
        | LayerType::UpBlock
        | LayerType::MidBlock => {
            let in_ch = layer
                .params
                .in_channels_diff
                .unwrap_or(layer.params.in_channels.unwrap_or(320));
            let out_ch = layer
                .params
                .out_channels_diff
                .unwrap_or(layer.params.out_channels.unwrap_or(320));
            // Same sqrt(seq) spatial-side stand-in the CNN ResidualBlock arm
            // above uses — no per-layer (H, W) tracking exists for diffusion
            // U-Net blocks either.
            let side = (seq as f64).sqrt().round().max(1.0) as usize;
            let flops = cnn_blocks::resnet_basic_block_flops(batch, in_ch, out_ch, side, side, 1);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::TimeEmbedding | LayerType::TimestepBlock => {
            // Linear + SiLU + Linear — exactly what this type's own param
            // formula (`mlp_params(channels, channels*4, true)`) already assumes.
            let channels = layer.params.hidden_size.unwrap_or(320);
            let flops = mlp::mlp_flops(batch, seq, channels, channels * 4, "silu");
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, channels])],
                output_shape: Shape::known(vec![batch, seq, channels]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::ConditionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            let flops = mlp::mlp_flops(batch, seq, hidden, hidden * 4, "gelu");
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::CrossAttention => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            let heads = layer.params.num_heads.unwrap_or(8);
            // Q from the image features, K/V from the conditioning sequence —
            // approximated here as self-attention over the image sequence
            // (K/V's real, usually much shorter, conditioning length isn't
            // tracked as a separate dimension anywhere in this pass yet).
            let flops = attention::attention_flops(batch, seq, hidden, heads, false);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Attention,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::NoisePredictor => {
            // Final conv to predict noise — same kernel=3, groups=1 shape as
            // this type's own param formula (`conv2d_params(channels, channels,
            // 3, 3, 1, false)`), with padding=1 to preserve spatial size.
            let channels = layer.params.out_channels_diff.unwrap_or(4);
            let side = (seq as f64).sqrt().round().max(1.0) as usize;
            let flops = conv::conv2d_flops(batch, channels, channels, side, side, 3, 3, 1, 1, 1);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::VaeEncoder | LayerType::VaeDecoder => {
            let in_ch = layer.params.in_channels.unwrap_or(3);
            let out_ch = layer.params.out_channels_diff.unwrap_or(4);
            let side = (seq as f64).sqrt().round().max(1.0) as usize;
            let flops = conv::conv2d_flops(batch, in_ch, out_ch, side, side, 3, 3, 1, 1, 1);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Conv2D,
                layer_id: layer.id.clone(),
                input_shapes: vec![],
                output_shape: crate::tensor::Shape::default(),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // Graph Neural Networks — real FLOPs (`neurax-formulas::gnn`), reading
        // the graph size from `global_params.extra` (`num_nodes`/`num_edges`),
        // the same map `GlobalResolutionContext` reads for the params side.
        // Falls back to the Cora citation-graph benchmark's real size — the
        // same default this analyser's own hyperparameter panel already
        // assumes for a GNN design with no explicit graph size — rather than
        // an arbitrary round number.
        LayerType::GraphConvNet | LayerType::MessagePassing => {
            let in_features = layer.params.in_features.unwrap_or(64);
            let out_features = layer.params.out_features.unwrap_or(64);
            let num_nodes = extra_usize(&ctx.config.model.global_params.extra, "num_nodes", 2708);
            let num_edges = extra_usize(&ctx.config.model.global_params.extra, "num_edges", 10556);
            let flops = gnn::gcn_flops(num_nodes, in_features, out_features, num_edges);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![num_nodes, in_features])],
                output_shape: Shape::known(vec![num_nodes, out_features]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::GraphAttentionNet => {
            let in_features = layer.params.in_features.unwrap_or(64);
            let out_features = layer.params.out_features.unwrap_or(64);
            let num_heads = layer.params.num_heads.unwrap_or(8);
            let num_nodes = extra_usize(&ctx.config.model.global_params.extra, "num_nodes", 2708);
            let num_edges = extra_usize(&ctx.config.model.global_params.extra, "num_edges", 10556);
            let flops = gnn::gat_flops(num_nodes, in_features, out_features, num_edges, num_heads);
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![num_nodes, in_features])],
                output_shape: Shape::known(vec![num_nodes, out_features]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        LayerType::RgcnConv => {
            let in_features = layer.params.in_features.unwrap_or(64);
            let out_features = layer.params.out_features.unwrap_or(64);
            let num_relations = layer.params.num_relations.unwrap_or(1);
            let num_nodes = extra_usize(&ctx.config.model.global_params.extra, "num_nodes", 2708);
            let num_edges = extra_usize(&ctx.config.model.global_params.extra, "num_edges", 10556);
            let flops = gnn::rgcn_flops(
                num_nodes,
                in_features,
                out_features,
                num_edges,
                num_relations,
            );
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Linear,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![num_nodes, in_features])],
                output_shape: Shape::known(vec![num_nodes, out_features]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: false,
            });
        }
        // Custom layer - use custom equations if provided
        LayerType::Custom => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            // Use custom equations or default FLOPs
            // Evaluate the author's FLOP equation.
            //
            // This used to `parse::<f64>()` the equation string, which succeeds
            // only for a bare literal: any real formula such as "2 * B * S * H"
            // failed to parse and the block was silently costed at zero FLOPs.
            let equation = layer
                .custom_equations
                .as_ref()
                .and_then(|eqs| eqs.flops_forward.as_ref());
            let flops = match equation {
                Some(equation) => {
                    let evaluated = crate::architecture::evaluate_custom_equation_with(
                        equation,
                        &to_parser_layer(layer),
                        batch,
                        seq,
                    );
                    if evaluated.is_none() {
                        // Report the broken formula. Otherwise the block costs
                        // nothing and the failure surfaces much later as an
                        // unexplained "zero FLOPs" error.
                        ctx.add_diagnostic(crate::Diagnostic {
                            severity: crate::Severity::Critical,
                            category: crate::DiagnosticCategory::CustomLayerFallback,
                            code: crate::DiagnosticCode::E003,
                            message: format!(
                                "Custom FLOP equation for layer '{}' could not be evaluated: `{}`",
                                layer.id, equation
                            ),
                            layer_id: Some(layer.id.clone()),
                            suggestion: Some(
                                "Use the documented variables (B, S, H, D, I, V, N) and operators \
                                 supported by the expression evaluator."
                                    .to_string(),
                            ),
                            precision_impact: 1.0,
                        });
                    }
                    evaluated.unwrap_or(0.0)
                }
                None => {
                    ctx.add_diagnostic(crate::Diagnostic {
                        severity: crate::Severity::Warning,
                        category: crate::DiagnosticCategory::CustomLayerFallback,
                        code: crate::DiagnosticCode::W001,
                        message: format!(
                            "Custom layer '{}' has no FLOP equation; it is costed as free.",
                            layer.id
                        ),
                        layer_id: Some(layer.id.clone()),
                        suggestion: Some(
                            "Add `custom_equations.flops_forward` so the block is accounted for."
                                .to_string(),
                        ),
                        precision_impact: 0.5,
                    });
                    0.0
                }
            };
            ops.push(AtomOp {
                id: ops.len(),
                op_type: OpType::Custom,
                layer_id: layer.id.clone(),
                input_shapes: vec![Shape::known(vec![batch, seq, hidden])],
                output_shape: Shape::known(vec![batch, seq, hidden]),
                flops,
                param_count: layer.param_count,
                activation_memory: 0,
                is_custom: true,
            });
        }
    }

    ops
}

/// Total FLOPs for one layer — the sum of every atomic operation
/// `decompose_layer_to_ops` breaks it into.
///
/// Exposed publicly so other consumers of the real per-layer-type formulas
/// this pass already applies don't have to re-derive their own, separate
/// approximation. The MLIR backend used to: its own `compiler.rs` counted
/// attention as `4 × hidden²` FLOPs and left every CNN, SSM, GAN, RNN and
/// diffusion layer type silently uncounted, simply because nothing connected
/// it to this pass.
pub fn layer_flops(
    layer: &crate::architecture::LayerDef,
    batch: usize,
    seq: usize,
    dtype: &str,
    ctx: &NeuraxContext,
) -> f64 {
    decompose_layer_to_ops(layer, batch, seq, dtype, ctx)
        .iter()
        .map(|op| op.flops)
        .sum()
}

/// Adapt an IR layer back to the parser type the shared helpers expect.
fn to_parser_layer(layer: &crate::architecture::LayerDef) -> neurax_parser::Layer {
    neurax_parser::Layer {
        id: layer.id.clone(),
        layer_type: layer.layer_type,
        input_shape: layer.input_shape.clone(),
        output_shape: layer.output_shape.clone(),
        params: layer.params.clone(),
        custom_equations: layer.custom_equations.clone(),
    }
}
