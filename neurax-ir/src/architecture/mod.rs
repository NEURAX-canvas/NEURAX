//! Architecture IR - Premier dialecte du pipeline NEURAX

mod ir;
mod metrics;
mod pass;

pub use ir::*;
pub use metrics::*;
pub use pass::*;

use neurax_formulas::*;
use neurax_parser::{Layer, LayerType};

/// Calculate parameters for a single layer
pub fn calculate_layer_params(layer: &Layer) -> u64 {
    match layer.layer_type {
        LayerType::Embedding => {
            let vocab = layer.params.vocab_size.unwrap_or(50000);
            let dim = layer
                .params
                .embedding_dim
                .unwrap_or(layer.params.hidden_size.unwrap_or(512));
            embedding::embedding_params(vocab, dim)
        }
        LayerType::Attention => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let heads = layer.params.num_heads.unwrap_or(8);
            let kv_heads = layer.params.num_kv_heads.unwrap_or(heads);
            if kv_heads == heads {
                attention::attention_params(hidden, heads, layer.params.bias)
            } else {
                attention::gqa_params(hidden, heads, kv_heads, layer.params.bias)
            }
        }
        LayerType::Mlp => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            if is_gated_mlp(&layer.params) {
                mlp::gated_mlp_params(hidden, intermediate, layer.params.bias)
            } else {
                mlp::mlp_params(hidden, intermediate, layer.params.bias)
            }
        }
        LayerType::Dense => {
            let in_features = layer
                .params
                .in_features
                .or(layer.params.in_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let out_features = layer
                .params
                .out_features
                .or(layer.params.out_channels)
                .or(layer.params.hidden_size)
                .unwrap_or(512);
            let bias = if layer.params.bias { out_features } else { 0 };
            (in_features * out_features + bias) as u64
        }
        LayerType::LoraLinear | LayerType::DoraLinear => {
            let in_features = layer.params.in_features.unwrap_or(512);
            let out_features = layer.params.out_features.unwrap_or(512);
            // The rank is the whole point of LoRA — defaulting it to
            // something in the tens (not `in_features`, which would just
            // reproduce a full dense layer's cost) when unset.
            let rank = layer.params.rank.unwrap_or(16);
            if layer.layer_type == LayerType::DoraLinear {
                lora::dora_params(in_features, out_features, rank)
            } else {
                lora::lora_params(in_features, out_features, rank)
            }
        }
        LayerType::Conv => {
            let in_ch = layer.params.in_channels.unwrap_or(3);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer
                .params
                .kernel_h
                .unwrap_or(layer.params.kernel_size.unwrap_or(3));
            let kw = layer
                .params
                .kernel_w
                .unwrap_or(layer.params.kernel_size.unwrap_or(3));
            let groups = layer.params.groups.unwrap_or(1);
            conv::conv2d_params(in_ch, out_ch, kh, kw, groups, layer.params.bias)
        }
        LayerType::Normalization => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            // Check if it's RMSNorm or LayerNorm
            if layer.params.activation.as_deref() == Some("rms") {
                normalization::rms_norm_params(hidden)
            } else {
                normalization::layer_norm_params(hidden, true)
            }
        }
        LayerType::Pooling => 0,
        LayerType::MoE => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let num_experts = layer.params.num_experts.unwrap_or(8);
            // No shared expert unless the config states one. Defaulting to 1
            // added an expert to every mixture that did not mention them —
            // Mixtral has none, and it was being counted with nine.
            let shared_experts = layer.params.shared_experts.unwrap_or(0);
            // Each expert is a gated MLP: gate, up and down projections.
            let expert_params = mlp::gated_mlp_params(hidden, intermediate, layer.params.bias);
            // Router + all experts + shared experts. A block diagram that
            // gives the router its own `MoeRouter` node (see below) pays for
            // that router's `hidden × num_experts` gating matrix twice —
            // once here, once there — which is a few thousand parameters
            // against a real model's billions, negligible enough not to be
            // worth threading graph context into a per-node function for.
            moe::moe_params_with_shared(
                hidden,
                intermediate,
                num_experts,
                shared_experts,
                expert_params,
            )
        }
        // The router's own weights: a `hidden × num_experts` gating matrix,
        // nothing else — no expert-sized cost belongs here. Folding this
        // into `MoE`'s formula (treating the router as if it were
        // `num_experts` full experts) is the bug this variant exists to fix.
        LayerType::MoeRouter => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let num_experts = layer.params.num_experts.unwrap_or(8);
            moe::moe_router_params(hidden, num_experts)
        }
        // Combining the top-k experts' outputs by their routing weight is a
        // weighted sum — no weights of its own to store.
        LayerType::MoeCombine => 0,
        // A DeepSeek-style shared expert: always active, never routed. Read
        // off this node's own `num_experts` (how many shared experts, the
        // name the importer and the reference templates give that count on
        // this specific node type — not the `shared_experts` field, which
        // belongs to a self-contained single-node `MoE` layer instead).
        // Passed through `moe_params_with_shared` with zero routed experts
        // so it contributes no router term of its own — the router is
        // whatever `MoeRouter` node sits beside it in the same layer.
        LayerType::MoeSharedExpert => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
            let shared_experts = layer.params.num_experts.unwrap_or(0);
            let expert_params = mlp::gated_mlp_params(hidden, intermediate, layer.params.bias);
            moe::moe_params_with_shared(hidden, intermediate, 0, shared_experts, expert_params)
        }
        // CNN layer types - use dedicated formulas
        LayerType::ResidualBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let stride = layer.params.stride.unwrap_or(1);
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, stride, layer.params.bias)
        }
        LayerType::ResnetBottleneck => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            // The classic ResNet-50 expansion factor: the bottleneck's
            // middle width is a quarter of its output width, unless the
            // config says otherwise.
            let out_ch = layer.params.out_channels.unwrap_or(256);
            let mid_ch = layer.params.mid_channels.unwrap_or(out_ch / 4);
            let stride = layer.params.stride.unwrap_or(1);
            // ResNeXt's defining difference from a plain ResNet bottleneck is
            // exactly this: the middle 3x3 conv split into `cardinality`
            // groups. Declared in the schema (`LayerParams::cardinality`,
            // "ResNeXt groups") but never read anywhere until now — a real
            // ResNeXt config's grouping had zero effect on its parameter count.
            let cardinality = layer.params.cardinality.unwrap_or(1);
            cnn_blocks::resnet_bottleneck_block_params(
                in_ch,
                mid_ch,
                out_ch,
                stride,
                cardinality,
                layer.params.bias,
            )
        }
        LayerType::Mbconv => {
            let in_ch = layer.params.in_channels.unwrap_or(32);
            let out_ch = layer.params.out_channels.unwrap_or(16);
            let expand = layer.params.expansion_factor.unwrap_or(6);
            let kernel = layer.params.kernel_size.unwrap_or(3);
            let stride = layer.params.stride.unwrap_or(1);
            let se_reduction = layer.params.se_reduction_ratio.unwrap_or(4);
            cnn_blocks::mbconv_params(
                in_ch,
                out_ch,
                expand,
                kernel,
                stride,
                layer.params.se,
                se_reduction,
                layer.params.bias,
            )
        }
        LayerType::Inception => {
            let in_ch = layer.params.in_channels.unwrap_or(288);
            let out_1x1 = layer.params.out_channels.unwrap_or(64);
            cnn_blocks::inception_module_params(
                in_ch,
                out_1x1,
                out_1x1 / 2,
                out_1x1, // 3x3 branch
                out_1x1 / 8,
                out_1x1 / 2, // 5x5 branch
                out_1x1,     // pool branch
                layer.params.bias,
            )
        }
        LayerType::DenseBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let growth = layer.params.growth_rate.unwrap_or(32);
            let num_layers = layer.params.num_layers.unwrap_or(4);
            cnn_blocks::dense_block_params(in_ch, growth, num_layers, 4, layer.params.bias)
        }
        LayerType::ConvnextBlock => {
            let channels = layer.params.hidden_size.unwrap_or(96);
            let mlp_ratio = layer.params.mlp_ratio.unwrap_or(4.0);
            cnn_blocks::convnext_block_params(channels, mlp_ratio, layer.params.bias)
        }
        LayerType::ShuffleUnit => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let groups = layer.params.groups.unwrap_or(2);
            let stride = layer.params.stride.unwrap_or(1);
            cnn_blocks::shuffle_unit_params(in_ch, out_ch, groups, stride, layer.params.bias)
        }
        LayerType::C2f => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let num_bn = layer.params.num_bottlenecks.unwrap_or(3);
            cnn_blocks::c2f_block_params(in_ch, out_ch, num_bn, true, layer.params.bias)
        }
        LayerType::Detection | LayerType::Transition => {
            // Detection heads and transition layers are typically simple convs
            let in_ch = layer.params.in_channels.unwrap_or(256);
            let out_ch = layer.params.out_channels.unwrap_or(256);
            let kernel = layer.params.kernel_size.unwrap_or(3);
            conv::conv2d_params(in_ch, out_ch, kernel, kernel, 1, layer.params.bias)
        }
        // State Space Model layer types - use dedicated formulas
        LayerType::MambaBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let expansion = layer.params.expansion_factor.unwrap_or(2);
            ssm::mamba_params(hidden, state_dim, expansion)
        }
        // Parameter count reuses Mamba's formula — a known approximation,
        // not a considered exact one: S4/H3 lack Mamba's selective
        // (input-dependent) B/C/Δ projections, so the real weight count
        // differs in composition, though it stays the same order of
        // magnitude (dominated by the same d_inner × state_dim state
        // matrices either way). The FLOPs side does have real, distinct
        // formulas (`s4_flops`/`h3_flops`, see operator/pass.rs) — the
        // params side does not yet.
        LayerType::S4Block | LayerType::H3Block | LayerType::StateSpace => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            let state_dim = layer.params.state_dim.unwrap_or(16);
            let expansion = layer.params.expansion_factor.unwrap_or(2);
            ssm::mamba_params(hidden, state_dim, expansion)
        }
        // RWKV and RetNet are not state-space models. Costing them with
        // Mamba's formula put RWKV-7B at 3.4B against a published 7.5B.
        LayerType::RwkvBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            ssm::rwkv_params(hidden, layer.params.intermediate_size)
        }
        LayerType::RetentionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(512);
            ssm::retention_params(hidden, layer.params.intermediate_size)
        }
        // GAN layer types
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer.params.kernel_size.unwrap_or(3);
            conv::conv2d_params(in_ch, out_ch, kh, kh, 1, layer.params.bias)
        }
        LayerType::StyleMod => {
            // Style modulation: affine transform per channel
            let channels = layer.params.out_channels.unwrap_or(512);
            (channels * 2) as u64 // scale + bias per channel
        }
        LayerType::AdaIN => {
            // Adaptive Instance Norm: no params, uses style input
            0
        }
        LayerType::MinibatchStd => 0, // No params
        LayerType::PixelNorm => 0,    // No params
        LayerType::SelfAttention => {
            let channels = layer.params.out_channels.unwrap_or(512);
            attention::attention_params(channels, channels / 64, false)
        }
        LayerType::SpectralNorm => {
            // Spectral norm adds one vector per weight matrix
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            (in_ch * out_ch / out_ch) as u64 // u vector
        }
        LayerType::ProgressiveBlock => {
            let in_ch = layer.params.in_channels.unwrap_or(64);
            let out_ch = layer.params.out_channels.unwrap_or(64);
            let kh = layer.params.kernel_size.unwrap_or(3);
            conv::conv2d_params(in_ch, out_ch, kh, kh, 1, layer.params.bias)
        }
        // LSTM/RNN layer types - use dedicated formulas
        LayerType::LstmBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
            rnn::lstm_params(hidden, input_size, true) * bidir_mult
        }
        LayerType::GruBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            let bidir_mult = if layer.params.bidirectional_rnn { 2 } else { 1 };
            rnn::gru_params(hidden, input_size, true) * bidir_mult
        }
        LayerType::RnnCell => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            rnn::rnn_params(hidden, input_size, true)
        }
        LayerType::Bidirectional => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            rnn::lstm_params(hidden, input_size, true) * 2
        }
        LayerType::EncoderBlock | LayerType::DecoderBlock => {
            let hidden = layer.params.rnn_hidden_size.unwrap_or(512);
            let input_size = layer.params.hidden_size.unwrap_or(hidden);
            rnn::lstm_params(hidden, input_size, true)
        }
        // Diffusion layer types - use dedicated formulas
        LayerType::UnetBlock | LayerType::ResnetBlock => {
            let in_ch = layer
                .params
                .in_channels_diff
                .unwrap_or(layer.params.in_channels.unwrap_or(320));
            let out_ch = layer
                .params
                .out_channels_diff
                .unwrap_or(layer.params.out_channels.unwrap_or(320));
            // UNet ResNet block: 2 convs + 2 norms + skip
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, 1, layer.params.bias)
        }
        LayerType::TimeEmbedding | LayerType::TimestepBlock => {
            let channels = layer.params.hidden_size.unwrap_or(320);
            // Time embedding: Linear + SiLU + Linear
            mlp::mlp_params(channels, channels * 4, true)
        }
        LayerType::CrossAttention => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            let heads = layer.params.num_heads.unwrap_or(8);
            // Cross attention: Q from hidden, K,V from conditioning
            attention::attention_params(hidden, heads, true)
        }
        LayerType::DownBlock | LayerType::UpBlock | LayerType::MidBlock => {
            let in_ch = layer
                .params
                .in_channels_diff
                .unwrap_or(layer.params.in_channels.unwrap_or(320));
            let out_ch = layer
                .params
                .out_channels_diff
                .unwrap_or(layer.params.out_channels.unwrap_or(320));
            cnn_blocks::resnet_basic_block_params(in_ch, out_ch, 1, layer.params.bias)
        }
        LayerType::ConditionBlock => {
            let hidden = layer.params.hidden_size.unwrap_or(320);
            mlp::mlp_params(hidden, hidden * 4, true)
        }
        LayerType::NoisePredictor => {
            let channels = layer.params.out_channels_diff.unwrap_or(4);
            // Final conv to predict noise
            conv::conv2d_params(channels, channels, 3, 3, 1, false)
        }
        LayerType::VaeEncoder | LayerType::VaeDecoder => {
            let in_ch = layer.params.in_channels.unwrap_or(3);
            let out_ch = layer.params.out_channels_diff.unwrap_or(4);
            // VAE encoder/decoder: multiple convs (simplified)
            conv::conv2d_params(in_ch, out_ch, 3, 3, 1, false)
        }
        // Graph Neural Networks — real formulas (`neurax-formulas::gnn`),
        // wired here for the first time: these three used to fall through to
        // `Custom` with no `param_count` supplied, costing 0 regardless of
        // the design. `MessagePassing` reuses the GCN linear-transform shape
        // rather than a dedicated formula — an approximation for
        // GraphSAGE/GIN's own aggregators, but a real, non-zero number
        // either way. RGCN gets its own arm below: a shared single matrix
        // understates it badly once there's more than a couple of relation
        // types (real knowledge graphs: hundreds).
        LayerType::GraphConvNet | LayerType::MessagePassing => {
            let in_features = layer.params.in_features.unwrap_or(64);
            let out_features = layer.params.out_features.unwrap_or(64);
            gnn::gcn_params(in_features, out_features, layer.params.bias)
        }
        LayerType::GraphAttentionNet => {
            let in_features = layer.params.in_features.unwrap_or(64);
            let out_features = layer.params.out_features.unwrap_or(64);
            let num_heads = layer.params.num_heads.unwrap_or(8);
            gnn::gat_params(in_features, out_features, num_heads, layer.params.bias)
        }
        LayerType::RgcnConv => {
            let in_features = layer.params.in_features.unwrap_or(64);
            let out_features = layer.params.out_features.unwrap_or(64);
            let num_relations = layer.params.num_relations.unwrap_or(1);
            gnn::rgcn_params(
                in_features,
                out_features,
                num_relations,
                layer.params.num_bases,
                layer.params.bias,
            )
        }
        // Custom layer - use param_count if provided, else estimate from shapes
        LayerType::Custom => {
            // An explicit count wins; otherwise evaluate a `params` equation if
            // the author supplied one. A custom block that yields no parameters
            // is legitimate (a normalisation or routing step), but it must not
            // be the silent outcome of an equation nobody evaluated.
            layer.params.param_count.unwrap_or_else(|| {
                layer
                    .custom_equations
                    .as_ref()
                    .and_then(|eqs| {
                        eqs.extra
                            .get("params")
                            .or_else(|| eqs.extra.get("parameters"))
                    })
                    .and_then(|equation| evaluate_custom_equation(equation, layer))
                    .map(|value| value.max(0.0) as u64)
                    .unwrap_or(0)
            })
        }
    }
}

/// Number of times a listed layer stands in for real layers of the model.
///
/// Configs commonly describe a deep model by listing one representative block
/// per kind and setting `global_params.num_layers` to the real depth. Both the
/// architecture and memory passes need the same multiplier; keeping it here
/// stops the two from disagreeing, which previously let the headline parameter
/// count come out roughly 20x below the memory pass's figure for the same model.
pub fn repeat_scale_for(config: &neurax_parser::ModelConfig, layer: &Layer) -> f64 {
    let layers = &config.model.layers;

    // Scaling only applies when the author states a depth the JSON does not
    // spell out. Defaulting the depth to the number of listed layers made a
    // fully-listed architecture look partial: a five-layer model with one
    // attention block was read as five attention blocks, inflating it fivefold.
    let Some(global_num_layers) = config.model.global_params.num_layers.map(|n| n as usize) else {
        return 1.0;
    };
    let num_dense_layers = config.model.global_params.num_dense_layers.unwrap_or(0) as usize;

    let count_of = |pred: &dyn Fn(&Layer) -> bool| layers.iter().filter(|l| pred(l)).count();
    let json_moe_count = count_of(&|l: &Layer| l.layer_type == LayerType::MoE);
    let json_attention_count = count_of(&|l: &Layer| l.layer_type == LayerType::Attention);
    let json_ssm_count = count_of(&|l: &Layer| {
        matches!(
            l.layer_type,
            LayerType::MambaBlock | LayerType::S4Block | LayerType::H3Block | LayerType::StateSpace
        )
    });

    // Only blocks that repeat through the depth of the model scale up; the
    // embedding, the final norm and the head appear exactly once.
    let is_repeatable = matches!(
        layer.layer_type,
        LayerType::Attention
            | LayerType::Mlp
            | LayerType::Normalization
            | LayerType::MoE
            | LayerType::MoeRouter
            | LayerType::MoeCombine
            | LayerType::MoeSharedExpert
            | LayerType::MambaBlock
            | LayerType::S4Block
            | LayerType::H3Block
            | LayerType::StateSpace
            | LayerType::RwkvBlock
            | LayerType::RetentionBlock
            // Recurrent stacks state their depth the same way a transformer
            // does. Leaving them out meant a config saying "3-layer LSTM,
            // num_layers: 3" was read as a single layer.
            | LayerType::LstmBlock
            | LayerType::GruBlock
    );
    if !is_repeatable {
        return 1.0;
    }

    // MoE models such as DeepSeek split their depth between early dense
    // layers and later expert layers — only the layer's *feed-forward*
    // sublayer changes between the two; attention, normalization and every
    // other repeatable kind still runs in every layer regardless of which
    // FFN a given layer uses. This block therefore only special-cases the
    // FFN-deciding kinds and falls through to the general per-kind scaling
    // below for everything else — an earlier version returned early for
    // every repeatable kind here, which would have scaled attention to only
    // `num_dense_layers` layers instead of the model's real depth the first
    // time this branch ran on a real design.
    if num_dense_layers > 0 && json_moe_count > 0 {
        let is_moe_role = matches!(
            layer.layer_type,
            LayerType::MoE
                | LayerType::MoeRouter
                | LayerType::MoeCombine
                | LayerType::MoeSharedExpert
        );
        if is_moe_role {
            let num_moe_layers = global_num_layers.saturating_sub(num_dense_layers);
            return num_moe_layers as f64 / json_moe_count.max(1) as f64;
        }
        if layer.layer_type == LayerType::Mlp {
            // The plain feed-forward used only in the first `num_dense_layers`
            // layers, before the model switches to routed experts.
            let dense_blocks = count_of(&|l: &Layer| l.layer_type == LayerType::Mlp).max(1);
            return num_dense_layers as f64 / dense_blocks as f64;
        }
        // Attention, normalization, ...: fall through to the general
        // per-kind scaling below, which already scales them to the model's
        // full depth.
    }

    // Each kind scales against how many of *its own* kind were listed, not
    // against the attention count.
    //
    // The previous version derived one scale from the attention blocks and
    // applied it to everything repeatable. That is correct only when every
    // kind is listed the same number of times. Mixtral lists 2 attention
    // blocks and 4 MoE blocks for a 32-layer model: the shared scale of 32/2
    // gave 32 attention layers — right — and 64 MoE layers, doubling a model
    // that was already the larger half of its parameters. Measured against the
    // published figure it came out at 103.8B instead of 46.7B, and DeepSeek-V3
    // at 1.39T instead of 671B.
    let listed_of_this_kind = match layer.layer_type {
        LayerType::MoE => json_moe_count,
        LayerType::Attention => json_attention_count,
        LayerType::MambaBlock | LayerType::S4Block | LayerType::H3Block | LayerType::StateSpace => {
            json_ssm_count
        }
        other => count_of(&|l: &Layer| l.layer_type == other),
    };

    // A kind listed once in a model whose depth is stated separately is a
    // representative block; a kind listed as many times as the depth is already
    // spelled out and must not be multiplied again.
    let listed = listed_of_this_kind.max(1);
    if global_num_layers > listed {
        global_num_layers as f64 / listed as f64
    } else {
        1.0
    }
}

/// Total parameters of the model the config describes, scaling the listed
/// blocks up to `global_params.num_layers`.
pub fn scaled_total_parameters(config: &neurax_parser::ModelConfig) -> u64 {
    config
        .model
        .layers
        .iter()
        .map(|layer| {
            let raw = calculate_layer_params(layer);
            let scale = repeat_scale_for(config, layer);
            (raw as f64 * scale).round() as u64
        })
        .sum()
}

/// Parameters actually touched per token, rather than parameters the model
/// owns.
///
/// For a dense model the two are the same number — every weight runs on
/// every token. For a mixture-of-experts model they are not: Mixtral-8x7B
/// owns 46.7B parameters but only 12.9B run on any given token, because a
/// router sends that token to 2 of its 8 experts, not all of them. "46.7B"
/// and "13B active" answer different real questions — the first is what has
/// to fit in memory, the second is closer to what a dense model of
/// comparable *speed* would need. Reporting only the first, as NEURAX did
/// before this, understates what an MoE model actually costs to run and
/// overstates it against a dense model of the same total size.
pub fn scaled_active_parameters(config: &neurax_parser::ModelConfig) -> u64 {
    config
        .model
        .layers
        .iter()
        .map(|layer| {
            let scale = repeat_scale_for(config, layer);
            let active = match layer.layer_type {
                // Only the routed experts a token is actually sent to are
                // active for that token — the router picks `top_k` out of
                // `num_experts`. The router's own gating matrix and any
                // shared expert run on every token regardless, so they are
                // computed here directly rather than by scaling the whole
                // `raw` total (from `calculate_layer_params`) by a single
                // fraction — that used to scale the router and shared-expert
                // weight down along with the routed experts too, for exactly
                // the single fused `MoE` node this arm handles (the split
                // `MoeRouter`/`MoeSharedExpert` node case was already correct,
                // since those fall into the `_ => raw` branch below at 1.0).
                LayerType::MoE => {
                    let hidden = layer.params.hidden_size.unwrap_or(512);
                    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
                    let num_experts = layer.params.num_experts.unwrap_or(8);
                    let top_k = layer.params.top_k.unwrap_or(2).min(num_experts);
                    let shared_experts = layer.params.shared_experts.unwrap_or(0);
                    let expert_params =
                        mlp::gated_mlp_params(hidden, intermediate, layer.params.bias);
                    moe::moe_active_params_with_shared(
                        hidden,
                        num_experts,
                        top_k,
                        shared_experts,
                        expert_params,
                    )
                }
                // The router, a shared expert, and every non-MoE layer type
                // run on every token — active parameters equal total.
                _ => calculate_layer_params(layer),
            };
            (active as f64 * scale).round() as u64
        })
        .sum()
}

/// Whether a feed-forward layer uses a gated (three-matrix) structure.
///
/// Gating can be stated two ways and both must agree: an explicit `gated: true`
/// flag, or an activation that is inherently gated. A config asking for SwiGLU
/// is describing a gate and an up projection whether or not it also sets the
/// flag, and treating it as a plain two-matrix MLP undercounts that layer's
/// parameters and FLOPs by a third.
pub fn is_gated_mlp(params: &neurax_parser::LayerParams) -> bool {
    params.gated
        || params
            .activation
            .as_deref()
            .is_some_and(neurax_formulas::activation::is_gated_activation)
}

/// Evaluate a user-supplied equation for a custom layer.
///
/// Custom blocks are how a researcher describes an operator NEURAX has no
/// built-in formula for. The equation is evaluated in the sandboxed evaluator
/// from `neurax-formulas`, with the layer's own dimensions bound to the
/// documented variable names (`B`, `S`, `H`, `D`, `I`, `V`).
///
/// Returns `None` when there is no equation or it cannot be evaluated, so the
/// caller can fall back rather than treating a broken formula as zero work.
pub fn evaluate_custom_equation(equation: &str, layer: &Layer) -> Option<f64> {
    evaluate_custom_equation_with(equation, layer, 1, 1)
}

/// As [`evaluate_custom_equation`], with the batch and sequence length the
/// analysis is running at.
pub fn evaluate_custom_equation_with(
    equation: &str,
    layer: &Layer,
    batch: usize,
    seq_len: usize,
) -> Option<f64> {
    let hidden = layer.params.hidden_size.unwrap_or(512);
    let heads = layer.params.num_heads.unwrap_or(8).max(1);
    let intermediate = layer.params.intermediate_size.unwrap_or(4 * hidden);
    let vocab = layer.params.vocab_size.unwrap_or(0);

    let mut evaluator = neurax_formulas::custom::CustomEquationEvaluator::new();
    evaluator.set_variables(&[
        ("B", batch as f64),
        ("S", seq_len as f64),
        ("H", hidden as f64),
        ("D", (hidden / heads) as f64),
        ("I", intermediate as f64),
        ("V", vocab as f64),
        ("N", heads as f64),
    ]);

    match evaluator.evaluate(equation) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::warn!(
                "custom equation for layer '{}' could not be evaluated: {} ({})",
                layer.id,
                equation,
                error
            );
            None
        }
    }
}

#[cfg(test)]
mod moe_decomposed_tests {
    use super::*;
    use neurax_parser::parse_model_config;

    /// A block diagram draws an MoE layer as separate router / experts /
    /// combine nodes — this is the exact shape both `huggingfaceImporter.ts`
    /// and every MoE reference template in `modelTemplates.ts` produce, not
    /// a simplification for the test. Treating each of those nodes as if it
    /// alone were a complete MoE layer (the router's tiny gating matrix
    /// costed as `num_experts` full experts, and the per-layer repeat count
    /// diluted across however many same-typed nodes one logical layer used)
    /// used to put Mixtral-8x7B at 36.2B parameters and DeepSeek-MoE-16B at
    /// 46.3B — respectively 22% low and 182% high against their published
    /// sizes.
    fn decomposed_moe_json(
        hidden: usize,
        num_layers: usize,
        num_experts: usize,
        top_k: usize,
        expert_intermediate: usize,
        shared_experts: usize,
    ) -> String {
        let shared_expert_layer = if shared_experts > 0 {
            format!(
                r#",{{"id":"shared","layer_type":"moe_shared_expert","params":{{"hidden_size":{hidden},"intermediate_size":{expert_intermediate},"num_experts":{shared_experts},"activation":"silu"}}}}"#
            )
        } else {
            String::new()
        };
        format!(
            r#"{{
                "schema_version": "1.0",
                "model": {{
                    "name": "MoeTest",
                    "type": "moe",
                    "global_params": {{ "num_layers": {num_layers} }},
                    "layers": [
                        {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": {hidden}, "num_heads": 8}}}},
                        {{"id": "router", "layer_type": "moe_router", "params": {{"hidden_size": {hidden}, "num_experts": {num_experts}}}}},
                        {{"id": "experts", "layer_type": "moe", "params": {{"hidden_size": {hidden}, "intermediate_size": {expert_intermediate}, "num_experts": {num_experts}, "top_k": {top_k}, "activation": "silu"}}}},
                        {{"id": "combine", "layer_type": "moe_combine", "params": {{"hidden_size": {hidden}, "top_k": {top_k}}}}}{shared_expert_layer}
                    ]
                }},
                "training": {{"batch_size": 1}},
                "hardware": {{"gpus": [{{"name": "A100", "count": 1}}]}}
            }}"#
        )
    }

    #[test]
    fn a_dense_mlp_block_scales_only_against_its_own_share_of_the_depth() {
        // The bug this guards: an earlier version of this branch returned
        // early for every repeatable layer type once num_dense_layers was
        // set, which would have scaled attention to only num_dense_layers
        // layers instead of the model's real depth.
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "DenseScaleTest",
                "type": "moe",
                "global_params": { "num_layers": 28, "num_dense_layers": 1 },
                "layers": [
                    {"id": "attn", "layer_type": "attention", "params": {"hidden_size": 2048, "num_heads": 16}},
                    {"id": "router", "layer_type": "moe_router", "params": {"hidden_size": 2048, "num_experts": 64}},
                    {"id": "experts", "layer_type": "moe", "params": {"hidden_size": 2048, "intermediate_size": 1408, "num_experts": 64, "top_k": 6}},
                    {"id": "combine", "layer_type": "moe_combine", "params": {"hidden_size": 2048, "top_k": 6}},
                    {"id": "dense_ffn", "layer_type": "mlp", "params": {"hidden_size": 2048, "intermediate_size": 10944, "gated": true}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        let config = parse_model_config(json).unwrap();

        let attn_layer = config
            .model
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Attention)
            .unwrap();
        let dense_layer = config
            .model
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Mlp)
            .unwrap();
        let moe_layer = config
            .model
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::MoE)
            .unwrap();

        // Attention runs in all 28 layers, dense or routed.
        assert_eq!(repeat_scale_for(&config, attn_layer), 28.0);
        // The dense FFN only replaces the routed one in the 1 dense layer.
        assert_eq!(repeat_scale_for(&config, dense_layer), 1.0);
        // The routed experts run in the other 27.
        assert_eq!(repeat_scale_for(&config, moe_layer), 27.0);
    }

    #[test]
    fn mixtral_8x7b_matches_its_published_size() {
        // hidden 4096, 32 layers, 8 experts, top-2, expert width 14336, no
        // shared experts. Published: 46.7B total, ~12.9B active per token.
        let json = decomposed_moe_json(4096, 32, 8, 2, 14336, 0);
        let config = parse_model_config(&json).unwrap();

        let total = scaled_total_parameters(&config) as f64 / 1e9;
        let active = scaled_active_parameters(&config) as f64 / 1e9;

        assert!(
            (total - 46.7).abs() < 1.0,
            "Mixtral-8x7B: expected ~46.7B total, got {total:.2}B"
        );
        assert!(
            (active - 12.9).abs() < 1.0,
            "Mixtral-8x7B: expected ~12.9B active, got {active:.2}B"
        );
        // Active must be meaningfully smaller than total — the entire point
        // of the metric — not just a rounding difference.
        assert!(active < total * 0.5);
    }

    #[test]
    fn deepseek_moe_16b_matches_its_published_size() {
        // hidden 2048, 28 layers, 64 routed experts top-6, expert width
        // 1408, 2 shared experts. Published: ~16.4B total, ~2.8B active.
        let json = decomposed_moe_json(2048, 28, 64, 6, 1408, 2);
        let config = parse_model_config(&json).unwrap();

        let total = scaled_total_parameters(&config) as f64 / 1e9;
        let active = scaled_active_parameters(&config) as f64 / 1e9;

        assert!(
            (total - 16.4).abs() < 1.0,
            "DeepSeek-MoE-16B: expected ~16.4B total, got {total:.2}B"
        );
        assert!(
            (active - 2.8).abs() < 0.5,
            "DeepSeek-MoE-16B: expected ~2.8B active, got {active:.2}B"
        );
    }

    #[test]
    fn the_router_alone_is_not_costed_as_num_experts_full_experts() {
        // The bug in miniature: a router for 64 experts of width 1408 must
        // cost close to `hidden × num_experts`, not anywhere near
        // `64 × (one expert's own parameter count)`.
        let json = decomposed_moe_json(2048, 1, 64, 6, 1408, 0);
        let config = parse_model_config(&json).unwrap();
        let router_layer = config
            .model
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::MoeRouter)
            .expect("router layer present");
        let router_params = calculate_layer_params(router_layer);
        assert_eq!(router_params, 2048 * 64);
    }

    #[test]
    fn expert_combine_carries_no_parameters() {
        let json = decomposed_moe_json(2048, 1, 8, 2, 1408, 0);
        let config = parse_model_config(&json).unwrap();
        let combine_layer = config
            .model
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::MoeCombine)
            .expect("combine layer present");
        assert_eq!(calculate_layer_params(combine_layer), 0);
    }
}

#[cfg(test)]
mod gnn_tests {
    //! `graph_conv`/`graph_attention`/`message_passing` used to fall through
    //! to `Custom` with no `param_count` supplied — 0 parameters and 0 FLOPs
    //! regardless of the design. This is the exact shape NEURAX's own GCN
    //! (Cora) reference template compiles to — two `graph_conv` layers with
    //! real feature dimensions and a dropout in between — not a simplified
    //! stand-in, and the expected values are the real `gnn::gcn_params`
    //! formula evaluated by hand, not copied from this code's own output.
    use super::*;
    use neurax_parser::parse_model_config;

    fn gcn_cora_json() -> String {
        r#"{
            "schema_version": "1.0",
            "model": {
                "name": "GCN (Cora)",
                "type": "gnn",
                "global_params": { "num_nodes": 2708, "num_edges": 10556, "node_features": 64 },
                "layers": [
                    {"id": "gc1", "layer_type": "graph_conv", "params": {"in_features": 1433, "out_features": 16}},
                    {"id": "drop", "layer_type": "custom", "params": {"rate": 0.5}},
                    {"id": "gc2", "layer_type": "graph_conv", "params": {"in_features": 16, "out_features": 7}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#
        .to_string()
    }

    #[test]
    fn gcn_layer_costs_a_real_linear_transform_not_zero() {
        let config = parse_model_config(&gcn_cora_json()).unwrap();
        let gc1 = config
            .model
            .layers
            .iter()
            .find(|l| l.id == "gc1")
            .expect("first graph_conv layer present");
        assert_eq!(gc1.layer_type, LayerType::GraphConvNet);
        // 1433 * 16 weight + 16 bias (bias defaults true when unspecified) —
        // `neurax-formulas::gnn::gcn_params(1433, 16, true)` by hand.
        assert_eq!(calculate_layer_params(gc1), 1433 * 16 + 16);
    }

    #[test]
    fn second_gcn_layer_uses_its_own_shape_not_the_first_layers() {
        let config = parse_model_config(&gcn_cora_json()).unwrap();
        let gc2 = config
            .model
            .layers
            .iter()
            .find(|l| l.id == "gc2")
            .expect("second graph_conv layer present");
        assert_eq!(calculate_layer_params(gc2), 16 * 7 + 7);
    }

    #[test]
    fn dropout_between_gcn_layers_carries_no_parameters() {
        let config = parse_model_config(&gcn_cora_json()).unwrap();
        let drop = config
            .model
            .layers
            .iter()
            .find(|l| l.id == "drop")
            .expect("dropout layer present");
        assert_eq!(calculate_layer_params(drop), 0);
    }

    #[test]
    fn gat_layer_accounts_for_per_head_attention_params() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "GAT test",
                "type": "gnn",
                "global_params": { "num_nodes": 2708, "num_edges": 10556 },
                "layers": [
                    {"id": "gat1", "layer_type": "gat_attention", "params": {"in_features": 1433, "out_features": 64, "num_heads": 8}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        let config = parse_model_config(json).unwrap();
        let gat = &config.model.layers[0];
        // `gat_attention` must resolve to the dedicated GAT type, not the
        // generic transformer `Attention` it used to fall into via fuzzy
        // substring matching (`.includes("attention")`).
        assert_eq!(gat.layer_type, LayerType::GraphAttentionNet);
        assert_eq!(
            calculate_layer_params(gat),
            neurax_formulas::gnn::gat_params(1433, 64, 8, true)
        );
        assert!(calculate_layer_params(gat) > 0);
    }

    #[test]
    fn message_passing_no_longer_falls_through_to_custom() {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "MPNN test",
                "type": "gnn",
                "global_params": { "num_nodes": 2708, "num_edges": 10556 },
                "layers": [
                    {"id": "mp1", "layer_type": "message_passing", "params": {"in_features": 64, "out_features": 64}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        let config = parse_model_config(json).unwrap();
        let mp = &config.model.layers[0];
        assert_eq!(mp.layer_type, LayerType::MessagePassing);
        // The bug this guards: this used to parse as `Custom` with no
        // `param_count` supplied, and cost exactly 0 — indistinguishable
        // from a layer that is genuinely parameter-free.
        assert!(calculate_layer_params(mp) > 0);
    }
}
