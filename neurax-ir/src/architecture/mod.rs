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
    // OpSpec-IR (`neurax-opspec`): a migrated layer type's params formula
    // lives in exactly one place, alongside its FLOPs formula. Anything not
    // yet migrated falls through to the match below unchanged.
    if let Some(spec) = neurax_opspec::op_spec(layer.layer_type) {
        return (spec.params_fn)(layer);
    }

    match layer.layer_type {
        // Migrated to OpSpec-IR (`neurax-opspec`) — the early-return above
        // already returns for every one of these before the match runs, so
        // reaching this arm would mean `op_spec()` and this match disagree
        // about which types are migrated. Every real `LayerType` except
        // `Custom` is migrated.
        LayerType::Embedding
        | LayerType::Attention
        | LayerType::Mlp
        | LayerType::Conv
        | LayerType::Dense
        | LayerType::LoraLinear
        | LayerType::DoraLinear
        | LayerType::Normalization
        | LayerType::Pooling
        | LayerType::MoE
        | LayerType::MoeRouter
        | LayerType::MoeCombine
        | LayerType::MoeSharedExpert
        | LayerType::ResidualBlock
        | LayerType::ResnetBottleneck
        | LayerType::Mbconv
        | LayerType::Inception
        | LayerType::DenseBlock
        | LayerType::ConvnextBlock
        | LayerType::ShuffleUnit
        | LayerType::C2f
        | LayerType::Detection
        | LayerType::Transition
        | LayerType::MambaBlock
        | LayerType::S4Block
        | LayerType::StateSpace
        | LayerType::H3Block
        | LayerType::RwkvBlock
        | LayerType::RetentionBlock
        | LayerType::GeneratorBlock
        | LayerType::DiscriminatorBlock
        | LayerType::StyleMod
        | LayerType::AdaIN
        | LayerType::MinibatchStd
        | LayerType::PixelNorm
        | LayerType::SelfAttention
        | LayerType::SpectralNorm
        | LayerType::ProgressiveBlock
        | LayerType::LstmBlock
        | LayerType::GruBlock
        | LayerType::RnnCell
        | LayerType::Bidirectional
        | LayerType::EncoderBlock
        | LayerType::DecoderBlock
        | LayerType::UnetBlock
        | LayerType::TimeEmbedding
        | LayerType::CrossAttention
        | LayerType::DownBlock
        | LayerType::UpBlock
        | LayerType::MidBlock
        | LayerType::ResnetBlock
        | LayerType::TimestepBlock
        | LayerType::ConditionBlock
        | LayerType::NoisePredictor
        | LayerType::VaeEncoder
        | LayerType::VaeDecoder
        | LayerType::GraphConvNet
        | LayerType::MessagePassing
        | LayerType::GraphAttentionNet
        | LayerType::RgcnConv => unreachable!(
            "{:?} is registered in OpSpec-IR (neurax-opspec); the early return above must have handled it",
            layer.layer_type
        ),
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

    // An encoder-decoder model (T5, BART, Pegasus...) lists an encoder's and
    // a decoder's representative blocks side by side in the same flat list,
    // each tagged `encoder_decoder_role` on its `params`. Plain models never
    // set this — `is_decoder` is `false` for every layer on every one of
    // them, `depth` falls straight through to `global_num_layers`, and
    // `count_of` counts exactly what it always counted, so this is a no-op
    // for every model this function already handled correctly.
    let is_decoder_role = |l: &Layer| l.params.encoder_decoder_role.as_deref() == Some("decoder");
    let is_decoder = is_decoder_role(layer);
    let depth = if is_decoder {
        config
            .model
            .global_params
            .num_decoder_layers
            .map(|n| n as usize)
            .unwrap_or(global_num_layers)
    } else {
        global_num_layers
    };

    // A Jamba-style hybrid's representative `Attention` and `MambaBlock`
    // each stand for their own share of `depth`, not all of it —
    // `ai21labs/Jamba-v0.1` runs attention on 4 of its 32 layers
    // (`attn_layer_period: 8`) and Mamba on the other 28; without this,
    // both blocks would scale to the model's full 32 layers each,
    // double-counting every layer as if it ran both sublayers at once.
    // Absent (every non-hybrid model) leaves `depth` untouched.
    let depth = match layer.params.repeat_fraction {
        Some(fraction) => ((depth as f64) * fraction).round() as usize,
        None => depth,
    };

    let count_of = |pred: &dyn Fn(&Layer) -> bool| {
        layers
            .iter()
            .filter(|l| pred(l) && is_decoder_role(l) == is_decoder)
            .count()
    };
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
    ) || (layer.layer_type == LayerType::CrossAttention && is_decoder);
    // `CrossAttention` is deliberately excluded from the `matches!` above: a
    // diffusion U-Net's cross-attention already scales itself through its
    // own `transformer_layers_per_block` field (see
    // `neurax-opspec::cross_attention_params_fn`) — repeat-scaling it again
    // here would double-count every image-conditioning block (verified
    // against `examples/models/sdxl_1.0.json`, which lists 3 real
    // `cross_attention` nodes under `global_params.num_layers: 50`). Only an
    // encoder-decoder text model's decoder cross-attention — reachable only
    // via `encoder_decoder_role`, a marker no diffusion config sets — is
    // meant to repeat by depth the way every other repeatable kind does.
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
            let num_moe_layers = depth.saturating_sub(num_dense_layers);
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
    if depth > listed {
        depth as f64 / listed as f64
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
    fn an_encoder_decoder_models_two_stacks_scale_independently() {
        // T5-shaped: a 6-layer encoder and a 6-layer decoder, each listed as
        // one representative block per sublayer. The bug this guards: before
        // `encoder_decoder_role` existed, T5 had no way to be told apart from
        // a plain decoder-only model, so its single listed `attention` block
        // absorbed both the encoder's and the decoder's self-attention counts
        // into one bucket and its decoder had no cross-attention at all.
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "T5Test",
                "type": "transformer",
                "global_params": { "num_layers": 6, "num_decoder_layers": 6 },
                "layers": [
                    {"id": "enc_attn", "layer_type": "attention", "params": {"hidden_size": 512, "num_heads": 8, "causal": false}},
                    {"id": "enc_mlp", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 2048}},
                    {"id": "dec_self_attn", "layer_type": "attention", "params": {"hidden_size": 512, "num_heads": 8, "causal": true, "encoder_decoder_role": "decoder"}},
                    {"id": "dec_cross_attn", "layer_type": "cross_attention", "params": {"hidden_size": 512, "num_heads": 8, "encoder_decoder_role": "decoder"}},
                    {"id": "dec_mlp", "layer_type": "mlp", "params": {"hidden_size": 512, "intermediate_size": 2048, "encoder_decoder_role": "decoder"}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        let config = parse_model_config(json).unwrap();
        let find = |id: &str| config.model.layers.iter().find(|l| l.id == id).unwrap();

        // Every one of the five representative blocks stands for its own
        // real 6 layers — not 3 (two same-typed decoder blocks splitting one
        // shared count) and not 12 (the two stacks' depths merged into one).
        assert_eq!(repeat_scale_for(&config, find("enc_attn")), 6.0);
        assert_eq!(repeat_scale_for(&config, find("enc_mlp")), 6.0);
        assert_eq!(repeat_scale_for(&config, find("dec_self_attn")), 6.0);
        assert_eq!(repeat_scale_for(&config, find("dec_cross_attn")), 6.0);
        assert_eq!(repeat_scale_for(&config, find("dec_mlp")), 6.0);
    }

    #[test]
    fn a_diffusion_unets_cross_attention_is_not_repeat_scaled() {
        // The regression this guards against directly: adding
        // `CrossAttention` to the repeatable set for T5 must not start
        // repeat-scaling a diffusion U-Net's cross-attention too — it
        // already scales itself via `transformer_layers_per_block` inside
        // its own opspec formula. No `encoder_decoder_role` is set here,
        // exactly like every real diffusion config on the Hub.
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "UnetCrossAttnTest",
                "type": "diffusion",
                "global_params": { "num_layers": 50 },
                "layers": [
                    {"id": "cross_attn", "layer_type": "cross_attention", "params": {"hidden_size": 320, "num_heads": 8, "transformer_layers_per_block": 2}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        let config = parse_model_config(json).unwrap();
        let cross_attn = &config.model.layers[0];

        assert_eq!(repeat_scale_for(&config, cross_attn), 1.0);
    }

    #[test]
    fn a_jamba_style_hybrid_splits_attention_and_mamba_by_repeat_fraction() {
        // Real structure, read from `ai21labs/Jamba-v0.1`'s own config.json:
        // 32 layers total, `attn_layer_period: 8` / `attn_layer_offset: 4`
        // puts attention on layers 4, 12, 20, 28 — 4 of 32 — and Mamba on
        // the other 28. (Its `expert_layer_period`/`expert_layer_offset`
        // MoE interleaving is a second, independent axis this block does
        // not model — out of scope for the canvas "Hybrid SSM+Attention"
        // block, which only ever had a single `mix_ratio` knob.)
        //
        // The bug this guards: before `repeat_fraction` existed, a
        // representative `Attention` and a representative `MambaBlock`
        // both scaled to the model's full depth (32 each) — as if every
        // layer ran both sublayers at once, rather than one or the other.
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "JambaHybridTest",
                "type": "ssm",
                "global_params": { "num_layers": 32 },
                "layers": [
                    {"id": "attn", "layer_type": "attention", "params": {"hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8, "causal": true, "repeat_fraction": 0.125}},
                    {"id": "mamba", "layer_type": "mamba_block", "params": {"hidden_size": 4096, "state_dim": 16, "expansion_factor": 2, "repeat_fraction": 0.875}}
                ]
            },
            "training": {"batch_size": 1},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        let config = parse_model_config(json).unwrap();
        let find = |id: &str| config.model.layers.iter().find(|l| l.id == id).unwrap();

        assert_eq!(repeat_scale_for(&config, find("attn")), 4.0);
        assert_eq!(repeat_scale_for(&config, find("mamba")), 28.0);
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
