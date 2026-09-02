//! Before/after regression test for the complete OpSpec-IR migration
//! (2026-09-02): every `LayerType` except `Custom` now resolves through
//! `neurax-opspec`'s registry instead of the two original, independently
//! maintained match arms in `architecture/mod.rs` and `operator/pass.rs`.
//!
//! These exact numbers were captured from the *pre-migration* match arms
//! (temporarily isolating `neurax_opspec::op_spec` to just the CNN block
//! family, so every other type fell through to its old arm, per-representative
//! layer, batch=2/seq=196/dtype=fp16) and diffed byte-for-byte against the
//! post-migration numbers for the same layers. Every value below is
//! identical between the two runs except `Conv`'s FLOPs — the one real bug
//! this migration fixed (see `conv_flops_reads_the_fixed_kernel_shape`
//! below) — which is why this file pins the *post-migration* (correct)
//! value there and documents the pre-migration one in the test itself.
//!
//! A future change to any formula this test covers must update this file
//! deliberately — that's the point: this is the safety net the premortem
//! in `neurax-opspec/README.md` flagged as still missing after the CNN-only
//! migration ("nothing prevents human regression").

use neurax_ir::architecture::{calculate_layer_params, LayerDef};
use neurax_ir::operator::layer_flops;
use neurax_ir::NeuraxContext;
use neurax_parser::{parse_model_config, Layer, LayerParams, LayerType};

fn test_ctx() -> NeuraxContext {
    let json = r#"{
        "schema_version": "1.0",
        "model": { "name": "before_after", "type": "transformer", "layers": [
            {"id": "dummy", "layer_type": "embedding", "params": {"vocab_size": 100}}
        ] },
        "training": {"batch_size": 2},
        "hardware": {"gpus": [{"name": "A100", "count": 1}]}
    }"#;
    NeuraxContext::new(parse_model_config(json).unwrap())
}

/// Computes (params, flops) for one layer through the real, current
/// pipeline entry points — the exact functions `neurax-ir` exposes and the
/// exact ones a migrated type's early-return in `calculate_layer_params`/
/// `decompose_layer_to_ops` now serves.
fn params_and_flops(layer_type: LayerType, params: LayerParams) -> (u64, f64) {
    let layer = Layer {
        id: "t".to_string(),
        layer_type,
        input_shape: vec![],
        output_shape: vec![],
        params,
        custom_equations: None,
    };
    let param_count = calculate_layer_params(&layer);
    let mut layer_def = LayerDef::from(&layer);
    layer_def.param_count = param_count;
    let flops = layer_flops(&layer_def, 2, 196, "fp16", &test_ctx());
    (param_count, flops)
}

fn params_and_flops_with_shapes(
    layer_type: LayerType,
    params: LayerParams,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
) -> (u64, f64) {
    let layer = Layer {
        id: "t".to_string(),
        layer_type,
        input_shape,
        output_shape,
        params,
        custom_equations: None,
    };
    let param_count = calculate_layer_params(&layer);
    let mut layer_def = LayerDef::from(&layer);
    layer_def.param_count = param_count;
    let flops = layer_flops(&layer_def, 2, 196, "fp16", &test_ctx());
    (param_count, flops)
}

#[test]
fn transformer_family_matches_pre_migration_values() {
    assert_eq!(
        params_and_flops(
            LayerType::Embedding,
            LayerParams {
                vocab_size: Some(32000),
                embedding_dim: Some(768),
                ..Default::default()
            }
        ),
        (24576000, 301056.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::Attention,
            LayerParams {
                hidden_size: Some(768),
                num_heads: Some(12),
                ..Default::default()
            }
        ),
        (2359296, 2091530112.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::Attention,
            LayerParams {
                hidden_size: Some(4096),
                num_heads: Some(32),
                num_kv_heads: Some(8),
                ..Default::default()
            }
        ),
        (41943040, 34148581376.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::Mlp,
            LayerParams {
                hidden_size: Some(768),
                intermediate_size: Some(3072),
                ..Default::default()
            }
        ),
        (4718592, 3711418368.0)
    );
    assert_eq!(
        params_and_flops_with_shapes(
            LayerType::Dense,
            LayerParams {
                in_features: Some(512),
                out_features: Some(256),
                bias: true,
                ..Default::default()
            },
            vec![2, 196, 512],
            vec![2, 196, 256],
        ),
        (131328, 102760448.0)
    );
    assert_eq!(
        params_and_flops_with_shapes(
            LayerType::LoraLinear,
            LayerParams {
                in_features: Some(512),
                out_features: Some(256),
                rank: Some(8),
                ..Default::default()
            },
            vec![2, 196, 512],
            vec![],
        ),
        (6144, 4816896.0)
    );
    assert_eq!(
        params_and_flops_with_shapes(
            LayerType::DoraLinear,
            LayerParams {
                in_features: Some(512),
                out_features: Some(256),
                rank: Some(8),
                ..Default::default()
            },
            vec![2, 196, 512],
            vec![],
        ),
        (6400, 4917248.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::Normalization,
            LayerParams {
                hidden_size: Some(768),
                ..Default::default()
            }
        ),
        (1536, 1505280.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::Normalization,
            LayerParams {
                hidden_size: Some(768),
                activation: Some("rms".to_string()),
                ..Default::default()
            }
        ),
        (768, 903168.0)
    );
    assert_eq!(
        params_and_flops_with_shapes(
            LayerType::Pooling,
            LayerParams {
                kernel_size: Some(3),
                stride: Some(2),
                ..Default::default()
            },
            vec![2, 64, 112, 112],
            vec![],
        ),
        (0, 1204224.0)
    );
}

#[test]
fn conv_flops_reads_the_fixed_kernel_shape() {
    // Real bug found and fixed during this migration: the pre-migration
    // FLOPs arm derived kernel_w from kernel_size alone, never reading
    // params.kernel_h — so this exact layer (kernel_h=7, kernel_w=7,
    // kernel_size unset) was silently costed with an effective (3, 7)
    // kernel, giving 205_922_304 FLOPs. The params side never had this bug
    // (it always read kernel_h/kernel_w directly), so the two disagreed on
    // the very shape they were describing — precisely the class of bug
    // OpSpec-IR exists to make impossible. Post-migration, both sides read
    // (7, 7) and agree on 472_055_808.
    let (params, flops) = params_and_flops_with_shapes(
        LayerType::Conv,
        LayerParams {
            in_channels: Some(3),
            out_channels: Some(64),
            kernel_h: Some(7),
            kernel_w: Some(7),
            stride: Some(2),
            padding: Some(3),
            ..Default::default()
        },
        vec![2, 3, 224, 224],
        vec![],
    );
    assert_eq!(params, 9408);
    assert_eq!(flops, 472055808.0);
    assert_ne!(
        flops, 205922304.0,
        "must not regress to the pre-migration (3, 7) kernel bug"
    );
}

#[test]
fn moe_family_matches_pre_migration_values() {
    assert_eq!(
        params_and_flops(
            LayerType::MoE,
            LayerParams {
                hidden_size: Some(4096),
                intermediate_size: Some(14336),
                num_experts: Some(8),
                top_k: Some(2),
                ..Default::default()
            }
        ),
        (1409318912, 276301993408.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::MoeRouter,
            LayerParams {
                hidden_size: Some(4096),
                num_experts: Some(8),
                ..Default::default()
            }
        ),
        (32768, 25705792.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::MoeCombine,
            LayerParams {
                hidden_size: Some(4096),
                top_k: Some(2),
                ..Default::default()
            }
        ),
        (0, 6422528.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::MoeSharedExpert,
            LayerParams {
                hidden_size: Some(4096),
                intermediate_size: Some(14336),
                num_experts: Some(2),
                ..Default::default()
            }
        ),
        (352321536, 276276281344.0)
    );
}

#[test]
fn ssm_family_matches_pre_migration_values() {
    assert_eq!(
        params_and_flops(
            LayerType::MambaBlock,
            LayerParams {
                hidden_size: Some(2560),
                state_dim: Some(16),
                expansion_factor: Some(2),
                ..Default::default()
            }
        ),
        (39592960, 30912757760.0)
    );
    let s4 = LayerParams {
        hidden_size: Some(512),
        state_dim: Some(64),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::S4Block, s4.clone()),
        (623104, 439788509.4491066)
    );
    assert_eq!(
        params_and_flops(LayerType::StateSpace, s4),
        (623104, 439788509.4491066)
    );
    assert_eq!(
        params_and_flops(
            LayerType::H3Block,
            LayerParams {
                hidden_size: Some(512),
                state_dim: Some(64),
                ..Default::default()
            }
        ),
        (721920, 1290618810.8982131)
    );
    assert_eq!(
        params_and_flops(
            LayerType::RwkvBlock,
            LayerParams {
                hidden_size: Some(4096),
                ..Default::default()
            }
        ),
        (218148864, 79055421440.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::RetentionBlock,
            LayerParams {
                hidden_size: Some(4096),
                ..Default::default()
            }
        ),
        (218120192, 79055421440.0)
    );
}

#[test]
fn gan_family_matches_pre_migration_values() {
    let conv_block = LayerParams {
        in_channels: Some(128),
        out_channels: Some(64),
        kernel_size: Some(3),
        stride: Some(1),
        padding: Some(1),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::GeneratorBlock, conv_block),
        (73728, 57802752.0)
    );
    let conv_block_2 = LayerParams {
        in_channels: Some(64),
        out_channels: Some(128),
        kernel_size: Some(3),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::DiscriminatorBlock, conv_block_2.clone()),
        (73728, 42467328.0)
    );
    assert_eq!(
        params_and_flops(LayerType::ProgressiveBlock, conv_block_2),
        (73728, 42467328.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::SelfAttention,
            LayerParams {
                out_channels: Some(512),
                ..Default::default()
            }
        ),
        (1048576, 982508800.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::StyleMod,
            LayerParams {
                out_channels: Some(512),
                ..Default::default()
            }
        ),
        (1024, 401408.0)
    );
    let norm_weight = LayerParams {
        out_channels: Some(512),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::AdaIN, norm_weight.clone()),
        (0, 602112.0)
    );
    assert_eq!(
        params_and_flops(LayerType::MinibatchStd, norm_weight.clone()),
        (0, 602112.0)
    );
    assert_eq!(
        params_and_flops(LayerType::PixelNorm, norm_weight),
        (0, 602112.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::SpectralNorm,
            LayerParams {
                in_channels: Some(64),
                out_channels: Some(128),
                ..Default::default()
            }
        ),
        (64, 150528.0)
    );
}

#[test]
fn rnn_family_matches_pre_migration_values() {
    assert_eq!(
        params_and_flops(
            LayerType::LstmBlock,
            LayerParams {
                rnn_hidden_size: Some(512),
                hidden_size: Some(768),
                num_rnn_layers: Some(2),
                bidirectional_rnn: true,
                ..Default::default()
            }
        ),
        (11542528, 9087074304.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::GruBlock,
            LayerParams {
                rnn_hidden_size: Some(512),
                hidden_size: Some(768),
                num_rnn_layers: Some(2),
                ..Default::default()
            }
        ),
        (3542016, 2787778560.0)
    );
    let rnn_base = LayerParams {
        rnn_hidden_size: Some(512),
        hidden_size: Some(768),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::RnnCell, rnn_base.clone()),
        (655872, 515809280.0)
    );
    assert_eq!(
        params_and_flops(LayerType::Bidirectional, rnn_base.clone()),
        (5246976, 4132495360.0)
    );
    assert_eq!(
        params_and_flops(LayerType::EncoderBlock, rnn_base.clone()),
        (2623488, 2066247680.0)
    );
    assert_eq!(
        params_and_flops(LayerType::DecoderBlock, rnn_base),
        (2623488, 2066247680.0)
    );
}

#[test]
fn diffusion_family_matches_pre_migration_values() {
    let unet_base = LayerParams {
        hidden_size: Some(320),
        layers_per_block: Some(2),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::UnetBlock, unet_base.clone()),
        (3688960, 2890137600.0)
    );
    assert_eq!(
        params_and_flops(LayerType::ResnetBlock, unet_base.clone()),
        (3688960, 2890137600.0)
    );
    assert_eq!(
        params_and_flops(LayerType::UpBlock, unet_base.clone()),
        (3688960, 2890137600.0)
    );
    assert_eq!(
        params_and_flops(LayerType::MidBlock, unet_base),
        (3688960, 2890137600.0)
    );
    let time_embed = LayerParams {
        hidden_size: Some(320),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::TimeEmbedding, time_embed.clone()),
        (820800, 644259840.0)
    );
    assert_eq!(
        params_and_flops(LayerType::TimestepBlock, time_embed.clone()),
        (820800, 644259840.0)
    );
    assert_eq!(
        params_and_flops(LayerType::ConditionBlock, time_embed),
        (820800, 647270400.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::CrossAttention,
            LayerParams {
                hidden_size: Some(1280),
                cross_attention_dim: Some(2048),
                num_heads: Some(20),
                transformer_layers_per_block: Some(10),
                ..Default::default()
            }
        ),
        (85248000, 55390854400.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::DownBlock,
            LayerParams {
                block_out_channels: Some(vec![128, 256, 256]),
                in_channels: Some(3),
                layers_per_block: Some(2),
                ..Default::default()
            }
        ),
        (4908544, 3843682304.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::NoisePredictor,
            LayerParams {
                out_channels_diff: Some(4),
                ..Default::default()
            }
        ),
        (144, 112896.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::VaeEncoder,
            LayerParams {
                in_channels: Some(3),
                out_channels_diff: Some(4),
                ..Default::default()
            }
        ),
        (108, 84672.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::VaeDecoder,
            LayerParams {
                in_channels: Some(4),
                out_channels_diff: Some(3),
                ..Default::default()
            }
        ),
        (108, 84672.0)
    );
}

#[test]
fn gnn_family_matches_pre_migration_values() {
    let gnn_base = LayerParams {
        in_features: Some(128),
        out_features: Some(64),
        ..Default::default()
    };
    assert_eq!(
        params_and_flops(LayerType::GraphConvNet, gnn_base.clone()),
        (8192, 45390080.0)
    );
    assert_eq!(
        params_and_flops(LayerType::MessagePassing, gnn_base.clone()),
        (16384, 90086912.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::GraphAttentionNet,
            LayerParams {
                num_heads: Some(8),
                ..gnn_base.clone()
            }
        ),
        (8320, 46816864.0)
    );
    assert_eq!(
        params_and_flops(
            LayerType::RgcnConv,
            LayerParams {
                num_relations: Some(5),
                ..gnn_base
            }
        ),
        (49152, 267229440.0)
    );
}
