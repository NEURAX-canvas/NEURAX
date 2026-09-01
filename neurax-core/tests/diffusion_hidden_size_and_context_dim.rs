//! Regression test for a real bug: `UnetBlock`/`ResnetBlock`/`DownBlock`/
//! `UpBlock`/`MidBlock` read `in_channels_diff`/`in_channels` for their
//! channel width, which no real diffusion template in this repo ever sets —
//! both SDXL and SD1.5 only set `hidden_size` on these blocks. Every stage
//! fell back to the same 320-channel default regardless of its real width,
//! and `CrossAttention` read K/V as `hidden_size`-wide instead of from the
//! conditioning sequence's own width (`context_dim` in these templates,
//! `cross_attention_dim` in the schema — two different names for the field,
//! neither of which reached this code path before the fix).
//!
//! The result: SDXL and SD1.5 — real models that differ by roughly 3x in
//! U-Net size — produced the exact same total parameter count,
//! 15,811,844, to the last digit. Verified directly before fixing.
//!
//! A second, related bug in the same fix: `layers_per_block` and
//! `transformer_layers_per_block` were already parsed from JSON into
//! `LayerParams` but read nowhere in the compiler — each U-Net/cross-
//! attention node represents one *stage*, and real U-Nets repeat several
//! blocks per stage (2 ResNet blocks per down/mid stage, 3 per up stage;
//! SDXL's deepest stage stacks 10 transformer layers, per its published
//! diffusers config). Wiring these in, plus adding the one cross-attention
//! stage (1280-channel) both templates were missing entirely, moved the
//! measured totals from 15.8M (both, identical) to 261.2M (SD1.5) and
//! 347.1M (SDXL) — real numbers, correctly ordered, but still well under
//! the true published U-Net sizes (~860M / ~2.6B). The templates still
//! model a coarser network than the real one (no time-embedding
//! projection inside each ResNet block, no separate down/upsample convs),
//! which is a template-fidelity gap, not a bug in the formulas that do
//! exist — left as a separate, larger piece of work rather than folded
//! into this fix.

use std::path::PathBuf;

fn total_parameters(file: &str) -> u64 {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate has a parent directory")
        .join("examples/models")
        .join(file);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} should be readable: {e}", path.display()));
    let config = neurax_parser::parse_model_config(&json)
        .unwrap_or_else(|e| panic!("{} should parse: {e}", path.display()));
    neurax_ir::architecture::scaled_total_parameters(&config)
}

#[test]
fn sdxl_and_sd15_no_longer_collapse_to_the_same_total() {
    let sdxl = total_parameters("sdxl_1.0.json");
    let sd15 = total_parameters("stable_diffusion_1.5.json");

    assert_ne!(
        sdxl, sd15,
        "SDXL and SD1.5 produced identical totals ({sdxl}) — the hidden_size/context_dim \
         fields that actually distinguish them are being ignored again"
    );

    // SDXL's real U-Net (~2.6B) is roughly 3x SD1.5's (~860M) — both
    // templates model a coarse, single-node-per-stage abstraction of the
    // real block count, so neither reaches its true published size (see
    // the module doc comment), but SDXL must still come out bigger than
    // SD1.5 once layers_per_block/transformer_layers_per_block (also
    // parsed-but-never-read before this fix) are actually applied —
    // SDXL's deepest stage alone stacks 10 transformer layers per the
    // real diffusers config vs SD1.5's uniform 1.
    assert!(
        sdxl > sd15,
        "SDXL ({sdxl}) should be larger than SD1.5 ({sd15}) — it has more attention \
         heads, a wider conditioning dimension (2048 vs 768), and a much deeper final \
         stage (transformer_layers_per_block=10 vs 1)"
    );
}

#[test]
fn cross_attention_param_count_reacts_to_its_own_context_dim() {
    use neurax_formulas::attention::cross_attention_params;

    let narrow_context = cross_attention_params(320, 768, true); // SD1.5-shaped
    let wide_context = cross_attention_params(320, 2048, true); // SDXL-shaped

    assert!(
        wide_context > narrow_context,
        "a wider conditioning context (2048) should cost more than a narrower one (768) \
         at the same hidden_size, got {wide_context} <= {narrow_context}"
    );

    // Regression guard for the specific bug: treating context_dim as
    // hidden_size collapses this to a fixed self-attention formula that
    // can't tell 768 from 2048 apart at all.
    let self_attention_shaped = cross_attention_params(320, 320, true);
    assert_ne!(narrow_context, self_attention_shaped);
    assert_ne!(wide_context, self_attention_shaped);
}

/// A third, separate bug found after the two above were fixed:
/// `neurax-ui`'s own templates (e.g. `mnist_diffusion`) build encoder/decoder
/// stages with a `unet_encoder`/`unet_decoder` node type carrying a
/// per-stage channel list (`channels: [128, 256, 256]`) and a
/// `num_res_blocks` repeat count — a shape `DownBlock`/`UpBlock`'s formula
/// couldn't consume at all. Two compounding problems: the frontend routed
/// both node types to the generic `.endsWith('encoder'/'decoder')` fallback
/// (`encoder_block`/`decoder_block`, an LSTM formula — costing an image
/// U-Net stage as if it were a recurrent cell), and even after fixing that
/// routing, `block_out_channels`/`layers_per_block` were parsed under their
/// diffusers names only, never under the `channels`/`num_res_blocks` names
/// these templates actually write.
#[test]
fn unet_encoder_with_a_per_stage_channel_list_costs_each_transition() {
    let json = r#"{
        "schema_version": "1.0",
        "model": {
            "name": "unet-encoder-test",
            "type": "diffusion",
            "global_params": { "num_layers": 1, "hidden_size": 128 },
            "layers": [
                {"id": "enc", "layer_type": "down_block", "params": {
                    "in_channels": 3, "channels": [128, 256, 256], "num_res_blocks": 2
                }}
            ]
        },
        "training": {"batch_size": 1, "max_steps": 100},
        "hardware": {"gpus": [{"name": "A100-SXM", "count": 1}]}
    }"#;
    let config = neurax_parser::parse_model_config(json).expect("parses");
    let measured = neurax_ir::architecture::scaled_total_parameters(&config);

    // Hand-computed: one stage transition per consecutive width pair
    // (3->128, 128->256, 256->256), each stage's first block carrying the
    // channel change (and its projection shortcut when in != out) and the
    // remaining num_res_blocks-1 block(s) at constant width.
    use neurax_formulas::cnn_blocks::resnet_basic_block_params;
    let widths = [3usize, 128, 256, 256];
    let expected: u64 = widths
        .windows(2)
        .map(|w| {
            let first = resnet_basic_block_params(w[0], w[1], 1, true);
            let rest = resnet_basic_block_params(w[1], w[1], 1, true);
            first + rest // num_res_blocks=2: 1 changing block + 1 constant-width block
        })
        .sum();

    assert_eq!(
        measured, expected,
        "a down_block with a per-stage channels list should cost each width \
         transition separately — got {measured}, expected {expected}"
    );
    assert!(
        measured > 0,
        "an encoder stage must never cost 0 parameters"
    );
}
