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
//! This test does not assert either model matches its true published size
//! (both example templates model a U-Net with one node per resolution stage
//! rather than the real per-stage block count, so neither is expected to
//! reach the true ~860M / ~2.6B figures yet — a template-completeness gap,
//! not a formula bug, and a separate piece of work). It only asserts the
//! bug's own symptom is gone: two differently-configured models must not
//! collapse to an identical total, and a cross-attention block's own count
//! must react to its own context_dim.

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
