//! Diffusion inference cost was never multiplied by the number of denoising
//! steps anywhere in the live pipeline — `diffusion_timesteps` parsed
//! correctly into `GlobalParams` but was read by nothing downstream, so a
//! diffusion model's reported FLOPs/latency were the cost of one U-Net
//! forward pass, not the 20-1000 a real sample takes. Classifier-free
//! guidance (Ho & Salimans, 2022 — on by default in Stable Diffusion, SDXL,
//! DALL-E) doubles that again by running the U-Net conditioned and
//! unconditioned every step.

use neurax_core::analyze_json;

fn diffusion_json(timesteps: Option<u32>, guidance_scale: Option<f64>) -> String {
    let timesteps_field = timesteps
        .map(|t| format!(r#""diffusion_timesteps": {t},"#))
        .unwrap_or_default();
    let guidance_field = guidance_scale
        .map(|g| format!(r#""guidance_scale": {g},"#))
        .unwrap_or_default();
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "Diffusion-Sampling-Cost",
                "type": "diffusion",
                "layers": [
                    {{"id": "unet", "layer_type": "unet_block", "params": {{"in_channels": 4, "out_channels": 4}}}}
                ],
                "global_params": {{ {timesteps_field} {guidance_field} "image_size": 64 }}
            }},
            "training": {{ "batch_size": 1, "precision": "fp16" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp16", "image_channels": 4, "image_height": 64, "image_width": 64 }}
        }}
        "#
    )
}

#[test]
fn no_timesteps_stated_stays_a_single_pass() {
    let json = diffusion_json(None, None);
    let result = analyze_json(&json).expect("analysis should succeed");
    let one_pass = result.compute.metrics.total_flops;
    assert!(one_pass > 0.0);
}

#[test]
fn timesteps_multiply_the_reported_cost() {
    let baseline = diffusion_json(None, None);
    let with_steps = diffusion_json(Some(50), None);

    let one_pass = analyze_json(&baseline).unwrap().compute.metrics.total_flops;
    let fifty_steps = analyze_json(&with_steps)
        .unwrap()
        .compute
        .metrics
        .total_flops;

    assert!(
        (fifty_steps / one_pass - 50.0).abs() < 0.01,
        "50 denoising steps should cost ~50x one pass, got {}x",
        fifty_steps / one_pass
    );
}

#[test]
fn classifier_free_guidance_doubles_it_again() {
    let with_steps = diffusion_json(Some(50), None);
    let with_cfg = diffusion_json(Some(50), Some(7.5));

    let without_guidance = analyze_json(&with_steps)
        .unwrap()
        .compute
        .metrics
        .total_flops;
    let with_guidance = analyze_json(&with_cfg).unwrap().compute.metrics.total_flops;

    assert!(
        (with_guidance / without_guidance - 2.0).abs() < 0.01,
        "CFG should double the cost, got {}x",
        with_guidance / without_guidance
    );
}

#[test]
fn a_real_sdxl_shaped_config_reflects_both_factors_in_latency_too() {
    // 1000 steps, CFG on — Stable Diffusion's actual defaults. Confirms the
    // fix reaches `latency_ms` (what a budget check actually reads), not
    // just the internal FLOPs metric.
    let one_pass = analyze_json(&diffusion_json(None, None)).unwrap();
    let full_sample = analyze_json(&diffusion_json(Some(1000), Some(7.5))).unwrap();

    let flops_ratio =
        full_sample.compute.metrics.total_flops / one_pass.compute.metrics.total_flops;
    let latency_ratio =
        full_sample.hardware.metrics.latency_ms / one_pass.hardware.metrics.latency_ms;

    assert!(
        (flops_ratio - 2000.0).abs() < 1.0,
        "flops ratio: {flops_ratio}"
    );
    assert!(
        (latency_ratio - 2000.0).abs() < 1.0,
        "latency ratio: {latency_ratio}"
    );
}
