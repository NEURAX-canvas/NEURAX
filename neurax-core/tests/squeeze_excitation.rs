//! `LayerParams.se: bool` existed but `mbconv_params` had no `se` parameter
//! in its signature at all — the branch was structurally absent, not just
//! ignored. SE is the difference between MobileNetV2 (no SE) and
//! MobileNetV3/EfficientNet (SE in every block) — a real EfficientNet
//! config had its SE blocks silently costed as zero regardless of the `se`
//! flag.

use neurax_core::analyze_json;

fn mbconv_json(se: bool) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "SE-Check",
                "type": "cnn",
                "layers": [
                    {{"id": "block1", "layer_type": "mbconv", "params": {{"in_channels": 32, "out_channels": 16, "expansion_factor": 6, "kernel_size": 3, "se": {se}}}}}
                ]
            }},
            "training": {{ "batch_size": 1, "precision": "fp32" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp32", "image_channels": 32, "image_height": 224, "image_width": 224 }}
        }}
        "#
    )
}

#[test]
fn a_block_with_se_costs_more_than_without() {
    let without_se = analyze_json(&mbconv_json(false))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    let with_se = analyze_json(&mbconv_json(true)).unwrap().arch.metrics.total_parameters;

    assert!(
        with_se > without_se,
        "an EfficientNet/MobileNetV3-style block with SE enabled must cost more than the same block without it"
    );
}
