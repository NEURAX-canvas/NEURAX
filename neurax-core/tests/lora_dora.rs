//! `lora_linear`/`dora_linear` had no arm in `LayerType::from_str` at all —
//! the canvas collapsed both into plain `"dense"`, costing a fine-tuning
//! adapter as a full `in_features x out_features` layer instead of the
//! rank-decomposed `rank x (in + out)` it actually is. For an outil whose
//! whole value proposition includes "can I afford to fine-tune this
//! model", measuring LoRA as no cheaper than full fine-tuning is the exact
//! opposite of correct.

use neurax_core::analyze_json;

fn linear_json(layer_type: &str, rank: Option<usize>) -> String {
    let rank_field = rank.map(|r| format!(r#","rank": {r}"#)).unwrap_or_default();
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "Lora-Check",
                "type": "transformer",
                "layers": [
                    {{"id": "l1", "layer_type": "{layer_type}", "params": {{"in_features": 4096, "out_features": 4096{rank_field}}}}}
                ]
            }},
            "training": {{ "batch_size": 1, "precision": "fp16" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp16" }}
        }}
        "#
    )
}

#[test]
fn lora_costs_far_less_than_the_dense_layer_it_adapts() {
    let dense = analyze_json(&linear_json("dense", None))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    let lora = analyze_json(&linear_json("lora_linear", Some(16)))
        .unwrap()
        .arch
        .metrics
        .total_parameters;

    assert!(
        lora < dense / 50,
        "LoRA (rank 16, 4096x4096) should be well under 1/50th of the dense layer, got {lora} vs {dense}"
    );
}

#[test]
fn dora_costs_slightly_more_than_lora() {
    let lora = analyze_json(&linear_json("lora_linear", Some(16)))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    let dora = analyze_json(&linear_json("dora_linear", Some(16)))
        .unwrap()
        .arch
        .metrics
        .total_parameters;

    assert!(
        dora > lora,
        "DoRA's magnitude vector should add some parameters over plain LoRA"
    );
    assert!(
        dora < lora * 2,
        "DoRA shouldn't cost anywhere close to double LoRA — got {dora} vs {lora}"
    );
}

#[test]
fn a_higher_rank_costs_more() {
    let low = analyze_json(&linear_json("lora_linear", Some(4)))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    let high = analyze_json(&linear_json("lora_linear", Some(64)))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    assert!(high > low);
}
