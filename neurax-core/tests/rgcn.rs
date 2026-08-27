//! `rgcn_conv` had no arm in `LayerType::from_str` at all — the canvas's
//! `toParserLayerType` fell all the way through to its generic
//! `.includes('conv')` catch-all and sent it to the compiler as `"conv"`,
//! costing a knowledge-graph layer as a plain image Conv2D
//! (in_channels/out_channels/kernel_size), not even the existing GCN
//! formula, let alone one that knows about `num_relations`.

use neurax_core::analyze_json;

fn rgcn_json(num_relations: usize) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "RGCN-Check",
                "type": "gnn",
                "layers": [
                    {{"id": "rgcn1", "layer_type": "rgcn_conv", "params": {{"in_features": 100, "out_features": 100, "num_relations": {num_relations}}}}}
                ],
                "global_params": {{ "extra": {{ "num_nodes": 1000, "num_edges": 5000 }} }}
            }},
            "training": {{ "batch_size": 1, "precision": "fp32" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp32" }}
        }}
        "#
    )
}

#[test]
fn rgcn_parses_as_its_own_layer_type_not_conv() {
    let result = analyze_json(&rgcn_json(5)).expect("analysis should succeed");
    assert_eq!(result.arch.metrics.num_layers, 1);
    assert!(result.arch.metrics.total_parameters > 0);
}

#[test]
fn more_relations_means_more_parameters() {
    let few = analyze_json(&rgcn_json(2))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    let many = analyze_json(&rgcn_json(50))
        .unwrap()
        .arch
        .metrics
        .total_parameters;
    assert!(
        many > few,
        "an RGCN with 50 relation types must cost more than one with 2 — got {many} vs {few}"
    );
}

#[test]
fn a_plain_conv2d_with_the_same_channel_numbers_costs_a_different_amount() {
    // The exact bug this fixes: rgcn_conv used to be indistinguishable from
    // a Conv2D with in_channels/out_channels matching in/out_features.
    let rgcn = analyze_json(&rgcn_json(5))
        .unwrap()
        .arch
        .metrics
        .total_parameters;

    let conv_json = r#"
    {
        "schema_version": "1.0",
        "model": {
            "name": "Conv-Comparison",
            "type": "cnn",
            "layers": [
                {"id": "c1", "layer_type": "conv", "params": {"in_channels": 100, "out_channels": 100, "kernel_size": 3}}
            ]
        },
        "training": {"batch_size": 1, "precision": "fp32"},
        "hardware": {"gpus": [{"name": "A100-80GB", "count": 1}]},
        "data": {"dtype": "fp32"}
    }
    "#;
    let conv = analyze_json(conv_json)
        .unwrap()
        .arch
        .metrics
        .total_parameters;

    assert_ne!(
        rgcn, conv,
        "an RGCN layer must not be costed identically to an unrelated Conv2D"
    );
}
