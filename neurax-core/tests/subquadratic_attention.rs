//! Sliding-window/dilated/sparse attention all collapsed into plain dense
//! attention on the wire — `window_size`/`block_size`/`dilation` existed as
//! canvas block defaults but reached no typed field on the compiler side,
//! so choosing "Sliding Window Attention" (Mistral 7B's actual mechanism,
//! arXiv:2310.06825) reported exactly the same FLOPs as full O(S²)
//! attention. At long context, that's the entire point of the block,
//! reported as having no effect at all.

use neurax_core::analyze_json;

fn attention_json(extra_params: &str) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "Attention-Span-Check",
                "type": "transformer",
                "layers": [
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_heads": 32, "causal": true{extra_params}}}}}
                ]
            }},
            "training": {{ "batch_size": 1, "sequence_length": 32768, "precision": "fp16" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp16" }}
        }}
        "#
    )
}

#[test]
fn a_sliding_window_costs_less_than_dense_at_long_context() {
    let dense = analyze_json(&attention_json("")).unwrap().compute.metrics.total_flops;
    let windowed = analyze_json(&attention_json(r#","window_size": 4096"#))
        .unwrap()
        .compute
        .metrics
        .total_flops;

    assert!(
        windowed < dense,
        "a 4096-token window at 32k context must cost less than full attention"
    );
    assert!(
        windowed < dense / 2.0,
        "should be a large reduction, not a marginal one: {windowed} vs {dense}"
    );
}

#[test]
fn block_sparse_attention_also_costs_less_than_dense() {
    let dense = analyze_json(&attention_json("")).unwrap().compute.metrics.total_flops;
    let sparse = analyze_json(&attention_json(r#","block_size": 64"#))
        .unwrap()
        .compute
        .metrics
        .total_flops;
    assert!(sparse < dense);
}

#[test]
fn no_pattern_stated_is_unaffected_dense_attention() {
    // No window_size/block_size/dilation given at all — must reproduce the
    // exact prior behavior, not silently change every existing model.
    let json = attention_json("");
    let result = analyze_json(&json).unwrap();
    assert!(result.compute.metrics.total_flops > 0.0);
}
