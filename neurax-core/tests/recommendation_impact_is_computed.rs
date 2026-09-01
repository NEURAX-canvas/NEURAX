//! Regression tests for the "Optimize" pillar's flagship output —
//! `generate_recommendations()` — whose impact estimates used to be
//! hardcoded strings unrelated to the model actually analyzed ("Save ~70%
//! on GPU costs", "Improve efficiency to ~90%", "Potential 2-3x speedup"),
//! rather than values derived from that model's own metrics. A 4-layer
//! model and a 96-layer one received the identical "gradient checkpointing"
//! savings estimate despite the real technique scaling very differently
//! with depth.
//!
//! Each test here proves a specific recommendation's number now reacts to
//! the model it was generated for, rather than being a fixed constant.

fn model_json(num_layers: u32, batch_size: u32) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{ "name": "test", "type": "transformer",
                "global_params": {{
                    "num_layers": {num_layers}, "hidden_size": 4096, "num_heads": 32,
                    "intermediate_size": 16384, "vocab_size": 32000, "sequence_length": 4096
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 32000, "hidden_size": 4096}}}},
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 16384}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 4096, "intermediate_size": 16384}}}}
                ]
            }},
            "training": {{"batch_size": {batch_size}, "sequence_length": 4096, "precision": "bf16", "max_steps": 100}},
            "hardware": {{"gpus": [{{"name": "A100-SXM", "count": 1}}]}},
            "data": {{"dataset_size": 1000000000, "vocab_size": 32000, "num_classes": 0}}
        }}"#
    )
}

fn analyze(num_layers: u32, batch_size: u32) -> neurax_core::AnalysisResult {
    let config =
        neurax_parser::parse_model_config(&model_json(num_layers, batch_size)).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

fn checkpointing_recommendation(r: &neurax_core::AnalysisResult) -> Option<String> {
    r.report
        .recommendations
        .iter()
        .find(|rec| rec.title == "Enable Gradient Checkpointing")
        .map(|rec| rec.impact.clone())
}

#[test]
fn gradient_checkpointing_savings_scale_with_real_depth_not_a_flat_constant() {
    // Both models are pushed over the 80% VRAM threshold (same batch), but
    // differ only in depth. The real technique (Chen et al. 2016) frees
    // 1 - 1/sqrt(L) of activation memory — a shallow model should show a
    // meaningfully smaller savings percentage than a deep one, not the
    // same flat estimate.
    let shallow = analyze(16, 128);
    let deep = analyze(96, 128);

    let shallow_impact = checkpointing_recommendation(&shallow)
        .expect("shallow model should trigger the memory recommendation");
    let deep_impact = checkpointing_recommendation(&deep)
        .expect("deep model should trigger the memory recommendation");

    assert_ne!(
        shallow_impact, deep_impact,
        "a 16-layer and a 96-layer model got the identical gradient-checkpointing \
         savings estimate — the real sqrt(L) technique should scale with depth"
    );

    // 16 layers: 1 - 1/sqrt(16) = 75%. 96 layers: 1 - 1/sqrt(96) ~= 89.8%.
    assert!(
        shallow_impact.contains("75%"),
        "expected ~75% for a 16-layer model, got: {shallow_impact}"
    );
    assert!(
        deep_impact.contains("90%"),
        "expected ~90% for a 96-layer model, got: {deep_impact}"
    );
}
