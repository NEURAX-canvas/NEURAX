//! Inference behaviour that genuinely depends on the model must depend on it.
//!
//! The pass used to take sampling parameters alone, so it assumed a 32,768-token
//! window and an eight-expert top-2 router for every model. Two very different
//! designs sampled identically produced identical reports, which made the
//! answers describe no model in particular.

use neurax_ir::inference::{InferenceParams, InferencePass, ModelProfile};

fn params(prompt: u32, output: u32) -> InferenceParams {
    InferenceParams {
        prompt_length: prompt,
        max_output_tokens: output,
        ..Default::default()
    }
}

/// A 7B model with grouped-query attention and a 4k window.
fn small_model() -> ModelProfile {
    ModelProfile {
        total_parameters: Some(7_000_000_000),
        num_layers: Some(32),
        hidden_size: Some(4096),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        trained_context: Some(4096),
        dtype_bytes: Some(2),
        ..Default::default()
    }
}

#[test]
fn context_degradation_uses_the_model_s_own_window() {
    // The same 8k request is comfortable on a 128k model and impossible on a 4k
    // one. Against a fixed 32k assumption both looked the same.
    let p = params(6000, 2000);

    let narrow = ModelProfile {
        trained_context: Some(4096),
        ..small_model()
    };
    let wide = ModelProfile {
        trained_context: Some(131_072),
        ..small_model()
    };

    let narrow_left = InferencePass::run_with_model(&p, Some(&narrow)).context_degradation;
    let wide_left = InferencePass::run_with_model(&p, Some(&wide)).context_degradation;

    assert!(
        narrow_left < wide_left,
        "a 4k model should have less headroom than a 128k one for the same request \
         ({narrow_left} vs {wide_left})"
    );
}

#[test]
fn hallucination_risk_accounts_for_model_capacity() {
    let p = params(1024, 512);
    let tiny = ModelProfile {
        total_parameters: Some(100_000_000),
        ..small_model()
    };
    let large = ModelProfile {
        total_parameters: Some(400_000_000_000),
        ..small_model()
    };

    let tiny_confidence = InferencePass::run_with_model(&p, Some(&tiny))
        .hallucination_risk
        .confidence;
    let large_confidence = InferencePass::run_with_model(&p, Some(&large))
        .hallucination_risk
        .confidence;

    assert!(
        tiny_confidence < large_confidence,
        "a 100M model should not be as trustworthy as a 400B one at identical sampling \
         ({tiny_confidence} vs {large_confidence})"
    );
}

#[test]
fn kv_cache_is_computed_from_the_model_s_shape() {
    let p = params(1024, 1024);
    let report = InferencePass::run_with_model(&p, Some(&small_model()));
    let kv = report
        .kv_cache
        .expect("a model with layers and heads has a KV cache");

    // 2 (K and V) x 32 layers x 8 kv heads x 128 head dim x 2 bytes = 131,072 B/token.
    assert_eq!(kv.bytes_per_token, 2 * 32 * 8 * 128 * 2);
    assert_eq!(kv.bytes_total, kv.bytes_per_token * 2048);
    // 32 query heads against 8 KV heads is a fourfold saving.
    assert!((kv.gqa_savings_factor - 4.0).abs() < 1e-9);
}

#[test]
fn kv_cache_is_absent_rather_than_guessed_when_the_model_is_unknown() {
    let report = InferencePass::run(&params(1024, 1024));
    assert!(
        report.kv_cache.is_none(),
        "without a model there is nothing to compute a KV cache from"
    );
    assert!(report.model_profile.is_none());
}

#[test]
fn router_load_follows_the_declared_experts() {
    let p = InferenceParams {
        architecture_family: "moe".to_string(),
        moe_router_mode: Some("top-k".to_string()),
        ..Default::default()
    };
    let mixtral_like = ModelProfile {
        num_experts: Some(8),
        top_k: Some(2),
        ..small_model()
    };
    let wide_router = ModelProfile {
        num_experts: Some(64),
        top_k: Some(1),
        ..small_model()
    };

    let a = InferencePass::run_with_model(&p, Some(&mixtral_like))
        .router_stability
        .expect("MoE reports router stability");
    let b = InferencePass::run_with_model(&p, Some(&wide_router))
        .router_stability
        .expect("MoE reports router stability");

    assert_eq!(
        a.distribution.len(),
        8,
        "eight declared experts, eight entries"
    );
    assert_eq!(
        b.distribution.len(),
        64,
        "sixty-four declared experts, sixty-four entries"
    );

    // Load must still add up.
    for dist in [&a.distribution, &b.distribution] {
        let total: f64 = dist.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "router load should sum to 1, got {total}"
        );
    }
}

#[test]
fn a_report_without_a_model_still_works() {
    // Sampling-only callers must keep working unchanged.
    let report = InferencePass::run(&params(2048, 1024));
    assert!(report.stability_index.score >= 0.0);
    assert_eq!(report.entropy_evolution.len(), 20);
}

#[test]
fn the_report_says_which_model_it_describes() {
    let report = InferencePass::run_with_model(&params(1024, 512), Some(&small_model()));
    let profile = report
        .model_profile
        .expect("the model should be echoed back");
    assert_eq!(profile.total_parameters, Some(7_000_000_000));
    assert_eq!(profile.trained_context, Some(4096));
}
