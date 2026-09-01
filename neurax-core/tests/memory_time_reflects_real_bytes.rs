//! Regression test for a real bug found while auditing the "Simulate" side
//! of NEURAX: `HardwarePass::calculate_layer_timings` estimated every op's
//! memory time as `compute_time_ms * 0.5` — a flat ratio of the *compute*
//! time, not a function of the op's actual memory traffic at all. Two
//! models with the same FLOPs but wildly different weight footprints (a
//! tiny embedding table vs a huge one) reported the exact same memory
//! time, latency, throughput and cost.
//!
//! The operator pass already computes each op's real
//! `param_count`/`activation_memory` (`AtomOp`), but `ComputePass` dropped
//! both fields when building `OpFlops`, leaving `HardwarePass` with
//! nothing but the flat guess to fall back on. Fixed by carrying a real
//! `bytes_accessed` figure through `OpFlops` and dividing it by the GPU's
//! actual HBM bandwidth instead.

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

fn model_with_vocab(vocab_size: u64) -> String {
    format!(
        r#"{{
            "schema_version": "1.0",
            "model": {{ "name": "vocab-test", "type": "transformer",
                "global_params": {{ "num_layers": 1, "hidden_size": 512, "sequence_length": 128 }},
                "layers": [
                    {{"id": "emb", "layer_type": "embedding", "params": {{"vocab_size": {vocab_size}, "embedding_dim": 512}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 512, "intermediate_size": 2048}}}}
                ]
            }},
            "training": {{"batch_size": 1, "max_steps": 100, "precision": "fp16"}},
            "hardware": {{"gpus": [{{"name": "A100-SXM", "count": 1}}]}}
        }}"#
    )
}

#[test]
fn a_much_larger_weight_footprint_costs_meaningfully_more_latency() {
    let small = analyze(&model_with_vocab(100));
    let huge = analyze(&model_with_vocab(500_000));

    assert!(
        huge.hardware.metrics.latency_ms > small.hardware.metrics.latency_ms * 2.0,
        "a 5000x larger embedding table (same FLOPs) should cost meaningfully \
         more latency once memory time reacts to real bytes — got small={}, huge={}",
        small.hardware.metrics.latency_ms,
        huge.hardware.metrics.latency_ms
    );
}

#[test]
fn memory_time_is_not_a_flat_fraction_of_compute_time() {
    // Regression guard for the specific bug shape: assert the ratio between
    // two models' memory-derived latency does NOT track their FLOPs ratio
    // (which is what a `compute_time * constant` formula would produce).
    let small = analyze(&model_with_vocab(100));
    let huge = analyze(&model_with_vocab(500_000));

    let flops_ratio =
        huge.compute.metrics.forward_flops / small.compute.metrics.forward_flops.max(1.0);
    let latency_ratio =
        huge.hardware.metrics.latency_ms / small.hardware.metrics.latency_ms.max(1e-9);

    assert!(
        latency_ratio > flops_ratio * 5.0,
        "latency should scale with the real byte footprint, not track the \
         (nearly identical) FLOPs ratio — flops_ratio={flops_ratio:.3}, \
         latency_ratio={latency_ratio:.3}"
    );
}
