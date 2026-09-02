//! Real bug found auditing 100B+ models: `optimal_gpu_count` used to just
//! echo the configured GPU count back (`num_gpus` in
//! `neurax-ir/src/parallelism/pass.rs`), not a computed recommendation.
//! A 405B-parameter model configured with a single GPU reported
//! `optimal_gpu_count: 1` in the same breath as `oom_risk: Overflow` — a
//! caller reading that field alone, without cross-checking `oom_risk`,
//! would conclude "1 GPU is optimal" for a model that cannot run on one.

use neurax_core::analyze_json;
use neurax_ir::memory::OomRisk;

fn llama_405b_shaped_json(gpu_count: u32) -> String {
    format!(
        r#"{{
            "schema_version": "1.0",
            "model": {{
                "name": "llama-3.1-405b-shaped",
                "type": "transformer",
                "global_params": {{ "num_layers": 126, "hidden_size": 16384, "sequence_length": 4096, "num_heads": 128, "num_kv_heads": 8 }},
                "layers": [
                    {{"id": "attn", "layer_type": "attention", "params": {{"hidden_size": 16384, "num_heads": 128, "num_kv_heads": 8}}}},
                    {{"id": "mlp", "layer_type": "mlp", "params": {{"hidden_size": 16384, "intermediate_size": 53248}}}}
                ]
            }},
            "training": {{"batch_size": 1, "precision": "bf16"}},
            "hardware": {{"gpus": [{{"name": "A100", "count": {gpu_count}}}]}}
        }}"#
    )
}

#[test]
fn under_provisioned_config_no_longer_reports_its_own_shortfall_as_optimal() {
    let under_provisioned = analyze_json(&llama_405b_shaped_json(1)).expect("analyzes");
    assert_eq!(under_provisioned.memory.metrics.oom_risk, OomRisk::Overflow);
    assert!(
        under_provisioned.parallelism.metrics.optimal_gpu_count > 1,
        "a model that overflows a single GPU must not report 1 as optimal, got {}",
        under_provisioned.parallelism.metrics.optimal_gpu_count
    );
}

#[test]
fn adequately_provisioned_config_keeps_its_own_gpu_count() {
    let well_provisioned = analyze_json(&llama_405b_shaped_json(16)).expect("analyzes");
    // 16 real GPUs is at least the configured count — never suggest
    // fewer than what's already there and sufficient.
    assert!(well_provisioned.parallelism.metrics.optimal_gpu_count >= 16);
}
