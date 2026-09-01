//! Regression test for a real bug found while auditing the "Simulate"
//! side: `gpu_utilization = compute_time_ms / latency_ms` used the
//! pre-division, whole-batch `compute_time_ms` in the numerator while
//! `latency_ms` was built from the per-GPU (divided by `num_gpus`)
//! figure. Existing coverage (`internal_coherence.rs`'s
//! `test_a22_gpu_utilization_range`) only exercised a single-GPU config,
//! where the bug is invisible — dividing by 1 changes nothing. Any
//! multi-GPU config inflated utilization by roughly `num_gpus`;
//! `examples/models/stable_diffusion_1.5.json` (8 GPUs) reported 486%.

use std::path::PathBuf;

#[test]
fn gpu_utilization_never_exceeds_100_percent_on_a_multi_gpu_config() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("has a parent")
        .join("examples/models/stable_diffusion_1.5.json");
    let json = std::fs::read_to_string(&path).expect("readable");
    let config = neurax_parser::parse_model_config(&json).expect("parses");
    let r = neurax_core::run_analysis(config).expect("analyzes");

    assert!(
        (0.0..=1.0).contains(&r.hardware.metrics.gpu_utilization),
        "gpu_utilization should be a fraction in [0, 1], got {} \
         (this model configures 8 GPUs — the bug scaled utilization by \
         roughly that factor)",
        r.hardware.metrics.gpu_utilization
    );
}
