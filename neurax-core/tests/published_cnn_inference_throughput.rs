//! Second external "Simulate" reference, in a deliberately different regime
//! from `published_hardware_scaling.rs` (which covers multi-GPU training
//! scaling for a transformer): single-GPU, inference, convolution-bound.
//!
//! Ground truth: NVIDIA's own published ResNet-50 v1.5 inference
//! performance numbers (NGC "ResNet v1.5 for TensorFlow" performance page),
//! a single A100 40GB GPU, mixed precision (fp16), batch size 256:
//! **3,229.32 images/sec**.
//!
//! Before this reference existed, NEURAX predicted ~23,286 img/s for the
//! same config — about 7.2x too fast. Root cause: `HardwarePass`'s
//! operation-type efficiency weighting only recognized `Attention` and
//! `Mlp` layers; every convolution-family type (`Conv`,
//! `ResnetBottleneck`, ...) fell outside that count, so a pure-CNN model
//! had zero recognized layers and used the 1.0 "no penalty" neutral
//! fallback meant for genuinely unrecognized architectures — not a
//! deliberate choice to model convolutions as running at full transformer-
//! matmul efficiency. Real convolutions are many small kernels (conv +
//! batchnorm + activation) rather than one large GEMM, which is exactly
//! why they achieve a much smaller fraction of theoretical peak.
//!
//! Fixed by adding a `conv_efficiency` term (hardware/pass.rs), calibrated
//! against this exact reference: matching NEURAX's own forward-FLOPs
//! figure for ResNet-50 to the real 3,229 img/s implies ~0.139 effective
//! efficiency at fp16. This test is therefore partly a *calibration lock*
//! (the constant was derived from this exact number, so close agreement
//! here is expected almost by construction) rather than a fully
//! independent validation — `published_hardware_scaling.rs`'s multi-GPU
//! reference is the independent one. Its value here is guarding against
//! silent drift: if `conv_efficiency`, the ResNet-50 FLOPs formula, or the
//! GPU database's A100 specs regress, this is what would catch it.

use std::path::PathBuf;

fn resnet50_at_batch_256_fp16_single_a100() -> neurax_core::AnalysisResult {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent")
        .join("examples/models/resnet50.json");
    let json = std::fs::read_to_string(&path).expect("resnet50.json should be readable");
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    // examples/models/resnet50.json already uses batch_size 256 (matching
    // this reference); override only what the NGC benchmark's config
    // actually differs on: precision and a single A100 40GB instead of
    // 8x V100.
    value["training"]["precision"] = serde_json::json!("fp16");
    value["hardware"]["gpus"] =
        serde_json::json!([{"name": "A100-40GB", "memory_gb": 40, "count": 1}]);

    let mutated = serde_json::to_string(&value).expect("re-serializes");
    let config = neurax_parser::parse_model_config(&mutated).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

#[test]
fn resnet50_single_a100_throughput_is_within_calibration_tolerance() {
    let r = resnet50_at_batch_256_fp16_single_a100();

    const REAL_IMG_PER_S: f64 = 3229.32;
    let predicted = r.hardware.metrics.samples_per_s;
    let error = (predicted - REAL_IMG_PER_S).abs() / REAL_IMG_PER_S;

    assert!(
        error < 0.20,
        "predicted {predicted:.1} img/s vs NVIDIA's published {REAL_IMG_PER_S} img/s \
         (single A100 40GB, mixed precision, batch 256) — {:.1}% off, outside the \
         ±20% calibration tolerance",
        error * 100.0
    );
}

#[test]
fn resnet50_throughput_is_no_longer_an_order_of_magnitude_too_fast() {
    // Regression guard for the specific failure mode: before conv_efficiency
    // existed, a pure-CNN model fell back to the 1.0 neutral efficiency and
    // predicted throughput ~7x too high. Assert this can't silently
    // regress back to "no CNN-specific penalty applied at all", without
    // pinning to the exact calibrated value (covered by the test above).
    let r = resnet50_at_batch_256_fp16_single_a100();
    assert!(
        r.hardware.metrics.samples_per_s < 10_000.0,
        "ResNet-50 on a single A100 predicting {:.0} img/s looks like the old \
         uncalibrated (1.0 efficiency) behavior, not a convolution-aware one",
        r.hardware.metrics.samples_per_s
    );
}
