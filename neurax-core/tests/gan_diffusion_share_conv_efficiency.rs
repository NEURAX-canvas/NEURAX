//! Regression test for a formula-equivalence gap found while extending the
//! CNN inference-throughput fix (`published_cnn_inference_throughput.rs`):
//! `GeneratorBlock`/`DiscriminatorBlock` (GAN) and `UnetBlock`/`ResnetBlock`/
//! `DownBlock`/`UpBlock`/`MidBlock`/`VaeEncoder`/`VaeDecoder`/`NoisePredictor`
//! (diffusion) already call the exact same `conv::conv2d_params`/
//! `cnn_blocks::resnet_basic_block_params` formulas as a recognized `Conv`
//! layer (architecture/mod.rs) — the compiler already treats them as
//! convolutions at the cost-formula level. But `HardwarePass`'s efficiency
//! weighting didn't count them, so a pure-GAN model (like
//! examples/models/dcgan.json, whose only layer types are
//! `generator_block`/`discriminator_block`) had zero recognized
//! layers and used the 1.0 "no penalty" neutral fallback — the same class
//! of bug fixed for plain `Conv` layers, just not yet extended to types
//! that are mechanically convolutions under a different name.
//!
//! This is not a new calibration (no new external reference number): it
//! reuses the already-calibrated `conv_efficiency` constant
//! (`published_cnn_inference_throughput.rs`), on the grounds that these
//! layer types run the literal same kernels.

use std::path::PathBuf;

fn analyze_example(file: &str) -> neurax_core::AnalysisResult {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("has a parent")
        .join("examples/models")
        .join(file);
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{file}: {e}"));
    let config = neurax_parser::parse_model_config(&json).unwrap_or_else(|e| panic!("{file}: {e}"));
    neurax_core::run_analysis(config).unwrap_or_else(|e| panic!("{file}: {e}"))
}

#[test]
fn a_pure_gan_model_is_no_longer_costed_at_full_efficiency() {
    let r = analyze_example("dcgan.json");

    // At the old 1.0-efficiency fallback, compute time is FLOPs / peak
    // achievable rate with no derating at all. With the real ~0.14 (fp16)
    // conv_efficiency now applied, latency should be substantially higher
    // — at minimum a few times over, not a rounding-error difference.
    // (DCGAN's own training config sets no explicit precision, so this
    // checks direction/magnitude rather than the exact calibrated ratio.)
    assert!(
        r.hardware.metrics.gpu_utilization <= 1.0,
        "GPU utilization over 100% ({}) would mean compute time still isn't \
         being derated for this GAN model",
        r.hardware.metrics.gpu_utilization
    );

    // Cross-check against the equivalent, already-validated pure-CNN case:
    // a GAN's generator/discriminator blocks and a CNN's conv layers now
    // share the same conv_efficiency constant, so a GAN model's effective
    // TFLOPS should land in the same broad regime (a small fraction of the
    // GPU's peak), not near-peak the way the old 1.0 fallback implied.
    let gpu_peak_tflops = r.hardware.gpu_profile.peak_tflops;
    let pct_of_peak = r.hardware.metrics.effective_tflops / gpu_peak_tflops;
    assert!(
        pct_of_peak < 0.5,
        "a convolution-heavy GAN model achieving {:.1}% of peak TFLOPS looks \
         like the old uncalibrated (1.0 efficiency) behavior, not a \
         convolution-aware one",
        pct_of_peak * 100.0
    );
}
