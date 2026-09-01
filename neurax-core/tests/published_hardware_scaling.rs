//! The "Simulate" equivalent of `published_model_accuracy.rs`: until this
//! test existed, nothing in the repository compared NEURAX's predicted
//! throughput/latency to a real, published, measured figure — every
//! hardware/cost test (`hardware_corrections.rs`, `internal_coherence.rs`
//! groups 4-5) only checks that a number is positive/finite/in a plausible
//! range, never against reality. `hardware_corrections.rs`'s own F06 tests
//! go further and assert literal "contention factor" constants (0.82 for 2
//! GPUs, 0.70 for 4, 0.55 for 8) that are compared only to themselves —
//! they are not wired into any real code path at all.
//!
//! Ground truth: Narayanan et al. 2021, "Efficient Large-Scale Language
//! Model Training on GPU Clusters Using Megatron-LM" (arXiv:2104.04473),
//! Table 2. For a fixed 174.6-billion-parameter GPT model (matching GPT-3
//! 175B's real architecture: 96 layers, hidden size 12288, 96 attention
//! heads) trained with ZeRO-3 and *no* model parallelism — the regime
//! NEURAX's own hardware pass models (pure data-parallel + an all-reduce
//! communication term) — at a fixed global batch size of 1536, the paper
//! reports achieved throughput *per GPU* falling as GPU count grows:
//!
//! | GPUs | achieved TFLOP/s per GPU |
//! |------|--------------------------|
//! | 384  | 144                      |
//! | 768  | 88                       |
//! | 1536 | 44                       |
//!
//! i.e. a real, measured ~3.3x degradation in per-GPU throughput across a
//! 4x increase in GPU count at fixed model size and batch — because the
//! gradient all-reduce cost stays roughly fixed per step while each GPU's
//! own compute share shrinks. The paper's own hardware (Table/§5:
//! "Each node has eight NVIDIA Mellanox 200Gbps HDR InfiniBand HCAs") is
//! used directly as this test's interconnect bandwidth (8 x 200 Gbit/s =
//! 200 GB/s), rather than an arbitrary guess.
//!
//! NEURAX cannot reproduce the exact figures — it models a single
//! effective ring-all-reduce bandwidth, not the paper's specific 3-level
//! fat-tree topology or ZeRO-3's exact parameter-gather traffic pattern —
//! so this test checks the property that actually matters for a design
//! tool: throughput per GPU must measurably *degrade* as GPU count grows
//! at fixed model size and batch, not stay flat or improve, and by a
//! non-trivial margin (not a rounding-error decline).

fn analyze(json: &str) -> neurax_core::AnalysisResult {
    let config = neurax_parser::parse_model_config(json).expect("parses");
    neurax_core::run_analysis(config).expect("analyzes")
}

/// The real GPT-3 175B architecture (matches examples/models/gpt3_175b.json
/// and Narayanan et al.'s own "174.6B" row), with ZeRO-3 and the paper's own
/// batch size (1536) and interconnect (200 GB/s, from the paper's stated
/// per-node InfiniBand HCA count) — everything held fixed except GPU count.
fn gpt3_175b_zero3_at(num_gpus: u32) -> String {
    format!(
        r#"{{
            "schema_version": "1.0.0",
            "model": {{
                "name": "GPT-3-175B",
                "type": "transformer",
                "global_params": {{
                    "num_layers": 96, "hidden_size": 12288, "num_heads": 96,
                    "intermediate_size": 49152, "vocab_size": 50257, "sequence_length": 2048
                }},
                "layers": [
                    {{"id": "embed", "layer_type": "embedding", "params": {{"vocab_size": 50257, "hidden_size": 12288}}}},
                    {{"id": "layer_0", "layer_type": "attention", "params": {{"hidden_size": 12288, "num_attention_heads": 96, "intermediate_size": 49152}}}},
                    {{"id": "mlp_0", "layer_type": "mlp", "params": {{"hidden_size": 12288, "intermediate_size": 49152}}}}
                ]
            }},
            "training": {{
                "batch_size": 1536, "sequence_length": 2048, "precision": "bf16",
                "learning_rate": 0.00001, "num_epochs": 300, "zero_stage": 3
            }},
            "hardware": {{
                "gpus": [{{"name": "A100-SXM", "count": {num_gpus}}}],
                "interconnect_bandwidth_gb_s": 200.0
            }},
            "data": {{"dataset_size": 500000000000, "vocab_size": 50257, "num_classes": 0}}
        }}"#
    )
}

#[test]
fn model_reproduces_the_real_174b_architecture() {
    let r = analyze(&gpt3_175b_zero3_at(384));
    // Narayanan et al. Table 2's own row: "174.6" billion parameters.
    let error = (r.arch.metrics.total_parameters as f64 - 174.6e9).abs() / 174.6e9;
    assert!(
        error < 0.01,
        "expected ~174.6B params (Narayanan et al. Table 2), got {}",
        r.arch.metrics.total_parameters
    );
}

#[test]
fn per_gpu_throughput_degrades_as_gpu_count_grows_at_fixed_model_and_batch() {
    let at_384 = analyze(&gpt3_175b_zero3_at(384));
    let at_768 = analyze(&gpt3_175b_zero3_at(768));
    let at_1536 = analyze(&gpt3_175b_zero3_at(1536));

    let per_gpu = |r: &neurax_core::AnalysisResult, n: f64| r.hardware.metrics.effective_tflops / n;
    let tflops_384 = per_gpu(&at_384, 384.0);
    let tflops_768 = per_gpu(&at_768, 768.0);
    let tflops_1536 = per_gpu(&at_1536, 1536.0);

    assert!(
        tflops_768 < tflops_384 && tflops_1536 < tflops_768,
        "per-GPU throughput should monotonically degrade as GPU count grows \
         at fixed model size/batch (Narayanan et al.: 144 -> 88 -> 44 TFLOP/s/GPU), \
         got {tflops_384:.1} -> {tflops_768:.1} -> {tflops_1536:.1}"
    );

    // The real measured degradation across a 4x GPU increase is ~3.3x
    // (144/44). NEURAX's simpler single-bandwidth ring-all-reduce model
    // isn't expected to match that exactly, but the effect must be
    // substantial, not a rounding error — anything under ~20% would mean
    // the communication term is not meaningfully engaging at this scale.
    let degradation_4x = tflops_384 / tflops_1536;
    assert!(
        degradation_4x > 1.2,
        "expected a non-trivial per-GPU throughput drop across a 4x GPU increase \
         (real data: ~3.3x), got only {degradation_4x:.2}x"
    );
}
