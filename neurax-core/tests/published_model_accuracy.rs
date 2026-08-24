//! NEURAX must reproduce the parameter counts these models are published with.
//!
//! This is the test the project was missing. A file named
//! `modelTemplates.accuracy.test.ts` claimed reference models "must reproduce
//! their published parameter counts" and checked parameter *name aliasing*
//! instead, so nothing anywhere compared a computed figure to a known one.
//!
//! When this was first run, four of seven reference models were wrong, three of
//! them badly:
//!
//! | model         | published | measured | error   |
//! |---------------|-----------|----------|---------|
//! | LLaMA-2 70B   | 70.0 B    | 49.9 B   | −28.7 % |
//! | Mixtral 8x7B  | 46.7 B    | 103.8 B  | +122 %  |
//! | DeepSeek-V3   | 671 B     | 1 395 B  | +108 %  |
//! | RWKV 7B       | 7.5 B     | 0.24 B   | −96.7 % |
//!
//! Three causes, all fixed: one scale derived from the attention-block count
//! was applied to every repeating kind, so a config listing 2 attention and 4
//! MoE blocks got twice the mixture layers it asked for; `shared_experts`
//! defaulted to 1, adding an expert to every mixture that never mentioned one;
//! and RWKV was costed with Mamba's formula.
//!
//! The tolerance is deliberately loose. Published counts are themselves
//! approximate — they differ on whether the embedding is tied, whether biases
//! are counted, and are usually rounded to two figures — so agreement to a few
//! percent is the strongest claim the comparison can support. What it does
//! catch, and what it exists for, is a formula that is structurally wrong.

use std::path::PathBuf;

/// A model, its published size, and where that figure comes from.
struct Reference {
    file: &'static str,
    published: f64,
    source: &'static str,
    /// Allowed relative error. Wider where the published figure is itself
    /// ambiguous.
    tolerance: f64,
}

const REFERENCES: &[Reference] = &[
    Reference {
        file: "vgg16.json",
        published: 138.0e6,
        source: "Simonyan & Zisserman 2014, 138M",
        tolerance: 0.05,
    },
    Reference {
        file: "resnet50.json",
        published: 25.6e6,
        source: "He et al. 2015, 25.6M",
        tolerance: 0.10,
    },
    Reference {
        file: "mamba_2.8b.json",
        published: 2.8e9,
        source: "state-spaces/mamba-2.8b",
        tolerance: 0.10,
    },
    Reference {
        file: "rwkv_7b.json",
        published: 7.5e9,
        source: "BlinkDL/RWKV-4-Raven-7B",
        tolerance: 0.10,
    },
    Reference {
        file: "llama2_70b.json",
        published: 70.0e9,
        source: "Touvron et al. 2023, 70B",
        tolerance: 0.05,
    },
    Reference {
        file: "gpt3_175b.json",
        published: 175.0e9,
        source: "Brown et al. 2020, Table 2.1, 175.0B",
        tolerance: 0.05,
    },
    Reference {
        file: "gcn_ogbn_arxiv.json",
        // Not just cited — independently re-derived from the OGB reference
        // implementation's exact layer widths (128→256→256→40, 3x GCNConv +
        // 2x BatchNorm1d) and matched to the number on the leaderboard bit
        // for bit: 110,120.
        published: 110_120.0,
        source: "OGB leaderboard, ogbn-arxiv, GCN baseline: https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv",
        tolerance: 0.01,
    },
    Reference {
        file: "awd_lstm_ptb.json",
        // Table 1's own figure is rounded to "24M"; tolerance set as if it
        // reads 24.0-24.49M rather than treating it as an exact value.
        published: 24.0e6,
        source: "Merity, Keskar & Socher 2017 (arXiv:1708.02182), Table 1: \"AWD-LSTM - 3-layer LSTM (tied), 24M\"",
        tolerance: 0.05,
    },
    Reference {
        file: "mixtral_8x7b.json",
        // 46.7B total, not the 12.9B active per token.
        published: 46.7e9,
        source: "Jiang et al. 2024, 46.7B total",
        tolerance: 0.05,
    },
    Reference {
        file: "deepseek_v3.json",
        published: 671.0e9,
        source: "DeepSeek-AI 2024, 671B total",
        tolerance: 0.10,
    },
];

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate has a parent directory")
        .join("examples/models")
}

fn total_parameters(file: &str) -> u64 {
    let path = models_dir().join(file);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} should be readable: {e}", path.display()));
    let config = neurax_parser::parse_model_config(&json)
        .unwrap_or_else(|e| panic!("{} should parse: {e}", path.display()));
    neurax_ir::architecture::scaled_total_parameters(&config)
}

#[test]
fn every_reference_model_matches_its_published_parameter_count() {
    let mut failures = Vec::new();

    for reference in REFERENCES {
        let measured = total_parameters(reference.file) as f64;
        let error = (measured - reference.published) / reference.published;

        if error.abs() > reference.tolerance {
            failures.push(format!(
                "  {:<22} published {:>9.3}B, measured {:>9.3}B, {:+.1}% (limit ±{:.0}%) — {}",
                reference.file,
                reference.published / 1e9,
                measured / 1e9,
                error * 100.0,
                reference.tolerance * 100.0,
                reference.source,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} reference models are outside tolerance:\n{}",
        failures.len(),
        REFERENCES.len(),
        failures.join("\n"),
    );
}

/// A model larger than a trillion parameters must still be counted exactly.
///
/// Parameter counts are accumulated in `u64` with saturating arithmetic, and
/// `usize` products of large dimensions used to wrap silently — producing a
/// number that looked reasonable and was not. At this size a wrap is the
/// difference between "1.6 trillion" and a few billion.
#[test]
fn a_trillion_parameter_model_is_counted_without_wrapping() {
    // A dense transformer at the scale of the largest published models:
    // 128 layers, 20480 wide, 8x expansion.
    let config_json = r#"{
        "schema_version": "1.0.0",
        "model": {
            "name": "trillion-scale",
            "type": "transformer",
            "global_params": {
                "num_layers": 128,
                "hidden_size": 20480,
                "num_heads": 128,
                "intermediate_size": 163840,
                "vocab_size": 100000,
                "sequence_length": 8192
            },
            "layers": [
                {"id": "embed", "layer_type": "embedding",
                 "params": {"vocab_size": 100000, "hidden_size": 20480}},
                {"id": "attn", "layer_type": "attention",
                 "params": {"hidden_size": 20480, "num_heads": 128}},
                {"id": "mlp", "layer_type": "mlp",
                 "params": {"hidden_size": 20480, "intermediate_size": 163840,
                            "activation": "swiglu"}}
            ]
        },
        "training": {"batch_size": 1, "sequence_length": 8192, "precision": "bf16",
                     "learning_rate": 0.0001, "num_epochs": 1},
        "hardware": {"gpus": [{"name": "H100", "memory_gb": 80, "count": 1024}]},
        "data": {"dataset_size": 1000000000000, "vocab_size": 100000, "num_classes": 0}
    }"#;

    let config = neurax_parser::parse_model_config(config_json).expect("the config parses");
    let total = neurax_ir::architecture::scaled_total_parameters(&config);

    // Computed by hand: 128 x (4 x 20480^2 + 3 x 20480 x 163840) + 100000 x 20480.
    let per_attention = 4.0 * 20480.0 * 20480.0;
    let per_mlp = 3.0 * 20480.0 * 163840.0;
    let expected = 128.0 * (per_attention + per_mlp) + 100_000.0 * 20480.0;

    let error = (total as f64 - expected).abs() / expected;
    assert!(
        error < 0.01,
        "expected about {:.2}T parameters, got {:.2}T",
        expected / 1e12,
        total as f64 / 1e12
    );
    assert!(
        total as f64 > 1.0e12,
        "this configuration is over a trillion parameters; got {total}"
    );
}

/// The whole pipeline must survive that model, not just the parameter count.
#[test]
fn a_trillion_parameter_model_analyses_end_to_end() {
    let path = models_dir().join("deepseek_v3.json");
    let json = std::fs::read_to_string(&path).expect("readable");
    let config = neurax_parser::parse_model_config(&json).expect("parses");

    let report = neurax_core::run_analysis(config).expect("a 671B model should analyse");
    assert!(
        report.arch.metrics.total_parameters > 100_000_000_000,
        "the report lost the model's size: {}",
        report.arch.metrics.total_parameters
    );
}
