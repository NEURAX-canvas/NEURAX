//! S4Block/H3Block/StateSpace shared `mamba_flops` — Mamba's whole
//! innovation is *selectivity* (B/C/Δ are outputs of input-dependent
//! projections, Gu & Dao, 2023); S4 (Gu et al., 2021) is explicitly
//! time-invariant, with no such projection cost, and H3 (Fu et al., 2023)
//! is a distinct two-SSM-layer structure. `s4_flops`/`h3_flops` already
//! existed in neurax-formulas with the right shape but were never called
//! from the live pipeline.

use neurax_core::analyze_json;

fn ssm_json(layer_type: &str) -> String {
    format!(
        r#"
        {{
            "schema_version": "1.0",
            "model": {{
                "name": "SSM-Formula-Check",
                "type": "ssm",
                "layers": [
                    {{"id": "block1", "layer_type": "{layer_type}", "params": {{"hidden_size": 2048, "state_dim": 16, "expansion_factor": 2}}}}
                ]
            }},
            "training": {{ "batch_size": 1, "sequence_length": 2048, "precision": "fp16" }},
            "hardware": {{ "gpus": [{{ "name": "A100-80GB", "count": 1 }}] }},
            "data": {{ "dtype": "fp16" }}
        }}
        "#
    )
}

#[test]
fn s4_and_mamba_report_different_flops_for_the_same_shape() {
    let mamba = analyze_json(&ssm_json("mamba_block"))
        .unwrap()
        .compute
        .metrics
        .total_flops;
    let s4 = analyze_json(&ssm_json("s4_block"))
        .unwrap()
        .compute
        .metrics
        .total_flops;
    assert_ne!(
        mamba, s4,
        "S4 (non-selective) should not report the identical FLOPs Mamba (selective) does"
    );
}

#[test]
fn h3_and_mamba_report_different_flops_for_the_same_shape() {
    let mamba = analyze_json(&ssm_json("mamba_block"))
        .unwrap()
        .compute
        .metrics
        .total_flops;
    let h3 = analyze_json(&ssm_json("h3_block"))
        .unwrap()
        .compute
        .metrics
        .total_flops;
    assert_ne!(mamba, h3);
}

#[test]
fn all_three_still_compile_to_nonzero_real_numbers() {
    for layer_type in ["mamba_block", "s4_block", "h3_block", "state_space"] {
        let flops = analyze_json(&ssm_json(layer_type))
            .unwrap()
            .compute
            .metrics
            .total_flops;
        assert!(flops > 0.0, "{layer_type} should report nonzero FLOPs");
    }
}
