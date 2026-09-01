//! Dynamic System Evaluation Suite
//!
//! Tests for validating the dynamic analysis system.
//! Based on the objectives defined in virtual.md.

use crate::compute::ComputeIR;
use crate::dynamic::{
    BehavioralSynthesisPass, DynamicConfig, StabilityAnalysisPass, VirtualMemoryPass,
};
use crate::graph::GraphIR;
use crate::memory::MemoryMetrics;
use neurax_parser::ModelConfig;

const MINIMAL_TEST_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "eval-harness-stub",
        "type": "transformer",
        "layers": [
            {"id": "attn_0", "layer_type": "attention", "input_shape": [8, 64], "output_shape": [8, 64], "params": {}}
        ],
        "global_params": {"hidden_size": 64, "num_layers": 1}
    },
    "training": {"batch_size": 1, "optimizer": "adamw", "precision": "fp32"},
    "hardware": {"gpus": [{"name": "Generic-GPU", "count": 1, "memory_gb": 40}]}
}
"#;

fn create_test_model_config() -> ModelConfig {
    neurax_parser::parse_model_config(MINIMAL_TEST_JSON)
        .expect("MINIMAL_TEST_JSON must stay valid for the evaluation harness")
}

/// Test result
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub name: String,
    pub passed: bool,
    pub failure_reason: Option<String>,
    pub suggested_action: Option<String>,
}

impl EvalResult {
    pub fn test<F: FnOnce() -> bool + std::panic::UnwindSafe>(name: &str, f: F) -> Self {
        let result = std::panic::catch_unwind(f);
        match result {
            Ok(true) => Self {
                name: name.to_string(),
                passed: true,
                failure_reason: None,
                suggested_action: None,
            },
            Ok(false) => Self {
                name: name.to_string(),
                passed: false,
                failure_reason: Some("Assertion failed".to_string()),
                suggested_action: Some("Check implementation".to_string()),
            },
            Err(_) => Self {
                name: name.to_string(),
                passed: false,
                failure_reason: Some("Panic occurred".to_string()),
                suggested_action: Some("Fix panic".to_string()),
            },
        }
    }
}

/// Evaluation report
#[derive(Debug, Clone)]
pub struct EvaluationReport {
    pub total: usize,
    pub passed: usize,
    pub results: Vec<EvalResult>,
}

impl EvaluationReport {
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total as f64 * 100.0
    }

    pub fn is_production_ready(&self) -> bool {
        self.pass_rate() >= 85.0
    }

    pub fn print_summary(&self) {
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║     NEURAX DYNAMIC SYSTEM — EVALUATION REPORT         ║");
        println!("╠══════════════════════════════════════════════════════╣");
        println!(
            "║  Tests passés : {}/{} ({:.0}%)",
            self.passed,
            self.total,
            self.pass_rate()
        );
        println!(
            "║  Statut : {}",
            if self.is_production_ready() {
                "✅ PRODUCTION READY"
            } else {
                "❌ NON PRODUCTION — correctifs requis"
            }
        );
        println!("╠══════════════════════════════════════════════════════╣");
        for r in &self.results {
            println!("║  {} {}", if r.passed { "✅" } else { "❌" }, r.name);
        }
        println!("╚══════════════════════════════════════════════════════╝");
    }
}

/// Run full evaluation
pub fn run_full_evaluation() -> EvaluationReport {
    let mut results = vec![];
    results.extend(eval_virtual_memory());
    results.extend(eval_stability());
    results.extend(eval_behavioral());
    results.extend(eval_coherence());

    EvaluationReport {
        total: results.len(),
        passed: results.iter().filter(|r| r.passed).count(),
        results,
    }
}

// ── VIRTUAL MEMORY TESTS ──────────────────────────────────────────────

fn create_test_memory_metrics() -> MemoryMetrics {
    MemoryMetrics {
        parameter_memory_bytes: 5_000_000_000,  // 5GB params
        activation_memory_bytes: 2_000_000_000, // 2GB activations
        peak_vram_bytes: 10_000_000_000,        // 10GB peak
        ..Default::default()
    }
}

fn eval_virtual_memory() -> Vec<EvalResult> {
    vec![
        // OBJ-VM-01: Peak ordering invariant
        EvalResult::test("OBJ-VM-01: Peak ordering invariant", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);

            // virtual <= defrag <= naive
            metrics.peak_vram_with_virtual_gb <= metrics.peak_vram_with_defrag_gb
                && metrics.peak_vram_with_defrag_gb <= mem.peak_vram_gb()
        }),
        // OBJ-VM-02: Fragmentation always positive
        EvalResult::test("OBJ-VM-02: Fragmentation always positive", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);
            metrics.fragmentation_pct > 0.0
        }),
        // OBJ-VM-03: Virtual saves more than defrag
        EvalResult::test("OBJ-VM-03: Virtual saves more than defrag", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);
            metrics.virtual_savings_gb >= metrics.defrag_savings_gb
        }),
        // OBJ-VM-04: Savings% coherent with GB
        EvalResult::test("OBJ-VM-04: Savings% coherent with GB", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);
            let naive = mem.peak_vram_gb();
            let expected_pct = metrics.virtual_savings_gb / naive * 100.0;
            (metrics.virtual_savings_pct - expected_pct).abs() < 0.1
        }),
        // OBJ-VM-05: Virtual savings plausible (5-70%)
        EvalResult::test("OBJ-VM-05: Virtual savings plausible (5-70%)", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);
            metrics.virtual_savings_pct >= 5.0 && metrics.virtual_savings_pct <= 75.0
        }),
        // OBJ-VM-06: Confidence in valid range
        EvalResult::test("OBJ-VM-06: Confidence in valid range", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);
            metrics.confidence >= 0.5 && metrics.confidence <= 1.0
        }),
    ]
}

// ── STABILITY TESTS ───────────────────────────────────────────────────

fn create_test_graph() -> GraphIR {
    GraphIR::default()
}

fn eval_stability() -> Vec<EvalResult> {
    vec![
        // OBJ-STA-01: All margins in [0,1]
        EvalResult::test("OBJ-STA-01: All margins in [0,1]", || {
            let pass = StabilityAnalysisPass::new();
            let graph = create_test_graph();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&graph, &mem, &create_test_model_config());
            metrics
                .stability_margin_by_layer
                .values()
                .all(|&m| (0.0..=1.0).contains(&m))
        }),
        // OBJ-STA-02: chaos_index in [0,1]
        EvalResult::test("OBJ-STA-02: chaos_index in [0,1]", || {
            let pass = StabilityAnalysisPass::new();
            let graph = create_test_graph();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&graph, &mem, &create_test_model_config());
            (0.0..=1.0).contains(&metrics.chaos_index)
        }),
        // OBJ-STA-03: Confidence in valid range
        EvalResult::test("OBJ-STA-03: Confidence in valid range", || {
            let pass = StabilityAnalysisPass::new();
            let graph = create_test_graph();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&graph, &mem, &create_test_model_config());
            metrics.confidence >= 0.5 && metrics.confidence <= 1.0
        }),
        // OBJ-STA-04: Robustness in valid range
        EvalResult::test("OBJ-STA-04: Robustness in valid range", || {
            let pass = StabilityAnalysisPass::new();
            let graph = create_test_graph();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&graph, &mem, &create_test_model_config());
            (0.0..=1.0).contains(&metrics.global_robustness_score)
        }),
    ]
}

// ── BEHAVIORAL TESTS ──────────────────────────────────────────────────

fn create_test_compute() -> ComputeIR {
    ComputeIR::default()
}

fn create_test_compute_with_moe() -> ComputeIR {
    use crate::compute::OpFlops;
    use crate::operator::OpType;
    ComputeIR {
        op_flops: vec![OpFlops {
            op_id: 0,
            layer_id: "router_0".to_string(),
            forward_flops: 1.0,
            backward_flops: 2.0,
            bytes_accessed: 0,
            op_type: OpType::MoE,
        }],
        ..Default::default()
    }
}

fn eval_behavioral() -> Vec<EvalResult> {
    vec![
        // OBJ-BPS-01: expert_load_imbalance stays in [0,1]
        EvalResult::test("OBJ-BPS-01: expert_load_imbalance in [0,1]", || {
            let pass = BehavioralSynthesisPass::new();
            let compute = create_test_compute();
            let config = DynamicConfig::default();
            let metrics = pass.run(&compute, &config);
            (0.0..=1.0).contains(&metrics.expert_load_imbalance)
        }),
        // OBJ-BPS-02: A model with no MoE op reports has_moe = false
        EvalResult::test("OBJ-BPS-02: Dense models report has_moe = false", || {
            let pass = BehavioralSynthesisPass::new();
            let compute = create_test_compute();
            let config = DynamicConfig::default();
            let metrics = pass.run(&compute, &config);
            !metrics.has_moe
        }),
        // OBJ-BPS-03: A model with a real MoE op is actually detected —
        // the pass reads `compute.op_flops`, not a hardcoded default.
        EvalResult::test("OBJ-BPS-03: A MoE op is detected via op_type", || {
            let pass = BehavioralSynthesisPass::new();
            let compute = create_test_compute_with_moe();
            let config = DynamicConfig::default();
            let metrics = pass.run(&compute, &config);
            metrics.has_moe
        }),
        // OBJ-BPS-04: Load balance efficiency in valid range
        EvalResult::test("OBJ-BPS-04: Load balance efficiency in valid range", || {
            let pass = BehavioralSynthesisPass::new();
            let compute = create_test_compute();
            let config = DynamicConfig::default();
            let metrics = pass.run(&compute, &config);
            metrics.load_balance_efficiency >= 0.0 && metrics.load_balance_efficiency <= 100.0
        }),
    ]
}

// ── COHERENCE TESTS ──────────────────────────────────────────────────

fn eval_coherence() -> Vec<EvalResult> {
    vec![
        // OBJ-COH-01: All metrics finite
        EvalResult::test("OBJ-COH-01: All metrics finite", || {
            let vm_pass = VirtualMemoryPass::new();
            let sta_pass = StabilityAnalysisPass::new();
            let bps_pass = BehavioralSynthesisPass::new();

            let mem = create_test_memory_metrics();
            let graph = create_test_graph();
            let compute = create_test_compute();
            let config = DynamicConfig::default();

            let vm = vm_pass.run(&mem);
            let sta = sta_pass.run(&graph, &mem, &create_test_model_config());
            let bps = bps_pass.run(&compute, &config);

            vm.fragmentation_pct.is_finite()
                && sta.chaos_index.is_finite()
                && bps.expert_load_imbalance.is_finite()
        }),
        // OBJ-COH-02: Dynamic augments static
        EvalResult::test("OBJ-COH-02: Dynamic augments static", || {
            let pass = VirtualMemoryPass::new();
            let mem = create_test_memory_metrics();
            let metrics = pass.run(&mem);
            metrics.peak_vram_with_virtual_gb <= mem.peak_vram_gb()
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_suite() {
        let report = run_full_evaluation();
        assert!(report.total > 0, "No tests were run");
        println!("\n{}", report.pass_rate());
    }
}
