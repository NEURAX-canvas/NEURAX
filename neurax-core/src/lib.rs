//! # NEURAX Core
//!
//! **The NEURAX unified analysis engine — one function call from JSON to a full
//! analytical report.**
//!
//! Part of the [NEURAX](https://github.com/rustnew/NEURAX) analytical compiler
//! for neural network architectures. Orchestrates the entire pipeline —
//! **parse → validate → 10-pass IR → report** — behind a tiny public API.
//!
//! ## Quick example
//!
//! ```rust
//! use neurax_core::analyze_json;
//!
//! let json = r#"{
//!   "schema_version": "1.0",
//!   "model": {
//!     "name": "tiny-gpt",
//!     "type": "transformer",
//!     "layers": [
//!       { "id": "attn_0", "layer_type": "attention",
//!         "input_shape": [128, 768], "output_shape": [128, 768],
//!         "params": { "num_heads": 12 } }
//!     ],
//!     "global_params": { "hidden_size": 768, "num_layers": 1 }
//!   },
//!   "training": { "batch_size": 32, "optimizer": "adamw", "precision": "bf16" },
//!   "hardware": {
//!     "gpus": [
//!       { "name": "A100-80GB", "count": 1, "memory_gb": 80,
//!         "tflops_fp16": 312, "tflops_fp32": 19.5,
//!         "memory_bandwidth_gb_s": 2039, "tensor_cores": true }
//!     ],
//!     "interconnect": "None", "interconnect_bandwidth_gb_s": 0
//!   }
//! }"#;
//!
//! let result = analyze_json(json).expect("analysis succeeds");
//! assert!(result.compute.metrics.total_flops > 0.0);
//! assert!(result.memory.metrics.peak_vram_bytes > 0);
//! ```
//!
//! ## Main entry points
//!
//! - [`analyze_json`] — full analysis from a JSON string
//! - [`analyze_file`] — full analysis from a file
//! - [`validate_json`] — parse + validate only
//! - [`get_model_summary`] — quick summary without full analysis
//!
//! ## Run the example
//!
//! ```bash
//! cargo run --example analyze_end_to_end
//! ```

mod engine;
pub mod export;
mod runner;
pub mod streaming;
pub mod units;

pub use engine::*;
pub use runner::*;
pub use units::{Bytes, FLOPs, LatencyMs, ParamCount, TokensPerSec};

use neurax_ir::report::{PhaseTimingEntry, ReportInput};
use neurax_ir::traits::{IrPass, ReportPass as ReportPassTrait};
use neurax_ir::{
    dynamic::{
        BehavioralSynthesisPass, DynamicConfig, DynamicResults, StabilityAnalysisPass,
        VirtualMemoryPass,
    },
    ArchitectureIR, ArchitecturePass, ComputeIR, ComputePass, CostIR, CostPass, GraphIR, GraphPass,
    HardwareIR, HardwarePass, MemoryIR, MemoryPass, NeuraxContext, NeuraxError, OperatorIR,
    OperatorPass, ParallelismIR, ParallelismPass, ReportIR, ReportPass, TensorIR, TensorPass,
};
use neurax_parser::ModelConfig;
use std::time::Instant;

/// Analysis result containing all IR outputs
#[derive(Debug)]
pub struct AnalysisResult {
    pub arch: ArchitectureIR,
    pub graph: GraphIR,
    pub tensor: TensorIR,
    pub operator: OperatorIR,
    pub compute: ComputeIR,
    pub memory: MemoryIR,
    pub parallelism: ParallelismIR,
    pub hardware: HardwareIR,
    pub cost: CostIR,
    pub report: ReportIR,
    /// Dynamic analysis results (M36-M55)
    pub dynamic: DynamicResults,
    pub analysis_time_ms: u64,
}

impl AnalysisResult {
    /// Export all metrics to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        use neurax_ir::report::JsonOutput;

        let output = JsonOutput::from_report_with_dynamic(
            &self.report,
            "model.json",
            self.analysis_time_ms,
            &self.dynamic,
        );
        output.to_json()
    }

    /// Export all metrics to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        use neurax_ir::report::JsonOutput;

        let output = JsonOutput::from_report_with_dynamic(
            &self.report,
            "model.json",
            self.analysis_time_ms,
            &self.dynamic,
        );
        output.to_json_bytes()
    }

    /// Save metrics to a JSON file
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = self
            .to_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

/// Run full analysis pipeline
pub fn run_analysis(config: ModelConfig) -> Result<AnalysisResult, NeuraxError> {
    let start = Instant::now();
    let ctx = NeuraxContext::new(config.clone());
    let mut phase_timeline: Vec<PhaseTimingEntry> = Vec::new();

    macro_rules! timed_phase {
        ($name:expr, $block:expr) => {{
            let t0 = Instant::now();
            let result = $block;
            phase_timeline.push(PhaseTimingEntry {
                name: $name.to_string(),
                duration_ms: t0.elapsed().as_millis() as u64,
                status: "completed".to_string(),
            });
            result
        }};
    }

    // Phase 1: Architecture
    let arch_pass = ArchitecturePass;
    let (arch, arch_metrics) = timed_phase!("Architecture", {
        let (mut a, _) = arch_pass.run(&config, &ctx)?;
        let m = arch_pass.compute_metrics(&mut a, &ctx)?;
        arch_pass.validate(&a, &m)?;
        (a, m)
    });
    let _ = arch_metrics;

    // Phase 2: Graph
    let graph_pass = GraphPass;
    let (graph, graph_metrics) = timed_phase!("Graph", {
        let (mut g, _) = graph_pass.run(&arch, &ctx)?;
        let m = graph_pass.compute_metrics(&mut g, &ctx)?;
        graph_pass.validate(&g, &m)?;
        (g, m)
    });
    let _ = graph_metrics;

    // Phase 3: Tensor
    let tensor_pass = TensorPass;
    let (tensor, tensor_metrics) = timed_phase!("Tensor", {
        let (mut t, _) = tensor_pass.run(&graph, &ctx)?;
        let m = tensor_pass.compute_metrics(&mut t, &ctx)?;
        tensor_pass.validate(&t, &m)?;
        (t, m)
    });
    let _ = tensor_metrics;

    // Phase 4: Operator
    let operator_pass = OperatorPass;
    let (operator, operator_metrics) = timed_phase!("Operator", {
        let (mut o, _) = operator_pass.run(&(tensor.clone(), arch.clone()), &ctx)?;
        let m = operator_pass.compute_metrics(&mut o, &ctx)?;
        operator_pass.validate(&o, &m)?;
        (o, m)
    });
    let _ = operator_metrics;

    // Phase 5: Compute
    let compute_pass = ComputePass;
    let (compute, compute_metrics) = timed_phase!("Compute", {
        let (mut c, _) = compute_pass.run(&operator, &ctx)?;
        let m = compute_pass.compute_metrics(&mut c, &ctx)?;
        compute_pass.validate(&c, &m)?;
        (c, m)
    });
    let _ = compute_metrics;

    // Phase 6: Memory
    let memory_pass = MemoryPass;
    let (memory, memory_metrics) = timed_phase!("Memory", {
        let (mut m, _) = memory_pass.run(&(compute.clone(), tensor.clone(), arch.clone()), &ctx)?;
        let metrics = memory_pass.compute_metrics(&mut m, &ctx)?;
        memory_pass.validate(&m, &metrics)?;
        (m, metrics)
    });
    let _ = memory_metrics;

    // Phases 7, 8 and 11 are all independent of each other and of Cost/Report:
    // Parallelism and Hardware only need Memory+Graph+Compute, and so does
    // Dynamic (VirtualMemory/Stability/BehavioralSynthesis — none of the
    // three reads hardware, cost, or report). Running all three concurrently
    // removes dead serialization: Dynamic used to wait for Cost+Report to
    // finish even though it never touched their output.
    //
    // HardwarePass::build() takes a ParallelismIR in its Input tuple but
    // never reads it (`let (compute_ir, memory_ir, _parallel_ir) = input;`,
    // same in CostPass::build()) — that slot exists for pipeline-shape
    // symmetry, not a real data dependency. So Hardware doesn't need to wait
    // on Parallelism either, and a placeholder here is not an approximation:
    // it is what build() would have ignored anyway. (This used to run
    // HardwarePass a second time, in full, with the real ParallelismIR
    // swapped in — byte-for-byte the same result, since the value swapped in
    // was never read either time.)
    let ((parallelism_result, hardware_result), dynamic) = rayon::join(
        || {
            rayon::join(
                || {
                    let parallelism_pass = ParallelismPass;
                    parallelism_pass.run(&(memory.clone(), graph.clone()), &ctx)
                },
                || {
                    let hardware_pass = HardwarePass;
                    hardware_pass.run(
                        &(compute.clone(), memory.clone(), ParallelismIR::default()),
                        &ctx,
                    )
                },
            )
        },
        || {
            let dynamic_config = DynamicConfig::default();
            let (vm_metrics, (sta_metrics, bps_metrics)) = rayon::join(
                || {
                    let vm_pass = VirtualMemoryPass::new();
                    Some(vm_pass.run(&memory.metrics))
                },
                || {
                    rayon::join(
                        || {
                            let sta_pass = StabilityAnalysisPass::new();
                            Some(sta_pass.run(&graph, &memory.metrics, &ctx.config))
                        },
                        || {
                            let bps_pass = BehavioralSynthesisPass::new();
                            Some(bps_pass.run(&compute, &dynamic_config))
                        },
                    )
                },
            );
            DynamicResults {
                virtual_memory: vm_metrics,
                stability: sta_metrics,
                behavioral: bps_metrics,
            }
        },
    );
    let (parallelism, _parallelism_metrics) = parallelism_result?;
    let (hardware, _hardware_metrics) = hardware_result?;

    // Phase 9: Cost
    let cost_pass = CostPass;
    let (cost, cost_metrics) = timed_phase!("Cost", {
        let (mut c, _) = cost_pass.run(&(hardware.clone(), parallelism.clone()), &ctx)?;
        let m = cost_pass.compute_metrics(&mut c, &ctx)?;
        cost_pass.validate(&c, &m)?;
        (c, m)
    });
    let _ = cost_metrics;

    // Phase 10: Report
    let report_pass = ReportPass;
    let mut report = timed_phase!("Report", {
        report_pass.build_report(
            &ReportInput {
                arch: &arch,
                graph: &graph,
                tensor: &tensor,
                operator: &operator,
                compute: &compute,
                memory: &memory,
                parallelism: &parallelism,
                hardware: &hardware,
                cost: &cost,
            },
            &ctx,
        )?
    });
    report.phase_timeline = phase_timeline;

    // Phase 11 (Dynamic Analysis, M36-M55) was computed above, concurrently
    // with Parallelism/Hardware — it depends on neither Cost nor Report. Its
    // stability output feeds the H007 hyperparameter hint below, so that
    // merge happens here, after both are done, rather than making Report
    // depend on Dynamic during computation (which would undo the join).
    report
        .diagnostics
        .extend(neurax_ir::report::generate_hyperparameter_diagnostics(
            &ctx,
            report.metrics.total_parameters,
            dynamic.stability.as_ref(),
        ));
    report.metrics.diagnostic_count = report.diagnostics.len();

    let analysis_time_ms = start.elapsed().as_millis() as u64;

    // Return result with owned values
    Ok(AnalysisResult {
        arch,
        graph,
        tensor,
        operator,
        compute,
        memory,
        parallelism,
        hardware,
        cost,
        report,
        dynamic,
        analysis_time_ms,
    })
}

/// Trait extension for running passes
pub trait IrPassExt: IrPass {
    fn run(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<(Self::Output, Self::Metrics), NeuraxError> {
        let mut output = self.build(input, ctx).map_err(|e| e.into())?;
        let metrics = self
            .compute_metrics(&mut output, ctx)
            .map_err(|e| e.into())?;
        self.validate(&output, &metrics).map_err(|e| e.into())?;
        Ok((output, metrics))
    }
}

impl<T: IrPass> IrPassExt for T {}
