//! Hyperparameter sweep: find the best training-speed hyperparameters for a
//! model by evaluating many candidate configurations with the existing
//! analytical pipeline — no execution, no new prediction machinery.
//!
//! NEURAX never runs a model, so a full analysis costs ~0-1ms (see the
//! per-model timings `AnalysisResult::analysis_time_ms` already reports).
//! That means sweeping thousands of (batch_size, zero_stage, gpu_count,
//! precision) combinations is cheap here in a way it never would be with
//! real training runs — this reuses `run_analysis` unchanged as a black-box
//! evaluator, called once per candidate.

use crate::{run_analysis, AnalysisResult};
use neurax_ir::NeuraxError;
use neurax_parser::ModelConfig;
use serde::{Deserialize, Serialize};

/// The hyperparameters this sweep varies. Any field left as a single-element
/// vec keeps that hyperparameter fixed at the base config's value.
#[derive(Debug, Clone)]
pub struct SweepCandidates {
    pub batch_sizes: Vec<usize>,
    pub zero_stages: Vec<u8>,
    pub gpu_counts: Vec<u32>,
    /// Precision strings as the schema accepts them, e.g. "fp32"/"bf16"/"fp16".
    pub precisions: Vec<String>,
}

impl SweepCandidates {
    /// A usable default range for a caller that doesn't know good candidates
    /// up front — gpu_count and precision are anchored to whatever the base
    /// config already declares, rather than guessed, since those two are
    /// hardware/precision choices the sweep isn't meant to second-guess.
    pub fn defaults_for(base_config: &ModelConfig) -> Self {
        Self {
            batch_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            zero_stages: vec![0, 1, 2, 3],
            gpu_counts: vec![base_config
                .hardware
                .gpus
                .first()
                .map(|g| g.count)
                .unwrap_or(1)],
            precisions: vec![base_config.training.precision.clone()],
        }
    }
}

/// One evaluated point in the sweep.
#[derive(Debug, Clone, Serialize)]
pub struct SweepPoint {
    pub batch_size: usize,
    pub zero_stage: u8,
    pub gpu_count: u32,
    pub precision: String,
    /// Whether `peak_vram_bytes` fits the configured GPU's VRAM.
    pub feasible: bool,
    pub peak_vram_gb: f64,
    pub throughput_tokens_per_s: f64,
    pub latency_ms: f64,
    pub training_cost_usd: f64,
}

/// What to optimize for among the feasible points. Infeasible points (OOM)
/// are never selected regardless of objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepObjective {
    MaxThroughput,
    MinCost,
    MinLatency,
    MaxBatchSize,
}

/// Result of a sweep: every evaluated point (for transparency — the caller
/// can see the whole feasibility frontier, not just the winner) plus the
/// best feasible point for the requested objective, if any point was feasible.
#[derive(Debug, Clone, Serialize)]
pub struct SweepResult {
    pub points: Vec<SweepPoint>,
    pub best: Option<SweepPoint>,
}

/// Evaluate every combination in `candidates` against `base_config` and pick
/// the best feasible one per `objective`. A candidate that fails to analyze
/// (e.g. an invalid combination) is skipped rather than aborting the sweep.
pub fn sweep_hyperparameters(
    base_config: &ModelConfig,
    candidates: &SweepCandidates,
    objective: SweepObjective,
) -> Result<SweepResult, NeuraxError> {
    let mut points = Vec::new();

    for &batch_size in &candidates.batch_sizes {
        for &zero_stage in &candidates.zero_stages {
            for &gpu_count in &candidates.gpu_counts {
                for precision in &candidates.precisions {
                    let mut config = base_config.clone();
                    config.training.batch_size = batch_size;
                    config.training.zero_stage = zero_stage;
                    config.training.precision = precision.clone();
                    for gpu in &mut config.hardware.gpus {
                        gpu.count = gpu_count;
                    }

                    let Ok(result) = run_analysis(config) else {
                        continue;
                    };

                    points.push(point_from_result(
                        batch_size,
                        zero_stage,
                        gpu_count,
                        precision.clone(),
                        &result,
                    ));
                }
            }
        }
    }

    let best = points
        .iter()
        .filter(|p| p.feasible)
        .max_by(|a, b| compare_by_objective(a, b, objective))
        .cloned();

    Ok(SweepResult { points, best })
}

fn point_from_result(
    batch_size: usize,
    zero_stage: u8,
    gpu_count: u32,
    precision: String,
    result: &AnalysisResult,
) -> SweepPoint {
    let peak = result.memory.metrics.peak_vram_bytes;
    let capacity = result.memory.metrics.gpu_vram_bytes;
    SweepPoint {
        batch_size,
        zero_stage,
        gpu_count,
        precision,
        feasible: capacity > 0 && peak <= capacity,
        peak_vram_gb: peak as f64 / 1e9,
        throughput_tokens_per_s: result.hardware.metrics.throughput_tokens_per_s,
        latency_ms: result.hardware.metrics.latency_ms,
        training_cost_usd: result.cost.metrics.training_cost_usd,
    }
}

fn compare_by_objective(
    a: &SweepPoint,
    b: &SweepPoint,
    objective: SweepObjective,
) -> std::cmp::Ordering {
    match objective {
        SweepObjective::MaxThroughput => a
            .throughput_tokens_per_s
            .total_cmp(&b.throughput_tokens_per_s),
        // Lower cost is "better", so invert the comparison for max_by.
        SweepObjective::MinCost => b.training_cost_usd.total_cmp(&a.training_cost_usd),
        SweepObjective::MinLatency => b.latency_ms.total_cmp(&a.latency_ms),
        SweepObjective::MaxBatchSize => a.batch_size.cmp(&b.batch_size),
    }
}
