//! Behavioral Synthesis Pass
//!
//! Static architecture facts about runtime load balancing — not a
//! predictor of cache behavior, memory contention, or numerical stability.
//!
//! This pass used to also report `memory_contention_score`,
//! `cache_locality_score`, `numerical_sensitivity`, `dynamic_hotspot_ratio`,
//! `execution_path_entropy`, `temporal_locality_score`,
//! `memory_bank_conflict_rate` and a `prediction_confidence` — every one of
//! them a hardcoded constant (0.1, 0.8, 0.5, 0.6, 1.0, 0.65, ...) returned
//! identically for every model, since `run()`'s own `compute`/`config`
//! parameters were never read. A 7-layer MLP and a 175B MoE transformer got
//! the exact same "Cache Locality: 0.800" and "Prediction Conf.: 65%".
//! Real cache locality, memory contention and execution hotspots are
//! properties of an actual execution trace on real hardware — no static
//! architecture pass can honestly produce them, so they were removed rather
//! than left as plausible-looking fake precision. Numerical sensitivity is
//! a real, computable question, but NEURAX already answers it honestly
//! elsewhere: `StabilityAnalysisPass`'s per-layer Lyapunov margins and
//! `fp32_required_pct` (`neurax-ir/src/dynamic/stability.rs`).
//!
//! What's left is the one thing genuinely readable from the static IR:
//! whether the model has any MoE routing at all.

use serde::{Deserialize, Serialize};

use crate::compute::ComputeIR;
use crate::dynamic::types::DynamicConfig;
use crate::operator::OpType;

/// Behavioral Synthesis Pass
#[derive(Debug, Clone, Default)]
pub struct BehavioralSynthesisPass;

/// Metrics from behavioral synthesis (M50, M54)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BehavioralMetrics {
    /// M50: Expert load imbalance [0,1]. No runtime token-routing trace is
    /// available to a static pass, so this is 0.0 whenever the model has no
    /// MoE layer at all (there is nothing to be imbalanced) and an assumed
    /// baseline of 0.0 — "balanced until proven otherwise", not a measured
    /// fact — when it does. `has_moe` is what actually distinguishes the
    /// two cases; don't read a MoE model's 0.0 here as "verified balanced".
    pub expert_load_imbalance: f64,
    /// Whether the model contains any MoE routing at all — the real,
    /// statically-checkable fact `expert_load_imbalance` alone doesn't
    /// convey (a plain CNN and an unbalanced-but-untraced MoE model would
    /// otherwise both read `0.0`).
    pub has_moe: bool,
    /// M54: Load balance efficiency [%], `(1 - expert_load_imbalance) * 100`
    /// under the same assumed-until-traced baseline as `expert_load_imbalance`.
    pub load_balance_efficiency: f64,
}

impl BehavioralSynthesisPass {
    pub fn new() -> Self {
        Self
    }

    pub fn run(&self, compute: &ComputeIR, _config: &DynamicConfig) -> BehavioralMetrics {
        let has_moe = compute.op_flops.iter().any(|op| op.op_type == OpType::MoE);
        // No runtime routing trace exists in a static pass, so there is no
        // honest way to measure real imbalance — 0.0 is the stated
        // assumption for a MoE model, not a claim of having checked.
        let expert_load_imbalance = 0.0;
        let load_balance_efficiency = (1.0 - expert_load_imbalance) * 100.0;

        BehavioralMetrics {
            expert_load_imbalance,
            has_moe,
            load_balance_efficiency,
        }
    }
}
