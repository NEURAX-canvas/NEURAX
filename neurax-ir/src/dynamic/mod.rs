//! NEURAX Dynamic Predictive System
//!
//! This module implements the dynamic analysis passes that extend the static
//! pipeline with predictive capabilities:
//!
//! - **VirtualMemoryPass**: Models memory fragmentation and virtualization savings
//! - **StabilityAnalysisPass**: Predicts training stability via Lyapunov exponents
//! - **BehavioralSynthesisPass**: Infers runtime behaviors (MoE imbalance, cache locality)
//!
//! These passes run in parallel after the static pipeline and provide
//! cross-pass feedback to enrich the final report.

pub mod behavioral;
pub mod evaluation;
pub mod stability;
pub mod types;
pub mod virtual_memory;

pub use behavioral::{BehavioralMetrics, BehavioralSynthesisPass};
pub use evaluation::{run_full_evaluation, EvaluationReport};
pub use stability::{StabilityAnalysisPass, StabilityMetrics};
pub use types::{DynamicConfig, DynamicResults};
pub use virtual_memory::{AllocationStrategy, VirtualMemoryMetrics, VirtualMemoryPass};
