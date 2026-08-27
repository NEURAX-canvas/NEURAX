//! Tensor IR - Dialecte de la propagation de formes

mod ir;
mod metrics;
mod pass;
mod shape_inference;

pub use ir::*;
pub use metrics::*;
pub use pass::*;
