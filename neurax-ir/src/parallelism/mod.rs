//! Parallelism IR - Dialecte de l'analyse de scalabilité

mod ir;
mod metrics;
mod pass;
mod strategies;

pub use ir::*;
pub use metrics::*;
pub use pass::*;
pub use strategies::*;
