//! Memory IR - Dialecte de la simulation mémoire

mod fragmentation;
mod ir;
mod liveness;
mod metrics;
mod pass;

pub use fragmentation::*;
pub use ir::*;
pub use liveness::*;
pub use metrics::*;
pub use pass::*;
