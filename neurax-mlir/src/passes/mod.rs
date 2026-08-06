//! NEURAX MLIR Passes
//!
//! This module provides MLIR passes for transforming and analyzing
//! NEURAX IR modules.

pub mod architecture;
pub mod compute;
pub mod cost;
pub mod hardware;
pub mod memory;
pub mod parallelism;

// Re-export pass traits
pub use architecture::ArchitecturePass;
pub use compute::ComputePass;
pub use cost::CostPass;
pub use hardware::HardwarePass;
pub use memory::MemoryPass;
pub use parallelism::ParallelismPass;
