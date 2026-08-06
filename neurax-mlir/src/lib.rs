//! NEURAX MLIR Code Generation Library
//!
//! Provides MLIR dialects and lowering infrastructure for the NEURAX compiler:
//! - Custom dialects for architecture, operators, memory, hardware, parallelism
//! - Multi-target lowering to CPU, CUDA, Vulkan, Metal, ROCm
//! - IREE integration for cross-platform deployment

pub mod compiler;
pub mod context;
pub mod dialects;
pub mod integration;
pub mod iree;
pub mod lowering;
pub mod module;
pub mod passes;
pub mod targets;

// Re-export key types for convenience
pub use compiler::compile_model_to_mlir;
pub use context::NeuraxContext;
pub use module::NeuraxModule;

// Lowering infrastructure
pub use lowering::{
    ArchitectureLowering, HardwareLowering, LoweringContext, LoweringPass, MemoryLowering,
    OperatorLowering, ParallelismLowering,
};

// Target backends
pub use targets::{
    CpuBackend, CudaBackend, MetalBackend, RocmBackend, TargetBackend, TargetLowering,
    VulkanBackend,
};

// IREE integration
pub use iree::{IreeCompiler, IreeDevice, IreeTarget};

#[cfg(test)]
mod tests;
