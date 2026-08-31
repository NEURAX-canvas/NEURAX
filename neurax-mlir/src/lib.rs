//! # NEURAX MLIR
//!
//! **MLIR code generation for NEURAX — neural architectures as MLIR dialects.**
//!
//! Part of the [NEURAX](https://github.com/rustnew/NEURAX) analytical compiler
//! for neural network architectures. Lowers a parsed [`ModelConfig`](neurax_parser::ModelConfig)
//! into **MLIR** using custom NEURAX dialects, with multi-target backends
//! (CPU, CUDA, Vulkan, Metal, ROCm) and IREE integration.
//!
//! ## Quick example
//!
//! ```rust
//! use neurax_mlir::{compile_model_to_mlir, NeuraxContext};
//! use neurax_parser::parse_model_config;
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
//! let config = parse_model_config(json).expect("valid NEURAX JSON");
//! let ctx = NeuraxContext::new();
//! let mlir = compile_model_to_mlir(ctx.as_context(), &config).expect("compiles");
//! assert!(mlir.contains("arch.model"));
//! ```
//!
//! ## Main entry points
//!
//! - [`compile_model_to_mlir`] — model → textual MLIR
//! - [`NeuraxContext`], [`NeuraxModule`] — MLIR context/module wrappers
//! - [`TargetBackend`](crate::targets::TargetBackend) — `CpuBackend`, `CudaBackend`,
//!   `VulkanBackend`, `MetalBackend`, `RocmBackend`
//! - [`IreeCompiler`](crate::iree::IreeCompiler) — IREE deployment
//!
//! ## Run the example
//!
//! ```bash
//! cargo run --example compile_to_mlir
//! ```

pub mod compiler;
pub mod context;
pub mod dialects;
pub mod integration;
pub mod iree;
pub mod module;
pub mod passes;
pub mod targets;

// Re-export key types for convenience
pub use compiler::compile_model_to_mlir;
pub use context::NeuraxContext;
pub use module::NeuraxModule;

// Target backends
pub use targets::{
    CpuBackend, CudaBackend, MetalBackend, RocmBackend, TargetBackend, TargetLowering,
    VulkanBackend,
};

// IREE integration
pub use iree::{IreeCompiler, IreeDevice, IreeTarget};

#[cfg(test)]
mod tests;
