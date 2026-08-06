//! NEURAX MLIR Dialects
//!
//! This module defines all NEURAX dialects using melior's API.

pub mod architecture;
pub mod compute;
pub mod cost;
pub mod data;
pub mod graph;
pub mod hardware;
pub mod memory;
pub mod operator;
pub mod optimization;
pub mod parallelism;
pub mod report;
pub mod tensor;
pub mod training;
pub mod utils;

pub use architecture::ArchitectureDialect;
pub use compute::ComputeDialect;
pub use cost::CostDialect;
pub use data::DataDialect;
pub use graph::GraphDialect;
pub use hardware::HardwareDialect;
pub use memory::MemoryDialect;
pub use operator::OperatorDialect;
pub use optimization::OptimizationDialect;
pub use parallelism::ParallelismDialect;
pub use report::ReportDialect;
pub use tensor::TensorDialect;
pub use training::TrainingDialect;
