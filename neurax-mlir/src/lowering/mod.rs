//! MLIR Lowering Infrastructure
//!
//! This module provides lowering passes to convert NEURAX dialects
//! to standard MLIR operations (linalg, tensor, arith, scf).

mod architecture;
mod context;
mod hardware;
mod memory;
mod operator;
mod parallelism;
mod pass;

pub use architecture::ArchitectureLowering;
pub use context::LoweringContext;
pub use hardware::HardwareLowering;
pub use memory::MemoryLowering;
pub use operator::OperatorLowering;
pub use parallelism::ParallelismLowering;
pub use pass::LoweringPass;

use melior::ir::Module;

/// Run all lowering passes on a module
pub fn run_lowering_pipeline<'c>(
    module: &mut Module<'c>,
    context: &mut LoweringContext<'c>,
) -> Result<(), String> {
    // Order matters: architecture first, then operators, then memory/hardware
    ArchitectureLowering::run(module, context)?;
    OperatorLowering::run(module, context)?;
    MemoryLowering::run(module, context)?;
    HardwareLowering::run(module, context)?;
    ParallelismLowering::run(module, context)?;

    Ok(())
}
