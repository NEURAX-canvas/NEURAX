//! Experimental, comparison-only proof of concept.
//!
//! This crate does **not** reimplement NEURAX's formulas — it calls the
//! exact same, already-verified passes (`ArchitecturePass`, `GraphPass`,
//! `TensorPass`, `OperatorPass`, `ComputePass`, `MemoryPass`,
//! `ParallelismPass`, `HardwarePass`, `CostPass` from `neurax-ir`,
//! unmodified) that `neurax_core::run_analysis` calls. The one thing that
//! differs is *how they're wired together*: every fact is a memoized method
//! on `StructuralFacts`/`View`, with real, type-checked Rust calls
//! expressing dependencies instead of `neurax_core::run_analysis`'s
//! hand-threaded `.clone()`d tuples plus a `ctx.set_metric()`/`get_metric()`
//! string-keyed side channel.
//!
//! ## The boundary here was wrong on the first attempt — corrected, not hidden
//!
//! The first version of this file shared Architecture/Graph/**Tensor**/
//! **Operator**/**Compute** across every `View`, on the claim that none of
//! them depend on batch_size/zero_stage/precision/gpu_count. That's true for
//! Architecture and Graph. It is **false** for Tensor: `infer_shapes()`
//! (`neurax-ir/src/tensor/shape_inference.rs:168`) reads
//! `config.training.batch_size` to build every entry tensor's shape, and
//! Operator/Compute's FLOPs are computed from those shapes — so sharing them
//! across different batch sizes silently froze `peak_vram_bytes` at an
//! identical value regardless of batch_size the first time this was run
//! against a real sweep (`examples/sweep_demo.rs`), instead of the real,
//! previously-verified-monotonic growth. Caught by that example's own sanity
//! assertion, not assumed away. Only `architecture`/`graph` are genuinely
//! invariant; `tensor`/`operator`/`compute` moved to `View`, keyed
//! implicitly by whichever `ModelConfig` that `View` was built from.

use neurax_core::IrPassExt;
use neurax_ir::architecture::{ArchitectureIR, ArchitectureMetrics, ArchitecturePass};
use neurax_ir::compute::{ComputeIR, ComputeMetrics, ComputePass};
use neurax_ir::cost::{CostIR, CostMetrics, CostPass};
use neurax_ir::graph::{GraphIR, GraphMetrics, GraphPass};
use neurax_ir::hardware::{HardwareIR, HardwareMetrics, HardwarePass};
use neurax_ir::memory::{MemoryIR, MemoryMetrics, MemoryPass};
use neurax_ir::operator::{OperatorIR, OperatorMetrics, OperatorPass};
use neurax_ir::parallelism::{ParallelismIR, ParallelismMetrics, ParallelismPass};
use neurax_ir::tensor::{TensorIR, TensorMetrics, TensorPass};
use neurax_ir::NeuraxContext;
use neurax_ir::NeuraxError;
use neurax_parser::ModelConfig;
use once_cell::sync::OnceCell;

/// Only what's verified to never change with batch_size/zero_stage/
/// precision/gpu_count: layer connectivity (Graph) and parameter counts
/// (Architecture). Computed once per model structure, shared by every
/// `View` built from it, regardless of what training/hardware knobs that
/// `View` varies.
pub struct StructuralFacts {
    pub architecture: (ArchitectureIR, ArchitectureMetrics),
    pub graph: (GraphIR, GraphMetrics),
}

impl StructuralFacts {
    /// Computed with a throwaway context — Architecture and Graph don't
    /// read `training`/`hardware` at all, only `model`, so any valid
    /// context for this config's *structure* works.
    pub fn compute(config: &ModelConfig) -> Result<Self, NeuraxError> {
        let ctx = NeuraxContext::new(config.clone());
        let architecture = ArchitecturePass.run(config, &ctx)?;
        let graph = GraphPass.run(&architecture.0, &ctx)?;
        Ok(Self {
            architecture,
            graph,
        })
    }
}

/// One training/hardware configuration's view onto a shared
/// `StructuralFacts`. `tensor`/`operator`/`compute` are NOT shared across
/// `View`s (see the module doc for why) but are still memoized *within* one
/// `View`, so asking for `cost()` after already asking for `hardware()` on
/// the same `View` doesn't redo any of it.
pub struct View<'a> {
    facts: &'a StructuralFacts,
    ctx: NeuraxContext,
    tensor: OnceCell<(TensorIR, TensorMetrics)>,
    operator: OnceCell<(OperatorIR, OperatorMetrics)>,
    compute: OnceCell<(ComputeIR, ComputeMetrics)>,
    memory: OnceCell<(MemoryIR, MemoryMetrics)>,
    parallelism: OnceCell<(ParallelismIR, ParallelismMetrics)>,
    hardware: OnceCell<(HardwareIR, HardwareMetrics)>,
    cost: OnceCell<(CostIR, CostMetrics)>,
}

impl<'a> View<'a> {
    pub fn new(facts: &'a StructuralFacts, config: ModelConfig) -> Self {
        Self {
            facts,
            ctx: NeuraxContext::new(config),
            tensor: OnceCell::new(),
            operator: OnceCell::new(),
            compute: OnceCell::new(),
            memory: OnceCell::new(),
            parallelism: OnceCell::new(),
            hardware: OnceCell::new(),
            cost: OnceCell::new(),
        }
    }

    pub fn tensor(&self) -> Result<&(TensorIR, TensorMetrics), NeuraxError> {
        self.tensor
            .get_or_try_init(|| TensorPass.run(&self.facts.graph.0, &self.ctx))
    }

    pub fn operator(&self) -> Result<&(OperatorIR, OperatorMetrics), NeuraxError> {
        self.operator.get_or_try_init(|| {
            let tensor = &self.tensor()?.0;
            let input = (tensor.clone(), self.facts.architecture.0.clone());
            OperatorPass.run(&input, &self.ctx)
        })
    }

    pub fn compute(&self) -> Result<&(ComputeIR, ComputeMetrics), NeuraxError> {
        self.compute
            .get_or_try_init(|| ComputePass.run(&self.operator()?.0, &self.ctx))
    }

    pub fn memory(&self) -> Result<&(MemoryIR, MemoryMetrics), NeuraxError> {
        self.memory.get_or_try_init(|| {
            let compute = self.compute()?.0.clone();
            let tensor = self.tensor()?.0.clone();
            let input = (compute, tensor, self.facts.architecture.0.clone());
            MemoryPass.run(&input, &self.ctx)
        })
    }

    /// Same real dependency ParallelismPass::build() actually reads (memory
    /// + graph) — a direct Rust call to `self.facts.graph`/`self.memory()`,
    /// not a typed Input slot that happens to go unread.
    ///
    /// `ParallelismPass::compute_metrics()` also reads `total_flops` via
    /// `ctx.get_metric()` internally — a real, unmodified pass, so this PoC
    /// doesn't rewrite its body. That channel only works if the *same*
    /// `NeuraxContext` saw both the `set_metric` (ComputePass) and this
    /// `get_metric`, so the value is injected here, right where it's
    /// needed, from something already typed and known-correct
    /// (`self.compute()`) instead of hoping a shared mutable object already
    /// has it — the same discipline as everywhere else in this file.
    pub fn parallelism(&self) -> Result<&(ParallelismIR, ParallelismMetrics), NeuraxError> {
        self.parallelism.get_or_try_init(|| {
            let total_flops = self.compute()?.1.total_flops;
            self.ctx.set_metric("total_flops", total_flops);
            let mem = self.memory()?.0.clone();
            let input = (mem, self.facts.graph.0.clone());
            ParallelismPass.run(&input, &self.ctx)
        })
    }

    /// HardwarePass::build() takes a ParallelismIR slot in its Input tuple
    /// but never reads it (fixed properly in the real pipeline in 89c56d8,
    /// after finding it ran the whole pass twice for exactly this reason).
    /// Passing `ParallelismIR::default()` here isn't a shortcut this PoC
    /// takes — it's what the real pass's own behavior already makes true:
    /// there is no dependency on `parallelism()` to express.
    pub fn hardware(&self) -> Result<&(HardwareIR, HardwareMetrics), NeuraxError> {
        self.hardware.get_or_try_init(|| {
            let compute = self.compute()?.0.clone();
            let mem = self.memory()?.0.clone();
            let input = (compute, mem, ParallelismIR::default());
            HardwarePass.run(&input, &self.ctx)
        })
    }

    pub fn cost(&self) -> Result<&(CostIR, CostMetrics), NeuraxError> {
        self.cost.get_or_try_init(|| {
            let hw = self.hardware()?.0.clone();
            let par = self.parallelism()?.0.clone();
            let input = (hw, par);
            CostPass.run(&input, &self.ctx)
        })
    }

    // ── The 5 target metrics this PoC is compared against the real
    // compiler on — the ones most central to (and, in three cases, most
    // recently corrected by) this session's audit.

    pub fn total_parameters(&self) -> u64 {
        self.facts.architecture.1.total_parameters
    }

    pub fn total_flops(&self) -> Result<f64, NeuraxError> {
        Ok(self.compute()?.1.total_flops)
    }

    pub fn peak_vram_bytes(&self) -> Result<u64, NeuraxError> {
        Ok(self.memory()?.1.peak_vram_bytes)
    }

    pub fn effective_tflops(&self) -> Result<f64, NeuraxError> {
        Ok(self.hardware()?.1.effective_tflops)
    }

    pub fn communication_overhead(&self) -> Result<f64, NeuraxError> {
        Ok(self.parallelism()?.1.communication_overhead)
    }
}
