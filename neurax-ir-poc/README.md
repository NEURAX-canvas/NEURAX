# neurax-ir-poc — architectural comparison, not a product

One specific claim, made in conversation, tested against the real compiler
on real models: that driving NEURAX's existing passes through a memoized,
typed query graph — instead of `neurax_core::run_analysis`'s hand-threaded
tuples plus a `ctx.set_metric()`/`get_metric()` string-keyed side channel —
produces identical results while being structurally safer, and lets a
sweep over hyperparameters skip recomputing whatever genuinely doesn't
depend on them.

**This is not a rewrite of NEURAX.** Every formula (parameter counts,
FLOPs, memory accounting, hardware/roofline, cost, ZeRO communication) is
the real, unmodified `neurax-ir` pass code — `ArchitecturePass`,
`GraphPass`, `TensorPass`, `OperatorPass`, `ComputePass`, `MemoryPass`,
`ParallelismPass`, `HardwarePass`, `CostPass`, called exactly as
`run_analysis` calls them. Only the orchestration differs. This is
deliberate: reimplementing the math from scratch would risk introducing new
transcription bugs that confound the comparison — a mismatch would prove
nothing about the architecture, only about whether this PoC's author copied
a formula correctly. Holding the computation fixed and changing only the
wiring isolates the one variable actually in question.

Excluded from the main workspace (`Cargo.toml`'s `exclude`) — it is not a
dependency of the product and CI does not need to keep it green.

## Run it yourself

```bash
# 5-metric comparison across one real model per family
cargo run --manifest-path neurax-ir-poc/Cargo.toml --example compare

# memoization: Architecture+Graph computed once vs. once per sweep point
cargo run --manifest-path neurax-ir-poc/Cargo.toml --release --example sweep_demo
```

## Result 1 — `examples/compare.rs`: 5 metrics, 8 families, one real model each

| family | model | total_parameters | total_flops | peak_vram_bytes | effective_tflops | communication_overhead |
|---|---|---|---|---|---|---|
| transformer | gpt3_175b.json | match | match | match | match | match |
| moe | mixtral_8x7b.json | match | match | match | match | match |
| cnn | resnet50.json | match | match | match | match | match |
| ssm | mamba_2.8b.json | match | match | match | match | match |
| diffusion | sdxl_1.0.json | match | match | match | match | match |
| gan | dcgan.json | match | match | match | match | match |
| rnn | awd_lstm_ptb.json | match | match | match | match | match |
| gnn | gcn_ogbn_arxiv.json | match | match | match | match | match |

All 40 values (5 metrics × 8 families) match exactly. `total_parameters` and
`total_flops` never diverge in either version tried below — they're read
straight off `ArchitectureMetrics`/`ComputeMetrics`, no cross-pass channel
involved. `peak_vram_bytes`/`effective_tflops`/`communication_overhead` are
where a wiring mistake would actually show up, and did, twice, before this
table was clean:

**Bug found on the very first run**: `communication_overhead` was `0.0` for
every multi-GPU model, `0.0` correctly for single-GPU ones. Cause:
`ParallelismPass::compute_metrics()` reads `total_flops` via
`ctx.get_metric()` — a real, unmodified pass. That only works if the *same*
`NeuraxContext` object saw both the `set_metric` (from `ComputePass`) and
this `get_metric`. `StructuralFacts` and `View` deliberately use separate
contexts (that's what lets many `View`s share one `StructuralFacts`), so the
value never arrived — silently, no error, no panic. This is not a
contrived example: it is the exact fragility this PoC exists to argue
against, reproduced in this PoC's own first draft. Fixed by injecting
`total_flops` from the already-computed, typed fact
(`self.compute()?.1.total_flops`) at the point `parallelism()` needs it,
instead of relying on the shared mutable channel.

**Bug found while building the sweep demo**: the first version of
`StructuralFacts` shared Architecture/Graph/**Tensor**/**Operator**/
**Compute** across every `View`, on the assumption that none of them depend
on batch_size. False for Tensor: `infer_shapes()`
(`neurax-ir/src/tensor/shape_inference.rs:168`) reads
`config.training.batch_size` to build entry tensor shapes, and
Operator/Compute's FLOPs come from those shapes. Sharing Tensor across
batch sizes silently froze `peak_vram_bytes` at an identical value
regardless of batch_size — caught by `sweep_demo.rs`'s own sanity assertion
comparing batch=1 against batch=32, not assumed correct. Fixed by moving
`tensor`/`operator`/`compute` onto `View` (recomputed per view) and keeping
only the verified-invariant `architecture`/`graph` on `StructuralFacts`.

Both bugs were caught by directly measuring the comparison, not by
reasoning about the architecture in the abstract — which is itself
evidence for the actual thesis of this whole exercise: verification against
something real catches what code review and confident-sounding comments do
not.

## Result 2 — `examples/sweep_demo.rs`: memoization, honestly bounded

Only `Architecture` and `Graph` are verified batch/zero_stage/precision/
gpu_count-invariant in the real pass implementations — not five phases, as
first (wrongly) claimed. Sweeping 24 configurations
(6 batch_sizes × 4 zero_stages) of `gpt3_175b.json`:

```
Current sweep_hyperparameters: 24 points, 12.26ms total (511µs/point)
This PoC: Architecture+Graph once (98µs) + 24 Views (5.44ms) = 5.54ms total
```

A real, honestly-measured ~2.2× reduction for this grid — not the larger
number a 5-phases-shared claim would have implied, because that claim was
wrong. `Tensor`/`Operator`/`Compute` still run once per point in both
versions, because they genuinely need to: `peak_vram_bytes` correctly grows
from 1411 GB (batch=1) to 1862 GB (batch=32), and
`communication_overhead` correctly moves from 90.5% to 93.5% between
zero_stage 0 and 3 — sanity checks that would fail immediately if either
PoC bug above had gone uncaught.

## What this does and doesn't establish

**Does**: for the 5 metrics and 8 models tested, restructuring the same
computations through typed, memoized queries instead of hand-threaded
tuples + a string-keyed side channel produces identical output, gives a
real (if modest, for these model sizes) memoization benefit on a sweep, and
converts a whole class of bug (a fact silently never reaching the pass that
needed it — the exact root cause of the real `27ac229` and `fbf867c` fixes
in the product) into something that has to be an explicit, typed call
someone can see in the code, rather than a hope about shared mutable state.

**Doesn't**: prove this at scale (all reference models here are small
enough that a full analysis is sub-millisecond either way), prove it across
every metric NEURAX reports (5 of 35+, chosen for centrality and history of
bugs, not exhaustiveness), or establish that a full incremental-query
system (e.g. an actual `salsa` integration) would be worth its own
implementation cost for the product. It establishes that the specific
architectural claim behind this exercise is correct in the cases tested,
including two cases where a naive version of the same idea was wrong until
measured against something real.
