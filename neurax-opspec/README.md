# neurax-opspec

**OpSpec-IR — the single specification for every NEURAX operation.**

> Status: built and fully integrated. Every real `LayerType` except
> `Custom` is migrated, tested, wired into `neurax-ir`; `cargo build
> --workspace` and `cargo test --workspace` pass. The rest of this
> document explains the reasoning — why this crate exists, what it draws
> on, and what it will never do.

---

## The problem that motivated its existence

On September 1st and 2nd, 2026, while fixing real bugs in NEURAX
(unwired CNN formulas, S4/H3 reusing Mamba's shape, RNN ignoring
`input_size`, GNN falling back to a generic placeholder), the same pattern
kept coming back: **the same `LayerType` was defined three times,
independently, in three separate files of the `neurax-ir` crate**:

| Property | File | Size |
|---|---|---|
| Parameter count | `neurax-ir/src/architecture/mod.rs` | ~1100 lines |
| FLOPs | `neurax-ir/src/operator/pass.rs` | ~1400 lines |
| Output shape | `neurax-ir/src/tensor/shape_inference.rs` | ~250 lines |

Nothing tied these three definitions together. A type could have correct
parameters and placeholder FLOPs for months with no mechanism to flag it —
which is exactly what happened to six CNN block types, discovered only by
manually re-reading the code.

**OpSpec-IR exists to make that class of bug structurally impossible**,
not merely detectable by audit.

---

## Role

OpSpec-IR is **not** an intermediate representation in the traditional
sense, and it will never become one more stage in NEURAX's analysis
pipeline (`ArchitectureIR → GraphIR → TensorIR → OperatorIR → ComputeIR →
MemoryIR → ParallelismIR → HardwareIR → CostIR → ReportIR`). It's a
**specification table, consulted by those stages** — static data, not a
stream that flows through them.

NEURAX isn't a compiler that generates executable code — this IR stays
true to that: **no parser, no text format, no code-generation target, no
graph-rewriting engine.** It does exactly one thing: guarantee that an
operation type is defined in exactly one place, with its shape, its
parameters, and its FLOPs born together.

## What concepts it draws on — verified sources

Three open-source projects were studied directly (source code and
official documentation) before designing this crate:

### ONNX — the primary inspiration

Every ONNX operator is declared **exactly once**, via
`ONNX_OPERATOR_SET_SCHEMA(Name, since_version, schema)`, with its shape
inference function attached in the same call
(`.TypeAndShapeInferenceFunction(...)`). The resulting schema is a
**programmatically queryable** object (`onnx.defs.get_schema()`), not just
documentation. Versioned via immutable snapshots: changing an operator
first archives the old definition in `old.cc`, so a graph already saved
keeps resolving against the schema that was true when it was created.

It's the only one of the projects studied with a genuinely
machine-readable schema — and the pattern taken directly from it here: one
registration point per type, bundling shape and cost together.

Sources: [ONNX AddNewOp](https://onnx.ai/onnx/repo-docs/AddNewOp.html),
[onnx.defs API](https://onnx.ai/onnx/api/defs.html),
[Versioning.md](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)

### StableHLO — decomposition as data

StableHLO defines a first-class `composite` operator, whose semantics come
from a `decomposition` attribute — the composite operator can be replaced
by its decomposition without changing the program's meaning. That
formalizes what `decompose_layer_to_ops`
(`neurax-ir/src/operator/pass.rs`) already does imperatively in NEURAX: a
future `decomposition` field on `OpSpec` would make that decomposition
queryable as data instead of hidden in code.

StableHLO also guarantees strict compatibility (5 years back, 2 years
forward) for its serialized artifacts, via a companion versioned dialect
(VHLO) where an operator's version number only increases when its
behavior actually changes. One caveat noted: StableHLO's own spec is
prose/markdown only, not a machine-readable schema — unlike ONNX, so it
isn't the model to follow for the "programmatically queryable" part.

Sources: [StableHLO spec](https://openxla.org/stablehlo/spec),
[Compatibility RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20230623-compatibility.md)

### MLIR — the verification philosophy

An MLIR operator is identity + operands + results + attributes + regions.
Structural verification runs before each operator's own semantic
verifiers; type inference is opt-in via an interface
(`InferTypeOpInterface`) — the result is derived from the operands rather
than re-declared at every call site. This philosophy (verify rather than
trust) motivates a future `verify_fn` field on `OpSpec`, not yet built.

Source: [MLIR LangRef](https://mlir.llvm.org/docs/LangRef/)

### What wasn't adopted, and why

- **IREE (Stream/HAL)** — separates "what to compute" from "how/where to
  schedule it". NEURAX already has this exact structural split
  (`ParallelismIR` → `HardwareIR`); the lesson confirms the existing
  pipeline is already sound, it doesn't apply to OpSpec-IR itself.
- **LMHLO** — MHLO's "bufferized" form (values → buffers with liveness).
  NEURAX already has this split too (`TensorIR` → `MemoryIR`). Same
  conclusion: validating, not actionable here.
- **The full SSA pattern of a real compiler (LIFT, MLIR)** — regions,
  control flow, graph rewriting. Deliberately excluded: NEURAX's graphs
  are DAGs of atomic operations describing an architecture, not programs
  with loops, and NEURAX never rewrites a user's design (see
  `DiagnosticCode::W008`: it flags, it never silently fixes).

---

## Architecture

```
neurax-opspec/
├── README.md       // this document
├── Cargo.toml      // deps: neurax-parser + neurax-formulas (serde_json as a dev-dependency, GNN tests)
└── src/
    ├── lib.rs       // OpSpec, ParamsFn, FlopsFn, FlopsContext, ActivationMemoryFn, re-exports op_spec()/attention_rope_flops
    └── registry.rs  // the static table, one OpSpec per migrated LayerType (Custom excepted)
```

```rust
pub struct FlopsContext<'a> {
    pub global_params: &'a GlobalParams,       // GNN: num_nodes/num_edges
    pub image_channels: Option<usize>,         // Conv: fallback when there's no input_shape
    pub image_height: Option<usize>,
    pub image_width: Option<usize>,
}

pub struct OpSpec {
    pub layer_type: LayerType,
    pub params_fn: fn(&Layer) -> u64,
    pub flops_fn: fn(&Layer, usize, usize, &FlopsContext) -> f64,   // layer, batch, seq, ctx
    pub activation_memory_fn: Option<fn(&Layer, usize, usize, &str) -> u64>, // layer, batch, seq, dtype
}

pub fn op_spec(layer_type: LayerType) -> Option<&'static OpSpec>;
```

`op_spec()` returns `None` only for `Custom` (a permanent exception, see
above) — that's what keeps the migration safe: a type missing from the
table would keep resolving exactly as before, in `neurax-ir`'s old files,
if such a type still existed.

`FlopsFn` was extended twice during the full migration, exactly as the
premortem anticipated ("don't freeze the signature before a second,
structurally different family"):

- `&LayerParams` → `&Layer`: `Dense`/LoRA/DoRA need the layer's real
  `input_shape`/`output_shape`, not just its hyperparameters.
- Added `&FlopsContext`: GNN reads `num_nodes`/`num_edges` from
  `global_params.extra` (nothing on the layer itself carries them), and
  `Conv` falls back to `data.image_channels/height/width` when no 4-D
  `input_shape` is known.

A third axis, `activation_memory_fn`, was added as an `Option` rather than
folded into `FlopsFn`: only six types (`Embedding`, `Attention`, `Mlp`,
`Dense`, `LoraLinear`, `DoraLinear`) tracked a real activation-memory cost
before the migration — every other type returned a hardcoded `0`. `None`
reproduces that exact `0` without forcing every entry to supply a useless
closure.

## How it fits into the current structure

Two entry points in `neurax-ir` consult the table — `Custom` is the only
one that still falls back to its historical logic:

```rust
// neurax-ir/src/architecture/mod.rs::calculate_layer_params
pub fn calculate_layer_params(layer: &Layer) -> u64 {
    if let Some(spec) = neurax_opspec::op_spec(layer.layer_type) {
        return (spec.params_fn)(layer);
    }
    match layer.layer_type { /* Custom only */ }
}
```

```rust
// neurax-ir/src/operator/pass.rs::decompose_layer_to_ops
fn decompose_layer_to_ops(layer: &LayerDef, batch: usize, seq: usize, dtype: &str, ctx: &NeuraxContext) -> Vec<AtomOp> {
    if let Some(spec) = neurax_opspec::op_spec(layer.layer_type) {
        let parser_layer = to_parser_layer(layer);
        let flops_ctx = neurax_opspec::FlopsContext { /* from ctx.config */ };
        let flops = (spec.flops_fn)(&parser_layer, batch, seq, &flops_ctx);
        let activation_memory = spec.activation_memory_fn
            .map(|f| f(&parser_layer, batch, seq, dtype))
            .unwrap_or(0);
        // OpType stays a decision made here, in neurax-ir (op_type_for()),
        // which owns it: this crate depends on no neurax-ir type.
        let mut ops = vec![AtomOp { op_type: op_type_for(&parser_layer), flops, activation_memory, ... }];
        // Attention is the one exception to "one migrated type = one AtomOp":
        // RoPE stays a second, real op, see below.
        if layer.layer_type == LayerType::Attention {
            ops.push(/* RoPE AtomOp, via neurax_opspec::attention_rope_flops */);
        }
        return ops;
    }
    match layer.layer_type { /* Custom only */ }
}
```

For every migrated type, the corresponding old `match` arm was
**deleted**, replaced by an explicit `unreachable!()` arm documenting why
it should never run — never left as a duplicate (dead code is forbidden
in this project).

`tensor/shape_inference.rs` still isn't wired in: no migrated type has a
real output shape today (they all fall back to a passthrough) — a
`shape_fn` would join `OpSpec` if a family genuinely needed one, which
none did during this migration.

### Migration status

**Migration complete — every real `LayerType`, except `Custom`, is in the
table.**

| Family | Types |
|---|---|
| Transformer | `Embedding`, `Attention`, `Mlp`, `Dense`, `LoraLinear`, `DoraLinear`, `Normalization`, `Conv`, `Pooling` |
| CNN blocks | `ResidualBlock`, `ResnetBottleneck`, `Mbconv`, `Inception`, `DenseBlock`, `ConvnextBlock`, `ShuffleUnit`, `C2f`, `Detection`, `Transition` |
| MoE | `MoE`, `MoeRouter`, `MoeCombine`, `MoeSharedExpert` |
| SSM | `MambaBlock`, `S4Block`, `StateSpace`, `H3Block`, `RwkvBlock`, `RetentionBlock` |
| GAN | `GeneratorBlock`, `DiscriminatorBlock`, `ProgressiveBlock`, `SelfAttention`, `StyleMod`, `AdaIN`, `PixelNorm`, `MinibatchStd`, `SpectralNorm` |
| RNN | `LstmBlock`, `GruBlock`, `RnnCell`, `Bidirectional`, `EncoderBlock`, `DecoderBlock` |
| Diffusion | `UnetBlock`, `ResnetBlock`, `TimeEmbedding`, `TimestepBlock`, `CrossAttention`, `DownBlock`, `UpBlock`, `MidBlock`, `ConditionBlock`, `NoisePredictor`, `VaeEncoder`, `VaeDecoder` |
| GNN | `GraphConvNet`, `MessagePassing`, `GraphAttentionNet`, `RgcnConv` |
| — | `Custom`: permanent exception (evaluates a user-supplied equation, reports `neurax-ir` diagnostics — outside this crate's scope) |

### Real bugs found during the migration

- **`ResidualBlock` / `ResnetBottleneck`** (CNN, first wave). One shared
  FLOPs `match` arm used `out_channels.unwrap_or(256)` — correct for
  `ResnetBottleneck`, wrong for `ResidualBlock` alone (whose params side
  uses `unwrap_or(64)`). Fixed by splitting the two into distinct entries,
  each with its own matching defaults.
- **`Conv`** (full migration). The pre-migration FLOPs arm derived
  `kernel_w` from `kernel_h` — itself always just `kernel_size`, never
  reading `params.kernel_h` — while the params arm read `kernel_h`/
  `kernel_w` directly. A `Conv` layer with an explicitly non-square kernel
  (`kernel_h=7, kernel_w=7`, `kernel_size` unset) was silently costed as a
  `(3, 7)` kernel. Fixed by having both formulas read the exact same two
  fields, in the same order — see
  `conv_flops_reads_kernel_h_and_kernel_w_not_just_kernel_size` in
  `registry.rs` and `conv_flops_reads_the_fixed_kernel_shape` in
  `neurax-ir`'s before/after test.

One pre-existing approximation was **deliberately preserved**, not
"fixed": `RwkvBlock`/`RetentionBlock` share `MambaBlock`'s FLOPs formula
(`neurax_formulas::ssm` has no `rwkv_flops`/`retention_flops` of its own
yet), even though each has its own real params formula. Inventing a
correct RWKV/RetNet FLOPs formula is real work for `neurax-formulas`, not
something this migration should decide silently.

### Before/after test

`neurax-ir/tests/opspec_full_migration_before_after.rs` pins, family by
family, the (params, FLOPs) values obtained for a representative layer of
every migrated type. These values were captured from the pre-migration
code (by temporarily isolating `op_spec()` to the CNN blocks only, just
for the capture) and diffed byte-for-byte against post-migration values:
identical everywhere, except `Conv` — the one expected difference,
documented above.

---

## Risks flagged by premortem, and what became of them

*(An exercise run before writing any code: "it's 12 months from now, this
effort has failed — why?" The "Outcome" column was filled in after the
fact, once the migration was complete.)*

| Risk | Proposed answer | Outcome |
|---|---|---|
| Migration stays partial forever (two systems coexist indefinitely) | Each family migrated independently, tested and committed separately — never a big bang. | ✅ Resolved — all 9 families (Transformer, CNN, MoE, SSM, GAN, RNN, Diffusion, GNN) are migrated; `Custom` stays alone, by permanent design. |
| The signature doesn't scale once a non-uniform family arrives | Don't freeze `FlopsFn` before a second, structurally different family. | ✅ Confirmed necessary, twice: `&LayerParams` → `&Layer` (Dense/LoRA need `input_shape`/`output_shape`), then adding `FlopsContext` (GNN needs `global_params.extra`, `Conv` needs `data.image_*`). A third axis (`activation_memory_fn`) also had to be added, as an `Option` — see "How it fits in". |
| The registry itself becomes a stale source | Old `match` arms deleted, not left as duplicates — `unreachable!()`. | ✅ Done for both files (`architecture/mod.rs`, `operator/pass.rs`) throughout the migration. |
| False sense of security — centralizing removes the disagreement signal that found bugs by accident | Keep the independent native cross-checker in place, uncoupled. | ✅ `neurax-core/tests/native_flops_cross_check.rs` unchanged. The signal actually worked again during this migration: the `Conv` bug (kernel_h/kernel_w) was found precisely by confronting the two old arms against each other. |
| Migration effort underestimated | Already confirmed during the CNN prototype. | ✅ Confirmed a second time: the `Attention` case (see "How it fits in") forced a walk-back of the "one migrated type = one `AtomOp`" simplification after an existing regression test (`kernel_launch_count_reflects_real_ops.rs`) revealed it changed real behavior (the kernel-launch count, which feeds a real latency-overhead model). |
| Nothing prevents human regression (a new type added outside the registry) | Deferred until two families were migrated. | ✅ Resolved — `every_real_layer_type_except_custom_is_registered` (in `registry.rs`) fails if a future `LayerType` is added to the parser without being registered here. |
| A poorly calibrated new inter-crate dependency | Dependency direction fixed at design time (`neurax-opspec` upstream, never the reverse). | ✅ Held throughout the migration — `neurax-opspec` depends only on `neurax-parser` + `neurax-formulas` (+ `serde_json` as a dev-dependency for the GNN tests). |

---

## What this crate will never do

- Generate executable code or a textual IR.
- Rewrite a user's design (NEURAX flags, it never silently fixes — see
  `DiagnosticCode::W008`).
- Replace `neurax-ir`'s 10-pass pipeline — it stays consulted BY it, never
  one more stage inside it.
- Decide an `OpType` or build an `AtomOp` — those are `neurax-ir`
  concepts, decided where they're consumed (`op_type_for()` in
  `operator/pass.rs`).

## License

Proprietary — closed-source, commercial software. All rights reserved.
Not published to crates.io (`publish = false`).
