//! # OpSpec-IR
//!
//! **One definition per operation, instead of three independent ones.**
//!
//! Before this crate, the same `LayerType` was dispatched three times, in
//! three separate files of `neurax-ir`, with nothing forcing them to agree:
//! `architecture/mod.rs::calculate_layer_params` (parameter count),
//! `operator/pass.rs::decompose_layer_to_ops` (FLOPs), and
//! `tensor/shape_inference.rs` (output shape). Every real bug found in a
//! CNN block, S4/H3, an RNN cell, or a GNN layer during the 2026-09-01/02
//! audit had the same shape: one side got fixed, the other silently
//! didn't, because nothing connected them.
//!
//! A type registered here bundles its parameter-count and FLOPs formulas
//! in one place — adding one without its counterpart is a compile error,
//! not a latent bug. See `README.md` for the full design rationale
//! (concepts it draws on, integration plan, premortem-driven safeguards).
//!
//! Every family — Transformer, MoE, SSM, GAN, RNN, Diffusion, GNN, and the 10
//! CNN blocks — is migrated (see [`registry`]). `Custom` is the one
//! deliberate, permanent exception: it evaluates a user-supplied equation
//! and reports parser diagnostics when that equation is missing or invalid,
//! which is `neurax-ir`-specific behavior this crate has no business owning.
//!
//! This crate deliberately depends on nothing from `neurax-ir` (only
//! `neurax-parser` and `neurax-formulas`, the same dependency shape
//! `neurax-formulas` itself has) — `neurax-ir` depends on this crate, never
//! the other way around. The `OpType` a migrated layer's `AtomOp` should
//! carry stays a decision made in `neurax-ir`, which owns that type; this
//! crate only answers "how many parameters" and "how many FLOPs".

mod registry;

pub use registry::{attention_rope_flops, op_spec};

use neurax_parser::{GlobalParams, Layer, LayerType};

/// Computes a layer's real parameter count from its own fields — the exact
/// signature `neurax-ir`'s `calculate_layer_params` already used per arm,
/// so migrating an arm here is a cut-and-paste, not a rewrite.
pub type ParamsFn = fn(&Layer) -> u64;

/// Context a [`FlopsFn`] may need beyond the layer itself and the batch/
/// sequence shape it runs at. Two families, migrated after the CNN blocks,
/// turned out to need more than a layer's own fields:
///
/// - GNN reads its real graph size from `global_params.extra`
///   (`num_nodes`/`num_edges`) — the same map `GlobalResolutionContext`
///   reads for the params side — because no per-layer field carries it.
/// - `Conv` falls back to `data.image_channels/height/width` when a real
///   4-D `input_shape` isn't tracked on the layer, exactly as it did before
///   migrating.
///
/// A migrated type that needs neither simply ignores this parameter — most
/// do.
pub struct FlopsContext<'a> {
    pub global_params: &'a GlobalParams,
    pub image_channels: Option<usize>,
    pub image_height: Option<usize>,
    pub image_width: Option<usize>,
}

/// Computes a layer's forward FLOPs from the layer itself (its params, and —
/// for `Dense`/LoRA/`Conv` — its real `input_shape`/`output_shape`), the
/// batch/sequence (or spatial-side-via-`sqrt(seq)`, the same stand-in every
/// CNN block and most other families already used) shape it runs at, and
/// [`FlopsContext`] for the few families that need more. Matches the
/// closures already inline in `neurax-ir`'s `decompose_layer_to_ops`.
pub type FlopsFn = fn(&Layer, usize, usize, &FlopsContext) -> f64;

/// Computes the size, in bytes, of the one activation tensor `neurax-ir`'s
/// original per-arm formulas tracked for a handful of types (`Embedding`,
/// `Attention`, `Mlp`, `Dense`, LoRA/DoRA) — real bytes that feed the
/// compute pass's memory-bandwidth estimate
/// (`compute/pass.rs`'s `param_bytes`), not a cosmetic figure. Every other
/// migrated type tracked `0` before migrating (no per-layer activation
/// tensor was ever costed for them), which is exactly what
/// `OpSpec::activation_memory_fn: None` reproduces — so this stays an
/// `Option`, not a required field every entry must fill in with a fake
/// zero-returning closure.
pub type ActivationMemoryFn = fn(&Layer, usize, usize, &str) -> u64;

/// One operation's complete definition: how many parameters it owns, how
/// expensive one forward pass through it is, and — for the few types that
/// need it — how many bytes its activation tensor costs. Deliberately does
/// not carry an `OpType` or any other `neurax-ir`-specific concept — see the
/// module doc for why.
pub struct OpSpec {
    pub layer_type: LayerType,
    pub params_fn: ParamsFn,
    pub flops_fn: FlopsFn,
    pub activation_memory_fn: Option<ActivationMemoryFn>,
}
