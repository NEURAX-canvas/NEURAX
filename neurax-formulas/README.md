# neurax-formulas

**Analytical formulas for ML operations — the math engine of NEURAX.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural network architectures. This crate contains the pure, deterministic closed-form formulas that compute FLOPs, parameters, memory and latency for every neural network layer type — **without a GPU, without training, without running the model**.

> **The vision:** in a world where training cost is decided before training begins, someone has to answer "what will this cost?" with numbers, not estimates. NEURAX answers with closed-form math. This crate is that math.

## Why analytical formulas?

| Approach | GPU needed | Deterministic | Speed | Before training? |
|----------|-----------|---------------|-------|------------------|
| **NEURAX (analytical)** | ❌ No | ✅ Yes | < 1ms | ✅ Yes |
| PyTorch execution | ✅ Yes | ❌ No | minutes-hours | ❌ No |
| Heuristics/rule-of-thumb | ❌ No | ❌ No | instant | ⚠️ Rough |

## Installation

```toml
[dependencies]
neurax-formulas = { git = "https://github.com/rustnew/NEURAX", package = "neurax-formulas" }
```

## Quick start

```rust
use neurax_formulas::{
    attention_flops, dtype_bytes, gated_mlp_flops, gated_mlp_params,
    mlp_flops, mlp_params,
};

// LLaMA-style decoder block (hidden=4096, intermediate=11008, bf16)
let hidden = 4096;
let intermediate = 11008;
let batch = 32;
let seq_len = 2048;

// MLP: standard (GELU) vs gated (SwiGLU — used in LLaMA)
let standard = mlp_flops(batch, seq_len, hidden, intermediate, "gelu");
let gated = gated_mlp_flops(batch, seq_len, hidden, intermediate, "silu");
let swiglu_params = gated_mlp_params(hidden, intermediate, false);
println!("standard MLP:  {:.2e} FLOPs", standard);
println!("SwiGLU MLP:    {:.2e} FLOPs", gated);
println!("SwiGLU params: {} (3×H×I, +50% vs standard)", swiglu_params);

// Attention (GQA-style)
let flops = attention_flops(batch, seq_len, hidden, 32, /* causal */ true);
println!("causal attention: {:.2e} FLOPs", flops);

// Dtype accounting
assert_eq!(dtype_bytes("fp32"), 4);
assert_eq!(dtype_bytes("bf16"), 2);
```

## Supported layer families

| Module | Functions |
|--------|-----------|
| Attention | `attention_flops`, `flash_attention_flops`, `gqa_flops`, `attention_params`, `gqa_params` |
| MLP | `mlp_flops`, `gated_mlp_flops`, `mlp_params`, `gated_mlp_params` |
| MoE | `moe_flops`, `sparse_moe_flops`, `moe_params`, `moe_params_with_shared`, `moe_expert_utilization`, `moe_router_flops` |
| Convolution | `conv*` (in `conv` module) |
| CNN blocks | `cnn_blocks` module |
| Embedding | `embedding` module |
| Normalization | `normalization` module (LayerNorm, RMSNorm, BatchNorm) |
| RNN / SSM | `rnn`, `ssm` modules |
| GNN | `gnn` module |
| Diffusion | `diffusion` module |
| Utilities | `dtype_bytes`, `backward_flops_multiplier`, `optimizer_flops_multiplier` |

## Using it with the full pipeline

`neurax-formulas` is the leaf crate of the [NEURAX ecosystem](https://github.com/rustnew/NEURAX):

```text
neurax-parser ─┐
neurax-formulas ─┤→ neurax-ir (10-pass analytical IR) → neurax-core (pipeline)
neurax-hardware-db ─┘
```

For end-to-end analysis (parse JSON → 55+ metrics), use [`neurax-core`](../neurax-core) instead:

```rust
use neurax_core::analyze_json;

let result = analyze_json(model_json)?; // <50ms, deterministic
println!("{:#?}", result.compute.total_flops);
```

## Determinism guarantee

Same input → same output. Always. Formulas are pure functions: no randomness, no floating-point accumulation order dependence, no hardware-dependent behavior. This makes NEURAX results auditable and reproducible in CI.

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.