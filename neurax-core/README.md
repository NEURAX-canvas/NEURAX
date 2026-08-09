# neurax-core

**Core pipeline and orchestration for NEURAX.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. Orchestrates the full analytical compilation pipeline: parse → 10-pass IR → report, plus ONNX export and optional MLIR lowering.

## Features

- End-to-end pipeline orchestration (parser → IR → report)
- Deterministic sub-50 ms analysis of 8B-parameter models
- 55+ metrics: FLOPs, VRAM, latency, cost, energy, carbon
- Export to 7 formats (PyTorch, ONNX, Triton, MLIR, Rust/Burn, JSON, Network Graph)
- Optional `mlir` feature for MLIR backend lowering

## Usage

```rust
use neurax_core::Neurax;

let neurax = Neurax::new();
let report = neurax.compile(model_json)?;
```

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.