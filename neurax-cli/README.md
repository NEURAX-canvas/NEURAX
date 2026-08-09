# neurax-cli

**Command-line interface for NEURAX.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. Analyze a neural architecture from the terminal and get FLOPs, VRAM, cost, energy and carbon predictions — before training, with zero GPU.

## Install

```bash
cargo install neurax-cli
```

> The binary is named `neurax`.

## Usage

```bash
# Analyze an architecture
neurax analyze models/gpt2_small.json

# With the MLIR backend (requires LLVM 18)
neurax analyze models/gpt2_small.json --mlir
```

## Features

- Deterministic <50 ms analysis of 8B-parameter models
- 55+ metrics (FLOPs, VRAM, latency, cost, energy, carbon)
- Optional `mlir` feature for MLIR backend lowering (requires LLVM 18)

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.