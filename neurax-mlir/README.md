# neurax-mlir

**MLIR dialects for the NEURAX compiler.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. Defines 10 custom MLIR dialects (Architecture, Graph, Tensor, Operator, Compute, Memory, Hardware, Parallelism, Cost, Report) plus lowering passes and targets (CPU, CUDA, ROCm, Vulkan, Metal), built on [melior](https://crates.io/crates/melior) / LLVM 18.

> **Requires LLVM 18 / MLIR 18.** Set the following environment variables to build:
> ```bash
> export LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
> export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
> export TABLEGEN_180_PREFIX=/usr/lib/llvm-18
> ```

## Features

- 10 custom MLIR dialects mirroring the 10-pass analytical IR
- Lowering passes for CPU, CUDA, ROCm, Vulkan, Metal targets
- IREE kernel lowering groundwork
- Deterministic, analytical compilation

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.