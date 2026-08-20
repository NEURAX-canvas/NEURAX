# NEURAX

**The Analytical Compiler for Neural Architectures**

NEURAX predicts the **cost, memory, and performance** of neural network architectures **before training** — in under 50 ms, with zero GPU, and fully deterministically.

- [Live Demo](https://rustnew.github.io/NEURAX/)
- [GitHub Repository](https://github.com/rustnew/NEURAX)
- [Releases](https://github.com/rustnew/NEURAX/releases)

## About

NEURAX is an analytical compiler for neural network architectures. Whereas training frameworks (PyTorch, TensorFlow) execute models and runtime compilers (IREE, OpenXLA) lower them for execution, NEURAX operates at **design time**: it answers the questions you need resolved before committing GPU resources.

- Will this architecture fit in VRAM?
- What is the training cost on 8x H100?
- Where are the memory bottlenecks?
- Is inference stable? What is the hallucination risk?
- Which parallelism strategy is optimal?

All in under 50 ms. Zero GPU required. Fully deterministic.

## Key capabilities

- **11 architecture families** — Transformer, CNN, MoE, SSM, Diffusion, GNN, GAN, RL, SNN, RNN, Experimental.
- **208 configurable blocks** and **88 reference templates**, both counted
  against the real catalogue by a test, not stated from memory.
- **10-pass analytical IR pipeline** producing 66 metrics.
- **MLIR / LLVM 18** compiler backend with 14 custom dialects.
- **Visual design canvas**, **AI copilot agent**, **Inference Intelligence** and **Time Machine**.
- **Export** to JSON, NEURAX IR, and GitHub — a canvas used to offer four more
  targets (PyTorch, ONNX, Triton, Rust/Burn); their generated code didn't run,
  so they were removed rather than left to mislead anyone who tried them.

## Documentation

Use the sidebar to navigate the full documentation set. Installing the
[Desktop App](DESKTOP.md) is the fastest way to run NEURAX yourself, on
Linux, macOS or Windows. Start with the [Architecture & Design](DESIGN.md)
chapter to understand how NEURAX works internally, then the
[API Reference](API_REFERENCE.md) and [Deployment Guide](DEPLOYMENT.md) to
run the server side yourself.
