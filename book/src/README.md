# NEURAX

**The Analytical Compiler for Neural Architectures**

NEURAX predicts the **cost, memory, and performance** of neural network architectures **before training** — in under 50 ms, with zero GPU, and fully deterministically.

- [Live Demo](https://neurax.ai)
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
- **680+ configurable blocks** and **88 reference templates**.
- **10-pass analytical IR pipeline** producing 55+ metrics.
- **MLIR / LLVM 18** compiler backend with 13 custom dialects.
- **Visual design canvas**, **AI copilot agent**, **Inference Intelligence** and **Time Machine**.
- **Export** to PyTorch, ONNX, Triton, MLIR, Rust/Burn, JSON and Network Graph.

## Documentation

Use the sidebar to navigate the full documentation set. Start with the [Architecture & Design](DESIGN.md) chapter to understand how NEURAX works internally, then the [API Reference](API_REFERENCE.md) and [Deployment Guide](DEPLOYMENT.md) to run it yourself.
