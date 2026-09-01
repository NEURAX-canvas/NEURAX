<p align="center"><img src="images/neurax-logo.svg" alt="NEURAX logo" width="96" height="96"></p>

# NEURAX

**An environment to design, simulate, and optimize neural architectures — before you spend a GPU-hour on one**

NEURAX predicts the **cost, memory, and performance** of neural network architectures **before training** — in under 50 ms, with zero GPU, and fully deterministically.

- [Live Demo](https://neurax-canvas.github.io/NEURAX/)
- [GitHub Repository](https://github.com/rustnew/NEURAX)
- [Releases](https://github.com/rustnew/NEURAX/releases)

## About

NEURAX is an environment for designing, simulating, and optimizing neural network architectures — a visual canvas, an AI copilot, and a Rust analytical engine underneath, not a single command-line tool. Whereas training frameworks (PyTorch, TensorFlow) execute models and runtime compilers (IREE, OpenXLA) lower them for execution, NEURAX operates at **design time**: it answers the questions you need resolved before committing GPU resources.

- Will this architecture fit in VRAM?
- What is the training cost on 8x H100?
- Where are the memory bottlenecks?
- Is inference stable? What is the hallucination risk?
- Which parallelism strategy is optimal?

All in under 50 ms. Zero GPU required. Fully deterministic.

## Key capabilities

- **8 architecture families, fully supported** — Transformer, CNN, MoE, SSM, Diffusion, GNN, GAN, RNN — every
  metric, not just parameter count. Four more (`Hybrid`, `Multimodal`, `Snn`, `Experimental`) exist as
  `ModelType` values the schema still parses — a hand-written or imported JSON that declares one still
  gets a report — but aren't offered as a family choice in the canvas: none has the compiler's own
  dedicated formulas behind it, only a generic layer-level fallback, so exposing it as a first-class
  choice would have left it partially supported.
- **64 reference templates**, counted against the real catalogue by a test, not stated from memory.
- **11-phase analytical IR pipeline** — three of the eleven phases run concurrently on a shared
  thread pool since none of them reads another's output. Produces ~76 fixed scalar metrics plus
  per-layer breakdowns that scale with model depth — see [Architecture & Design](DESIGN.md#a-note-on-how-many-metrics--there-is-no-single-honest-number)
  for why this project has previously stated five different single numbers for that count, and why none of them was the full answer.
- **MLIR** dialect emission (14 custom dialects, real per-layer formulas) as a standalone,
  independently-testable crate — not part of the request path a user's analysis takes; see
  [Architecture & Design](DESIGN.md#neurax-mlir) for exactly what is and isn't wired up.
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
