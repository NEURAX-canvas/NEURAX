# Changelog

All notable changes to **NEURAX** — The Pre‑Flight Compiler for Artificial Intelligence are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] — 2026‑07

### ✅ Complete (v0.5.0)

| Phase | Feature | Status |
|---|---|---|
| Core | 10‑pass analytical IR pipeline | ✅ |
| Core | MLIR compiler backend (13 dialects, LLVM 18) | ✅ |
| Core | CLI: `analyze`, `compile`, `validate`, `summary` | ✅ |
| Core | Hardware database (20 GPUs, 2 CPUs, 5 interconnects) | ✅ |
| Core | ONNX binary export | ✅ |
| Core | Streaming SSE analysis API | ✅ |
| Core | Inference simulation pass (22 parameters, 10 widgets) | ✅ |
| Core | Dynamic analysis (virtual memory, stability, behavioral synthesis) | ✅ |
| Core | Multi‑hardware comparison (up to 8 configurations) | ✅ |
| Core | Time Machine cost/carbon projection | ✅ |
| Core | Regulatory compliance configuration | ✅ |
| Core | Credits system with plan‑based limits | ✅ |
| Core | API key management (scopes, revocation) | ✅ |
| Core | Stripe billing integration | ✅ |
| Core | Supabase auth integration | ✅ |
| Core | GitHub export with PR creation | ✅ |
| Core | Plugin validation endpoint | ✅ |
| Core | Presets (88 reference templates) | ✅ |
| Web | React 18 + TypeScript + Vite frontend | ✅ |
| Web | Visual canvas with React Flow | ✅ |
| Web | Real‑time metrics dashboard (40+ metrics) | ✅ |
| Web | AI Chat Drawer with agent integration | ✅ |
| Web | Hyperparameter Optimization (3 strategies, 6 objectives) | ✅ |
| Web | Cloud project CRUD | ✅ |
| Web | Multi‑hardware comparison UI | ✅ |
| Agent | Architecture planning via FastAPI + LangChain | ✅ |
| Agent | 3‑phase declarative pipeline (plan → validate → materialize) | ✅ |
| Agent | Auto‑correction with retry (up to 3 attempts) | ✅ |
| Agent | Catalogue store with 11 model families | ✅ |
| MCP | Model Context Protocol server | ✅ |
| TUI | Ratatui terminal interface | ✅ |

### 🚧 In Progress

- **NEURAX‑MLIR → IREE kernels** — Lowering NEURAX MLIR dialects to runnable IREE kernels for cross‑platform deployment (CPU, CUDA, Vulkan, Metal, ROCm).
- **Public benchmark suite** — Validation set comparing analytical predictions against measured real‑world runs across 11 model families.
- **Batch HPO backend API** — Server‑side batch hyperparameter optimization via backend API (frontend already has client‑side HPO).

### 📋 Planned

- PostgreSQL persistence for cloud project storage
- Multi‑node distributed training projections
- Model Hub with HuggingFace integration
- Fine‑tuning cost projections (LoRA, QLoRA, full)
- Integration with actual training frameworks (PyTorch Lightning, HF Trainer)
- Collaborative multi‑user editing with CRDT

---

## [0.4.0] — 2026‑06

### Added

- Hyperparameter Optimization system with 3 strategies (Grid Search, Random Search, Bayesian), 6 objectives, and hardware‑aware recommendations across 11 model families.
- 14‑GPU frontend database (H200, GH200, H100 SXM/PCIe, A100 SXM/PCIe, L40S, L40, V100, RTX 4090/4080/3090, RTX A6000, T4).
- Hardware‑aware optimizer with VRAM, bandwidth, and ridge‑point capacity analysis.
- Inference Intelligence panel with 22 parameters and 10 widgets (stability, hallucination risk, sampling volatility).
- Time Machine compliance overlay with EU AI Act, CSRD, DSA regulatory data.
- GitHub export panel with direct push and PR creation.
- Credits system with plan‑based usage limits and billing integration.
- API key management with scope‑based authorization.

### Changed

- Aligned README claims with audited codebase reality.
- Replaced ASCII art diagrams with Mermaid diagrams for better rendering.

---

## [0.3.0] — 2026‑02

### Added

- Streaming SSE analysis with authentication.
- Multi‑hardware comparison (up to 8 configurations).
- Cloud project CRUD (create, read, update, delete).
- ONNX binary export.
- GitHub push with PR creation.
- Billing, credits, and compliance infrastructure.
- Docker multi‑service orchestration.

### Changed

- Web platform visual canvas, drag‑and‑drop, and live metrics.
- AI Chat Drawer with agent integration.

---

## [0.2.0] — 2025‑06

### Added

- 10‑pass analytical IR pipeline (Architecture → Graph → Tensor → Operator → Compute → Memory → Parallelism → Hardware → Cost → Report).
- MLIR compiler backend with 13 custom dialects (Architecture, Graph, Tensor, Operator, Compute, Memory, Parallelism, Hardware, Cost, Report, Training, Data, Optimization).
- CLI with `analyze`, `compile`, `validate`, and `summary` commands.
- Hardware database with 20 GPUs, 2 CPUs, and 5 interconnect specifications.
- 88 reference architecture templates across 11 model families.
- Terminal UI (TUI) for model compilation visualization.

---

## [0.1.0] — 2024‑12

### Added

- Initial analytical compiler framework.
- JSON model config parser with schema validation.
- Core IR pipeline with FLOPs, parameter count, memory, and cost formulas.
- MLIR code generation backend.
- Foundation for 11 model families: Transformer, MoE, CNN, SSM, Diffusion, GNN, GAN, RL, SNN, RNN, Experimental.
