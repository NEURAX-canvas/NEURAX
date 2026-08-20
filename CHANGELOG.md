# Changelog

All notable changes to **NEURAX** — The Pre‑Flight Compiler for Artificial Intelligence are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.9.0] — 2026-08

### Added
- Import Mamba and Mamba-2 models directly from HuggingFace — a new family
  (no attention, no `num_attention_heads` to look for), field names verified
  against real live configs on the Hub.
- Publish metadata in the HuggingFace import dialog: author, license,
  download/like counts, and publish date, fetched from the Hub's model-info
  API alongside the config — verified live to be public and CORS-enabled the
  same way the existing raw-config fetch is.
- Honest refusal for HuggingFace families NEURAX doesn't import yet (CNN,
  ViT, Whisper, diffusers pipelines, RWKV, ...): names the real family
  instead of reporting a generic "no hidden size in this config".
- TOML as a third export format, alongside JSON and NEURAX IR — the same
  compiled topology, round-tripped losslessly (verified structurally, not
  just visually).
- Rename an open design in place, from the header, without a full Save As.
- Auto-arrange: untangle overlapping or crowded canvas blocks with one
  click, following the actual connections between them (dagre), instead of
  dragging each one apart by hand.

### Fixed
- Seven Mamba/Mamba-2 reference templates were computing roughly double
  their real parameter count: a visual six-node decomposition of one block
  had two nodes each independently carrying the block's *whole* formula,
  while the node that was conceptually the block's core carried none of it.
  Rebuilt as one node per block, carrying the one real formula. Three of the
  five Mamba sizes also stated half their real layer count (130M: 12 vs. the
  real 24).
- The canvas's connection-delete button was drawn inside its own
  connection's group, where a *different* connection's wide invisible
  click-target — rendered later — could paint on top of it and swallow a
  click a few pixels off centre: visibly selected, red, and unreliable,
  which read as "doesn't work" rather than "works most of the time."
  Rendered in its own pass, after every connection, so nothing painted
  later sits on top of it.
- Compilation warnings accumulated across repeated analysis runs instead of
  being replaced — the same architectural issue re-analysed twice showed
  its warning twice.
- Changing account avatar didn't reach anywhere the avatar is actually
  displayed: three separate storage locations all claimed to be "the
  avatar", and only one of them (never the one either "save avatar" affordance
  wrote to) was ever read by the header.
- The model-templates panel had no visible scrollbar over 88 reference
  architectures — same fix already applied to the hyperparameter panel,
  now applied here too.
- `neurax-mlir`'s per-layer parameter and FLOPs formulas were a second, separate,
  cruder set from the ones `neurax-formulas`/`neurax-ir` actually use — e.g.
  attention's parameter count omitted the ×4 for Q/K/V/O projections — and
  only 5 of ~40 layer types were handled at all. Now reuses the real formulas
  directly rather than re-deriving an approximation of them.
- ResNet-50's reference template approximated its four stages as attention
  blocks (no bottleneck-block layer type existed); parameter accuracy against
  the published 25.6M figure went from +3.53% to -0.04%. Along the way, every
  BatchNorm-bearing CNN block formula (ResNet, MobileNet, DenseNet, ConvNeXt,
  ShuffleNet, YOLO's C2f) was counting BatchNorm's non-trainable running
  statistics as if they were trainable parameters.
- CI's `build-and-test` job was failing on every PR with "No space left on
  device" — the cargo cache had grown to 4.26 GB from caching `target`
  alongside the registry, which combined with the LLVM/MLIR toolchain
  exceeded the runner's disk. Stopped caching `target`; added a cleanup step
  freeing the preinstalled toolchains the workspace never uses.
- The desktop release pipeline never built Windows — confirmed against every
  past release (v0.7.1 through v0.8.0): zero Windows assets, ever, despite
  the desktop README documenting a Windows installer. Added `windows-latest`
  to the build matrix and verified a real run produces a real installer.

### Changed
- `book/src/DESIGN.md`'s architecture diagrams no longer show `neurax-mlir`
  wired into the live request path (`CORE --> MLIR --> LLVM/IREE`, generating
  MLIR for every analysis) — it isn't; nothing in the shipped product calls
  it. Also replaced a reference to `neurax-cli`, a crate removed from the
  repository, with `neurax-desktop`, which ships and was previously absent
  from the diagram entirely.
- Added `book/src/DESKTOP.md`: install instructions for Linux, macOS, *and*
  Windows. The root README pointed to "the documentation" for Windows
  install steps that the documentation didn't actually have.

## [0.8.0] — 2026-08

### Added
- Local-only account: a name, an avatar, and an id, created automatically on
  first launch and kept on this machine — no cloud identity provider, no
  project to configure, for both the web and desktop builds alike. Replaces
  the previous Supabase-backed account, whose configured project had no DNS
  record at all (confirmed live), so every signup and signin failed by
  default. `@supabase/supabase-js` and all `VITE_SUPABASE_*` configuration
  are removed from the frontend entirely; the main web bundle dropped from
  320 kB to 155 kB gzipped as a result.
- Provider support for Google (Gemini), Mistral, Fireworks AI, DeepSeek, and
  GLM (Zhipu) in the AI agent, alongside the existing OpenAI and Anthropic —
  real brand icons in the provider picker instead of emoji.
- `active_parameters` metric: for Mixture-of-Experts models, the parameters
  actually touched per token (routed subset + shared experts + dense layers),
  separate from the total parameter count.
- A multi-precision weight-memory comparison chart (FP32/FP16/BF16/INT8/INT4)
  using the compiler's own byte-per-parameter figures.
- Import a model directly from the HuggingFace Hub by ID or URL — fetches
  `config.json` straight from `huggingface.co`, no copy-paste round trip.
- Model `first_k_dense_replace` (DeepSeek-style MoE: dense layers before the
  routed ones) parsed and compiled as its own block, at the model's real
  dense width.

### Fixed
- Mixture-of-Experts models were mis-costed on two independent bugs: a
  router/expert-combine decomposition that costed the router as if it held
  full expert weights, and a depth-scaling pass that diluted per-layer
  costs when a canvas MoE block is drawn as several nodes (router, experts,
  combine, shared-expert) representing one logical layer. Mixtral 8x7B and
  DeepSeek-MoE-16B now match their published sizes within about 1%, verified
  against a live compiler and cross-checked with an independent reference
  implementation, not just the code under test.
- A separate MoE FLOPs bug: every argument after `hidden_size` was shifted
  one position from the real function signature, computing a value that
  was off by 4–9 orders of magnitude while still compiling cleanly, because
  every argument happened to share the same Rust type.
- Destructive canvas keyboard shortcuts (Delete, Ctrl+A, Ctrl+G, Ctrl+D)
  no longer fire while typing in a text field.
- HuggingFace configs with an abbreviated multimodal `text_config` (a bare
  `{"model_type": "llama"}` with no width) now get a specific, actionable
  error instead of a generic "no hidden size found".

## [0.7.4] — 2026‑08
Restore the Production, Inference, and Time Machine workspaces to their
working state after the simulation-workspace rebuild.

## [0.7.3] — 2026‑08
Desktop: undo/redo, `.neurax` project documents, HuggingFace import, A/B
architecture comparison, and an in-app guide.

## [0.7.2] — 2026‑08
Fix the architecture family name rendering as a raw code fragment in the UI.

## [0.7.1] — 2026‑08
- Remove the standalone `neurax-cli` crate — `neurax` is the desktop
  application; there is no separate CLI binary.
- Installer: a banner and a line stating what NEURAX is for.
- Hyperparameter panel: a visible scrollbar, and remove the OSS plan badge.

## [0.7.0] — 2026‑07
The desktop application, and a pass through the whole product correcting
numbers that didn't hold up:
- NEURAX as an installable desktop app (Tauri): one-command install,
  project persistence, CORS and window-control fixes, parity tests against
  the web build.
- Reference models corrected to match their published parameter counts;
  the compile summary no longer reports two contradicting numbers; the
  README no longer advertises figures the repository doesn't hold.
- Bring-your-own-key path for the AI agent, with current Claude models and
  a way to point it at a gateway.
- Landing page rewritten from scratch; several UI defects fixed by walking
  every workspace tab by hand.
- Simulate the model that is actually on the canvas, not an assumed one.

## [0.6.3] — 2026‑08

### Changed
- Replace generic emojis with the **Notionists** avatar family across the UI.
- Add **Multimodal (VLM)** support: new `ModelType::Multimodal` for
  vision+language models (CLIP, LLaVA-style, mobile VLMs) — parser, agent
  planning template (parallel vision+text branches with fusion), and tests.
- Fix parameter calculation: fall back to `ffn_dim`/`num_heads` when
  `intermediate_size`/`num_attention_heads` are absent.
- Add `.cargo/config.toml` pointing MLIR crates at LLVM 18.
- Add MIT `LICENSE`, `ROADMAP_NEURAX_2.0.md`, deployment scripts, and
  `examples/models` reference configs.

## [0.6.2] — 2026‑08

### Changed
- Refactor README, increase fonts, integrate Notionists avatars system.

## [0.6.1] — 2026‑07

### Added
- Modernized landing page with 5 components + Notionists avatar system.
- **Hyperparameter Optimization** system (3 strategies, 11 families,
  hardware-aware).
- LLM-builder implementation and testing.

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
