# Changelog

All notable changes to **NEURAX** — The Pre‑Flight Compiler for Artificial Intelligence are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.13.0] — 2026-08

### Changed
- **The Rust compiler is now the single shape authority.** The frontend and
  agent used to compute or guess tensor shapes themselves before sending a
  design to the compiler; a model's `connections` (real graph edges) can now
  be sent instead, and a new per-family shape-inference pass
  (`neurax-ir::tensor::shape_inference`) walks the real graph topologically
  to derive every shape. `neurax-agent`'s own pre-flight self-check
  (`budget_check.spec_to_topology`) was rewritten to send the real graph
  instead of hand-threading a hidden dimension through sequence-family
  nodes only — the same path the frontend already used post-materialization.
- **Consolidated four independently-drifted fan-in/merge-capable-type
  lists into one.** The frontend, agent's catalogue store, and agent's
  constants module each hand-maintained their own list of which block types
  can accept multiple inputs; they had already drifted apart. All three now
  read `block_constraints.json`'s `merge_capable_types` as the single source
  of truth.

### Fixed
- **Diffusion models were never charged for their denoising steps.** A
  diffusion model's reported cost came from a single U-Net forward pass —
  `diffusion_timesteps` was parsed but never multiplied into any real
  calculation, and classifier-free guidance's second forward pass wasn't
  accounted for either. A 1000-step Stable Diffusion XL run was reported at
  roughly 1/1000th (or 1/2000th, with CFG) of its real inference cost. Fixed
  by wiring `diffusion_sampling_flops` into the live compute pass and adding
  `guidance_scale` as a real, parseable field.
- **RGCN was routed to a plain Conv2D formula**, not even the right GNN
  formula — `rgcn_conv` had no dedicated backend type and fell through the
  frontend's generic `.includes('conv')` catch-all. Added `RgcnConv` as a
  real type with its own params (`num_relations`, `num_bases`) and formula.
- **int4 quantization was costed at int8's memory footprint** — `dtype_bytes`
  returned `1` for int4 instead of `0.5`, a 2x memory overstatement for
  every int4-quantized model.
- **LoRA/DoRA adapters were costed as a full dense layer** instead of their
  actual low-rank parameter count — `lora_linear`/`dora_linear` had no
  dedicated type or `rank`/`lora_alpha` fields and collapsed to `dense`.
- **Sliding-window, dilated, and sparse attention were costed as dense
  O(seq²) attention.** Mistral-shaped long-context models in particular were
  reported at many times their real attention cost. `window_size`/
  `block_size`/`dilation` are now real fields that reduce the KV span the
  FLOPs formula sees.
- Squeeze-and-Excitation blocks were declared (`LayerParams.se`) but
  `mbconv_params`/`mbconv_flops` had no `se` parameter in their signature at
  all — the branch was structurally absent, not just unused.
- S4/H3/StateSpace blocks reused Mamba's selective-scan formula, which
  charges for input-dependent projection matrices those non-selective
  architectures don't have.
- Adafactor and Lion optimizers — both real, reachable choices in the
  hyperparameter dropdown — were costed as AdamW (2x parameter memory)
  instead of their real, much smaller memory footprint.
- Removed `sparse_moe_flops`: dead code with zero callers that silently
  discarded its own `capacity_factor` argument.
- MoE's active-parameter count was double-scaling shared-expert
  parameters; ResNeXt's `cardinality` (grouped-convolution width) was
  parsed but never read by the params/FLOPs formulas.
- `neurax-mcp`'s standalone server had three independent bugs found only by
  actually running it end-to-end, not by reading it: it failed to import at
  all against `mcp>=2.0` (pinned to `<2.0.0`), sent graph edges under a
  `graph.edges` key the compiler's schema doesn't have (silently dropped —
  fixed to `connections`), and carried a small, stale layer-type vocabulary
  missing MoE/GNN/SSM/RGCN/LoRA types entirely.
- The light theme's neutral colors (background/card/border/etc.) were
  Gruvbox Light's actual warm cream (`#f9f5d7`/`#ebdbb2`) despite the
  file's own header comment already claiming "Pure White Board Theme."
  Replaced with genuine white/cool-gray neutrals and a clean blue accent.
  The dark theme's Gruvbox gold/olive pairing was replaced with a dark
  neutral and a single pink accent for the same reason, and is now the
  default a first-time visitor sees.

### Removed
- Two root-level files with zero references anywhere in the repo:
  `templates.ts` (a stale, unimported 471-line duplicate of
  `neurax-ui/src/data/modelTemplates.ts`, the file actually used) and
  `tests/absorption_integration.rs` (never wired to any Cargo package —
  root `Cargo.toml` is workspace-only, so `cargo test --workspace` never
  ran it). A root-level `models/` directory duplicating the documented
  `examples/models/` fixture location, including 8 committed
  `neurax analyze` output files that should never have been tracked, was
  also removed (its two files actually used by `neurax-core`'s test suite,
  `gpt2_medium.json`/`gpt2_small.json`, were kept in place).

## [0.10.0] — 2026-08

### Changed
- **Reduced to 8 fully-supported architecture families** — Transformer, CNN,
  MoE, SSM, Diffusion, GAN, GNN, RNN. Reinforcement Learning, Spiking Neural
  Networks and the "Experimental" catch-all were removed: neither the
  backend `LayerType` enum nor `neurax-formulas` had dedicated coverage for
  any of them, so every block in those families silently cost 0 regardless
  of the design. A family in the picker with no real formula behind it is
  worse than one that isn't offered. 24 reference templates removed with
  them (88 → 64), matching the families that remain.
- Every family kept now has full, audited metric coverage: every
  parameter-bearing block type used across all 64 reference templates
  resolves to a real formula, not the generic zero-cost fallback. Verified
  by compiling all 64 templates and listing every block that still fell
  through — every one left was legitimately parameter-free (dropout,
  residual adds, reshapes, activations, positional-encoding markers), not a
  missed formula.

### Fixed
- **GNN was not actually compiled.** `graph_conv`, `gat_attention` and
  `message_passing` — the layer types every real GNN reference template
  uses — had no dedicated backend `LayerType`, so `graph_conv`/
  `message_passing` fell to the generic zero-cost type and `gat_attention`
  silently costed a plain transformer attention block instead of a real GAT.
  Separately, the graph size (`num_nodes`/`num_edges`) the hardware panel
  already computed was never actually reaching the backend — written onto a
  local object nothing serialized, so no GNN FLOPs calculation ever saw a
  real node/edge count regardless. Fixed on both sides: new `GraphConvNet`/
  `GraphAttentionNet`/`MessagePassing` backend types wired to real formulas
  (`neurax-formulas::gnn`), and the graph size now reaches
  `GlobalResolutionContext` the same way `node_features`/`edge_features`
  already did.
- Diffusion's `mmdit_block` (Stable Diffusion 3 / FLUX's core block),
  `unet_mid`, `unet_latent`, `unet_eff`, `refiner` and `caption_refiner`
  fell to the generic zero-cost type. GAN's `synthesis_block` and `to_rgb`
  (both real convolutions in StyleGAN) did too. All six now route to the
  real formula the equivalent, already-verified block type uses.
- HuggingFace import now reads **CNN** (ResNet, RegNet, ConvNeXt,
  EfficientNet — configs fetched live from the Hub while building this) and
  a **diffusion pipeline's UNet** (Stable Diffusion's `model_index.json` is
  followed to `unet/config.json` automatically — a repo ID alone is enough,
  the same as a language model). GAN, RNN and GNN stay unsupported for
  import: verified live against the Hub that none of the three has a
  standardised `config.json` convention to read.
- `neurax-agent`'s natural-language budget parser missed a real, common way
  clients state a resource ceiling — "runs on 3GB of RAM" / "tourne sur
  3 giga" — because it only recognised "under X" / "moins de X" phrasing.
  The constraint was dropped silently, not flagged as unparsed. Now
  recognised in both languages.

### Added
- Export a complete, runnable PyTorch project — not just the topology.
  `model.py` (real `nn.Module`s generated from the same formulas behind the
  numbers already on screen, verified against the analysis before being
  offered), `train.py` wired to the hyperparameters already set, a
  generated README, and `requirements.txt` — as a downloadable `.zip` or
  pushed straight to GitHub. Every generated file is watermarked with
  whether it was verified or not; nothing is shipped silently unverified.
- The GitHub export target creates the destination repository if it
  doesn't exist yet (private by default), and seeds the first commit by
  hand via the Git Data API when a repository exists but has never had one
  — both used to surface as an opaque 404 partway through the export.
- Gruvbox — light and dark — is now NEURAX's own theme, replacing a generic
  indigo default with the palette the spider mark already used. Four more
  palettes (Molten Core, Signal & Static, Web & Amber, Slate & Ember), each
  with full semantic coverage (destructive/warning/success/info and all 8
  chart colors), not just a background and a primary color.
- Chart lines and areas mark their current point with a real emphasized
  dot on hover, instead of recharts' unstyled default. Time Machine's
  charts (cost projection, carbon footprint, cost breakdown) now follow the
  active theme's semantic colors instead of fixed hues that ignored it.

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
