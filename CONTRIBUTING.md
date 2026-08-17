# Contributing to NEURAX

First off, thanks for taking the time to contribute! 🧠

NEURAX is the Analytical Compiler for Neural Network Architectures — it predicts
cost, memory, and performance of AI models **before training**, at design time.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [The Web UI](#the-web-ui)
- [Adding a Model Family](#adding-a-model-family)
- [Commit Conventions](#commit-conventions)
- [Publishing](#publishing)

## Code of Conduct

Be respectful and assume good faith. Disagree about code, not about people.

## Project Layout

NEURAX is a Cargo workspace plus a React web UI and a Python agent:

| Directory | Description |
|-----------|-------------|
| `neurax-core` | Core IR, analysis passes, metrics engine |
| `neurax-parser` | Model config parsing (JSON, family detection) |
| `neurax-ir` | Intermediate representation types |
| `neurax-formulas` | Analytical cost/memory/performance formulas |
| `neurax-hardware-db` | Hardware database (GPUs, CPUs, interconnects) |
| `neurax-mlir` | MLIR / LLVM 18 compiler backend (14 dialects) |
| `neurax-service` | Streaming SSE analysis API |
| `neurax-mcp` | MCP (Model Context Protocol) integration |
| `neurax-tui` | Terminal UI (ratatui) |
| `neurax-ui` | React 18 + TypeScript visual design canvas |
| `neurax-agent` | Python agent for architecture planning (LangChain) |
| `neurax-desktop` | Tauri desktop application — the studio, offline |

## Prerequisites

- **Rust** (stable, edition 2021)
- **LLVM 18 / MLIR** — required by `neurax-mlir`:

  ```bash
  sudo apt install llvm-18 llvm-18-dev libmlir-18-dev mlir-18-tools
  export LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
  export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
  ```

  A `.cargo/config.toml` sets these automatically for the workspace.

- **Node.js 20+** (for `neurax-ui`), **Python 3.10+** (for `neurax-agent`).

## Development Workflow

1. **Fork** the repository and create your branch from `main`:

   ```bash
   git checkout -b feat/my-change
   ```

2. **Make your changes** with tests.

3. **Run checks locally** (see below).

4. **Open a Pull Request** with a clear description of what and why.

## Running Tests

```bash
# Rust workspace (all crates)
cargo build --workspace
cargo test --workspace

# Just the Rust crates
cargo test -p neurax-core
cargo test -p neurax-parser

# Lint
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

## The Web UI

```bash
cd neurax-ui
bun install        # or npm install
bun run dev        # Vite dev server
```

## Adding a Model Family

1. Add the new `ModelType` variant in `neurax-parser/src/model_config.rs`
   (both `from_str` and `as_str`).
2. Update the family detection in `neurax-parser`.
3. Add a planning template in `neurax-agent/arch_planner.py`.
4. Add a reference template under `examples/models/`.
5. Extend `neurax-parser/tests/universal_compiler_test.rs` with the new family.

## Commit Conventions

Use conventional commit prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `chore:` — maintenance (bumps, metadata, tooling)
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests

Example: `feat(neurax): add Multimodal (VLM) support + build tooling`

## Publishing

Releases are versioned with SemVer and tagged `vX.Y.Z`. See the
[Deployment Guide](https://rustnew.github.io/NEURAX/DEPLOYMENT.html) (or
[`book/src/DEPLOYMENT.md`](book/src/DEPLOYMENT.md) in a checkout) for the
full release and deployment process.
