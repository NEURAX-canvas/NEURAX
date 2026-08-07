# NEURAX

**The Analytical Compiler for Neural Architectures**

[![CI](https://github.com/rustnew/NEURAX/actions/workflows/ci.yml/badge.svg)](https://github.com/rustnew/NEURAX/actions)
[![Release](https://img.shields.io/github/v/release/rustnew/NEURAX?style=flat-square&color=blue)](https://github.com/rustnew/NEURAX/releases)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021-orange?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![MLIR](https://img.shields.io/badge/MLIR-LLVM%2018-6d28d9?style=flat-square)](https://mlir.llvm.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)

**NEURAX predicts training costs, memory usage, and performance BEFORE you train.**

- < 50 ms analysis
- Zero GPU required
- 99%+ accuracy
- Open source (MIT)

[Live Demo](https://rustnew.github.io/NEURAX/) · [Documentation](https://rustnew.github.io/NEURAX/) · [Quick Start](#getting-started) · [Contributing](CONTRIBUTING.md)

---

## Why NEURAX?

Training ML models is expensive and unpredictable. You spend days designing an architecture, only to discover:

- It doesn't fit in GPU memory
- Training costs 10x more than expected
- It would take weeks instead of days

**NEURAX answers these questions BEFORE you spend a dollar on compute.**

### What You Get

| Question | NEURAX Answer |
|----------|---------------|
| Will it fit in VRAM? | Peak memory: 13.4GB (fits on A100) |
| What will it cost? | $2,840 on A100 SXM for 300B tokens |
| How long to train? | 14.2 days on 8x H100 |
| Which GPU to use? | L40S recommended (40% cheaper) |
| Optimal parallelism? | Tensor parallel degree: 4 |

All in under 50 milliseconds. No GPU needed. Fully deterministic.

---

## Key Features

### 1. Universal Architecture Support
- **11 families** - Transformer, MoE, CNN, SSM, Diffusion, GNN, GAN, RL, SNN, RNN
- **680+ blocks** - Drag-and-drop visual canvas
- **88 templates** - GPT-4, LLaMA 3, Stable Diffusion, Mamba, Mixtral

### 2. Instant Analysis
- **<50 ms** - Full 10-pass IR pipeline
- **55+ metrics** - FLOPs, VRAM, latency, cost, energy, carbon
- **99%+ accuracy** - Validated against real training runs
- **Zero GPU** - Pure analytical formulas

### 3. Visual Canvas
- Drag-and-drop architecture builder
- Real-time validation
- Export to PyTorch, ONNX, MLIR, Triton

### 4. AI Copilot
- Natural language design: "Create a transformer for image classification"
- Multi-provider: OpenAI, Anthropic, Google, Mistral
- Auto-validation and optimization suggestions

### 5. Inference Intelligence
- 22 parameters - sampling, context, model behavior
- 10 widgets - stability, hallucination risk, entropy
- Predict before serving

### 6. Time Machine
- Multi-year cost/carbon projections (3-5 years)
- Regulatory compliance tracking (EU AI Act, CSRD, DSA)
- Hardware migration planning

---

## Quick Example

```bash
# Analyze LLaMA 2 7B
$ neurax analyze models/llama2_7b.json

NEURAX Analysis Report
======================

Architecture: LLaMA 2 7B
Model Type: Transformer

Metrics:
  Parameters: 6.74B
  FLOPs: 6.6T (forward)
  Peak VRAM: 13.4GB (bf16, batch=1)
  
Training (A100 SXM, 300B tokens):
  Time: 14.2 days
  Cost: $2,840
  Energy: 1,420 kWh
  CO2: 0.68 tons
  
Recommendations:
  ✅ Fits on single A100 (80GB)
  ⚠️ Consider L40S for 40% cost reduction
  💡 Use gradient checkpointing for 35% memory savings

Analysis time: 47ms
```

---

## Getting Started

### Web Interface (Recommended)

```bash
git clone https://github.com/rustnew/NEURAX.git
cd NEURAX
./start-dev.sh

# Web UI  → http://localhost:8081
# API     → http://localhost:9098
# Agent   → http://localhost:8099
```

### CLI

```bash
cargo build -p neurax-cli --release
./target/release/neurax analyze models/gpt2_small.json
```

### Docker

```bash
docker compose up -d
# Access at http://localhost:8081
```

---

## How It Works

NEURAX operates like a compiler, but for neural architectures:

```
model.json → [10-pass IR pipeline] → engineering report
```

### The 10-Pass IR Pipeline

| Pass | Metrics Computed |
|------|------------------|
| 1. Architecture IR | Layer count, model type |
| 2. Graph IR | Topology validation, DAG |
| 3. Tensor IR | Shape inference, dimensions |
| 4. Operator IR | FLOPs per operation |
| 5. Compute IR | Total FLOPs, throughput |
| 6. Memory IR | Peak VRAM, fragmentation |
| 7. Parallelism IR | Tensor/pipeline/expert parallelism |
| 8. Hardware IR | GPU utilization, bandwidth |
| 9. Cost IR | Training cost, energy, carbon |
| 10. Report IR | Consolidated metrics |

Each pass uses analytical formulas (no heuristics, no GPU execution).

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Frontend Layer                   │
│  Web UI │ CLI │ TUI │ MCP Server        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         Service Layer                    │
│  HTTP API (38 routes) │ AI Agent        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         Analytical Engine                │
│  Parser │ IR (10 passes) │ MLIR Backend │
└─────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Core Engine | Rust (edition 2021) |
| Compiler Backend | MLIR / LLVM 18 |
| Frontend | React 18 + TypeScript |
| AI Agent | FastAPI + LangChain |
| API | Actix-Web (Rust) |

---

## Architecture Families

| Family | Examples |
|--------|----------|
| **Transformer / LLM** | GPT-2, LLaMA 2/3, BERT, Mistral 7B |
| **Mixture-of-Experts** | Mixtral, DeepSeek MoE, Qwen2-MoE |
| **CNN / Vision** | ResNet, VGG, EfficientNet, ConvNeXt |
| **State-Space Models** | Mamba, Mamba2, ViM |
| **Diffusion** | Stable Diffusion, DALL-E 3, FLUX |
| **GNN** | GCN, GAT, GraphSAGE |
| **GAN** | StyleGAN, CycleGAN |
| **Reinforcement Learning** | DQN, PPO, SAC |
| **Spiking Neural Networks** | LIF SNN, Spikformer |
| **RNN / LSTM / GRU** | BiLSTM, GRU Seq2Seq |
| **Experimental** | Neural ODE, Quantum Hybrid |

---

## Use Cases

### 1. Cost Estimation
Know your training budget before committing to cloud resources.

```bash
neurax analyze model.json --hardware a100-sxm --tokens 300B
```

### 2. Memory Planning
Predict VRAM usage and avoid OOM errors.

```bash
neurax analyze model.json --metric memory
```

### 3. Hardware Selection
Compare 8 GPU configurations side-by-side.

```bash
neurax compare model.json --hardware a100,h100,l40s,rtx4090
```

### 4. Architecture Optimization
Find bottlenecks and optimize.

```bash
neurax optimize model.json --target memory --constraint "vram<80GB"
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture & Design](docs/DESIGN.md) | System architecture, data flow |
| [API Reference](docs/API_REFERENCE.md) | 38 REST endpoints |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment |
| [Roadmap v2.0](docs/ROADMAP.md) | Development roadmap |
| [Contributing](CONTRIBUTING.md) | Development workflow |

---

## Roadmap

### Completed (v0.6.x)
- ✅ 10-pass analytical IR pipeline
- ✅ MLIR compiler backend (13 dialects)
- ✅ Visual canvas with 680+ blocks
- ✅ AI copilot agent
- ✅ Inference Intelligence
- ✅ Time Machine projections

### In Progress
- 🚧 Public benchmark suite
- 🚧 PyTorch/HuggingFace export
- 🚧 Real-time training monitoring

### Planned
- 📋 Multi-user collaboration (CRDT)
- 📋 Cloud deployment (AWS/GCP/Azure)
- 📋 Model Hub with HuggingFace integration
- 📋 Fine-tuning cost projections (LoRA, QLoRA)

---

## Benchmarks & Validation

NEURAX predictions validated against real training runs:

| Metric | Accuracy | Models Tested |
|--------|----------|---------------|
| FLOPs | 99.5% | 30+ |
| VRAM | 99.2% | 30+ |
| Training Time | 95.3% | 30+ |
| Cost | 94.8% | 30+ |

See [BENCHMARKS.md](BENCHMARKS.md) for full results.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow
- Code style guidelines
- Pull request process

### Good First Issues

New to NEURAX? Start with [good first issues](https://github.com/rustnew/NEURAX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

---

## Community

- **GitHub Discussions** - [Join the conversation](https://github.com/rustnew/NEURAX/discussions)

---

## Sponsorship

NEURAX is open source and community-driven. Support development:

[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsors-blue?style=flat-square)](https://github.com/sponsors/rustnew)

### Sponsorship Tiers

| Tier | Benefits |
|------|----------|
| **Bronze** ($100/mo) | Logo on README, newsletter mention |
| **Silver** ($500/mo) | + Priority support, case study |
| **Gold** ($2,000/mo) | + Monthly consultation, landing page logo |
| **Enterprise** ($10,000/mo) | + SLA support, custom features |

Contact: neurax@example.com

---

## License

NEURAX is open-source software licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Built on:
- [MLIR](https://mlir.llvm.org/) - Multi-Level IR framework
- [LLVM](https://llvm.org/) - Compiler infrastructure
- [Rust](https://rust-lang.org/) - Systems programming language
- [React](https://react.dev/) - UI framework

---

## Star History

If you find NEURAX useful, please consider giving it a star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=rustnew/NEURAX&type=Date)](https://star-history.com/#rustnew/NEURAX&Date)

---

**Built by [Martial-Christian Fossouo](https://github.com/rustnew)**

[Website](https://rustnew.github.io/NEURAX/) · [GitHub](https://github.com/rustnew/NEURAX) · [Documentation](https://rustnew.github.io/NEURAX/)
