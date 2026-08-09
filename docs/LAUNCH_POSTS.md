# NEURAX — Launch Posts (ready to copy-paste)

Generated 2026-08-09. Post on the same day for GitHub Trending (stars in a short window).

---

## 1. Hacker News — Show HN (Tuesday–Thursday, 9:00–10:00 ET)

**Title:**
```
Show HN: NEURAX – Analytical compiler that predicts ML training cost in <50ms, no GPU
```

**Body:**
```
I built a compiler that answers the questions you need before committing GPU budget:
- Will this architecture fit in VRAM?
- What does training cost on 8x H100?
- Where are the memory bottlenecks?

NEURAX is an analytical compiler for neural architectures. It runs a 10-pass IR
(Architecture → Graph → Tensor → Operator → Compute → Memory → Parallelism →
Hardware → Cost → Report) over a JSON model description and produces 55+ metrics:
FLOPs, VRAM, latency, USD cost, kWh, CO2. Deterministically, in under 50 ms,
on a CPU, with zero GPU.

It ships with 88 reference templates (GPT-4, LLaMA-2, Mixtral, DeepSeek-V3,
Mamba, SDXL, ResNet...) across 11 architecture families, and a custom MLIR
backend (10 dialects, LLVM 18) for lowering.

Why analytical instead of measured? Because you don't have the GPU yet. You're
deciding whether to buy it. NEURAX gives you the numbers before you commit.

All 9 crates are on crates.io:
cargo install neurax-cli
neurax analyze examples/models/gpt3_175b.json

Benchmarks (11 models, <10ms each): https://github.com/rustnew/NEURAX/blob/main/BENCHMARKS.md

The 10-pass IR design: https://github.com/rustnew/NEURAX
Live demo: https://rustnew.github.io/NEURAX/

Happy to discuss the analytical formulas, the IR design, or the MLIR lowering.
```

---

## 2. Reddit r/rust — "What are you working on?" weekly thread

```
I've been building NEURAX, an analytical compiler for neural network
architectures, in Rust. It answers "will this model fit in VRAM / what does
training cost" before you rent the GPU — via a 10-pass IR, no training, no GPU,
deterministic, <50ms.

The whole engine is Rust: parser → IR (10 dialects) → formulas → report.
There's also a custom MLIR backend (10 dialects on LLVM 18) and a Ratatui TUI.

Just published all 9 crates to crates.io:
- neurax-core (pipeline orchestration)
- neurax-ir (10-pass analytical IR)
- neurax-parser / neurax-formulas / neurax-hardware-db
- neurax-mlir (MLIR dialects, LLVM 18)
- neurax-cli / neurax-tui / neurax-service

cargo install neurax-cli

Looking for feedback on the IR design and the analytical formulas.
https://github.com/rustnew/NEURAX
```

---

## 3. Reddit r/MachineLearning — [P] research post

**Title:**
```
[P] NEURAX: analytical compiler predicts training FLOPs/VRAM/cost before training — 11 models benchmarked in <10ms each
```

**Body:**
```
We built an analytical compiler that predicts FLOPs, peak VRAM, latency, cost,
energy and CO2 for a neural architecture from its JSON description alone —
before any training run, with zero GPU, fully deterministically.

It runs a 10-pass IR (Architecture, Graph, Tensor, Operator, Compute, Memory,
Parallelism, Hardware, Cost, Report) and computes 55+ metrics using closed-form
analytical formulas per architecture family (Transformer, MoE, CNN, SSM, RNN,
Diffusion, GNN, GAN, RL, SNN).

Benchmarks on 11 reference architectures (GPT-4, LLaMA-2-70B, DeepSeek-V3,
Mixtral-8x7B, Mamba-2.8B, SDXL...): every analysis completes in <10ms on CPU.
Full table: https://github.com/rustnew/NEURAX/blob/main/BENCHMARKS.md

The interesting part is the analytical formulas themselves — e.g. exact
activation memory for gated MLPs, optimizer-state accounting per precision
(bf16/fp8), and parallelism efficiency models. Happy to discuss the math.

Rust implementation, 9 crates on crates.io, MLIR backend (LLVM 18):
https://github.com/rustnew/NEURAX

We're actively looking for validation partners to compare predictions against
real training runs (the benchmark table currently shows predictions only).
```

---

## 4. Reddit r/coolgithubprojects

**Title:**
```
[Project] NEURAX — predict ML training cost in <50ms, no GPU required (Rust + MLIR)
```

**Body:**
```
NEURAX is an analytical compiler for neural architectures. Give it a model JSON
(GPT-4, LLaMA-2, Mixtral, SDXL...), it returns FLOPs, VRAM, cost, energy, CO2 —
in under 50ms, on a CPU, no GPU needed.

- 10-pass analytical IR, 55+ metrics
- 88 reference templates, 11 architecture families
- Custom MLIR backend (10 dialects, LLVM 18)
- 9 crates on crates.io: cargo install neurax-cli

Demo: https://rustnew.github.io/NEURAX/
Repo: https://github.com/rustnew/NEURAX
```

---

## 5. CSDN (Chinese) — 技术文章标题

**标题:**
```
NEURAX：50ms 内预测神经网络训练成本的解析编译器（Rust + MLIR，已开源）
```

**正文要点:**
```
- 问题：训练大模型前，如何知道显存够不够？8×H100 训练成本多少？
- 方案：NEURAX 解析编译器，10 道 IR 流水线（架构→图→张量→算子→计算→
  显存→并行→硬件→成本→报告），55+ 指标，<50ms，无需 GPU，完全确定性。
- 11 个参考架构基准（GPT-4、LLaMA-2-70B、DeepSeek-V3、Mixtral、SDXL 等），
  每个分析 <10ms。
- 技术栈：Rust 9 个 crate（crates.io 已发布）+ 自定义 MLIR 后端（LLVM 18）。
- 安装：cargo install neurax-cli
- 仓库：https://github.com/rustnew/NEURAX
- 中文文档与讨论欢迎在评论区交流。
```

---

## 6. Dev.to — article

**Title:**
```
NEURAX: I built a compiler that predicts ML training costs before you rent the GPU
```

**Tags:** `#rust` `#machinelearning` `#compiler` `#showdev`

**Body:** (use the HN body, expanded with the 10-pass IR table and benchmark table)

---

## 7. X/Twitter — launch thread (7 tweets)

```
1/ 🧠 NEURAX: an analytical compiler for neural architectures.
   Predict FLOPs, VRAM, cost, energy, CO2 BEFORE training.
   No GPU. <50ms. Deterministic. Open source.

2/ The problem: you're deciding whether to rent 8x H100.
   PyTorch can't tell you the cost — you have to train first.
   NEURAX computes it analytically from the architecture JSON.

3/ 10-pass IR pipeline:
   Architecture → Graph → Tensor → Operator → Compute →
   Memory → Parallelism → Hardware → Cost → Report
   55+ metrics, fully deterministic.

4/ 88 reference templates: GPT-4, LLaMA-2, Mixtral, DeepSeek-V3,
   Mamba, SDXL, ResNet... 11 architecture families.

5/ Benchmarked 11 models in <10ms each on CPU:
   github.com/rustnew/NEURAX/blob/main/BENCHMARKS.md

6/ Custom MLIR backend: 10 dialects on LLVM 18.
   All 9 crates on crates.io:
   cargo install neurax-cli

7/ Live demo: rustnew.github.io/NEURAX
   Repo: github.com/rustnew/NEURAX
   Built with Rust + MLIR + React. #buildinpublic #rustlang #mlir
```