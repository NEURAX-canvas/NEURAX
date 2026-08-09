# neurax-formulas

**Analytical formulas for ML operations — the hot path of NEURAX.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. This crate contains the pure, deterministic formulas that compute FLOPs, memory, latency, cost, energy and carbon for every neural network operation — without a GPU.

## Features

- Per-architecture analytical formulas (Transformer, CNN, MoE, SSM, Diffusion, GNN, GAN, RL, SNN, RNN, Experimental)
- Deterministic: identical input → identical output
- Zero GPU required, sub-millisecond evaluation
- Expression-based evaluation via `evalexpr`

## Usage

```rust
use neurax_formulas::compute_flops;

let flops = compute_flops("attention", &params)?;
```

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.