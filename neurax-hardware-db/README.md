# neurax-hardware-db

**Hardware database for NEURAX — GPU/CPU specifications.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. Ships with specs for 20+ GPUs and CPUs (VRAM, bandwidth, FLOPs, TDP) used to predict training cost, memory and performance before training.

## Features

- GPU/CPU spec database (NVIDIA, AMD, Apple Silicon, CPUs)
- Serde-serializable, easy to extend
- Used by `neurax-ir` for the Hardware pass

## Usage

```rust
use neurax_hardware_db::HardwareDb;

let db = HardwareDb::default();
let gpu = db.gpu("H100")?;
println!("{} GB VRAM", gpu.vram_gb);
```

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.