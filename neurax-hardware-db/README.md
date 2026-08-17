# neurax-hardware-db

**Hardware specification database for NEURAX — GPU/CPU/Interconnect specs with roofline analysis.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural network architectures. This crate ships a built-in database of GPU, CPU and interconnect specifications and provides the analytical primitives needed to answer hardware questions at **design time** — *"will this fit in VRAM?", "is this compute-bound or memory-bound?", "how long will this take on an H100?"*

> **The vision:** hardware decisions should be made with the same rigor as software decisions. This crate turns "which GPU should I rent?" into a queryable, deterministic question.

## What's inside

- **20+ GPUs**: A100 (SXM/PCIe), H100 (SXM/PCIe), V100, RTX 3080/3090/4080/4090, RTX A5000/A6000, RTX 6000 Ada, and more
- **CPUs**: server and workstation models
- **Interconnects**: NVLink, PCIe, InfiniBand specs
- **Roofline analysis**: `ridge_point`, `efficiency_factor`, compute-vs-memory-bound detection
- **Precision-aware**: `tflops_for_precision("fp16" | "fp32" | "bf16" | "fp8" | ...)`
- **Budget checks**: `fits_in_memory()`, `compute_time_ms()`, `memory_time_ms()`

## Installation

```toml
[dependencies]
neurax-hardware-db = { git = "https://github.com/rustnew/NEURAX", package = "neurax-hardware-db" }
```

## Quick start

```rust
use neurax_hardware_db::{HardwareDatabase, GpuSpec};

// Built-in database (20+ GPUs, CPUs, interconnects)
let db = HardwareDatabase::new();

// Look up an H100 and run a roofline analysis
let h100 = db.get_gpu("H100-SXM").expect("H100 in database");
println!("H100: {} GB VRAM, {:.0} TFLOPS fp16, {} W TDP",
    h100.memory_gb, h100.tflops_fp16, h100.tdp_watts);

// Precision-aware throughput
let tflops = h100.tflops_for_precision("fp8");
println!("H100 fp8 throughput: {:.0} TFLOPS", tflops);

// Roofline: where is the ridge point?
let ridge = h100.ridge_point("fp16"); // FLOPs per byte
println!("Ridge point (fp16): {:.1} FLOPs/byte — above it → compute-bound", ridge);

// Can a 70B model in bf16 (140 GB weights) fit on one H100 (80 GB)?
let fits = h100.fits_in_memory(140 * 1024 * 1024 * 1024);
println!("70B bf16 weights on one H100? {}", fits); // false — you need multi-GPU

// Time estimates for a given workload
let flops = 3.38e16; // LLaMA-2-70B forward+backward, see neurax-formulas
println!("Estimated: {:.0} ms compute", h100.compute_time_ms(flops, "fp16"));
```

## Working with the full pipeline

`neurax-hardware-db` feeds the **Hardware pass** of the [NEURAX 10-pass IR](https://github.com/rustnew/NEURAX). You rarely need it directly — the full pipeline handles it:

```rust
use neurax_core::analyze_json;

let result = analyze_json(model_json)?;
println!("GPU utilization: {:.2}", result.hardware.utilization);
println!("Peak VRAM:       {:.0} GB", result.memory.peak_vram_gb);
```

## API overview

### `HardwareDatabase`
| Method | Description |
|--------|-------------|
| `new()` | Database with built-in specs |
| `get_gpu(name) -> Option<&GpuSpec>` | Look up a GPU by name |
| `get_gpu_or_fallback(name) -> GpuSpec` | Fallback to generic spec |
| `get_cpu(name) -> Option<&CpuSpec>` | Look up a CPU |
| `get_interconnect(name) -> Option<&InterconnectSpec>` | Look up an interconnect |
| `list_gpus() -> Vec<&GpuSpec>` | All GPUs in the database |

### `GpuSpec` (roofline methods)
| Method | Description |
|--------|-------------|
| `tflops_for_precision(precision)` | Throughput for fp32/fp16/bf16/int8/fp8 |
| `efficiency_factor()` | Achievable fraction of peak (memory-bound discount) |
| `effective_tflops(precision)` | Realistic throughput after efficiency |
| `ridge_point(precision)` | FLOPs/byte — the compute-vs-memory boundary |
| `compute_time_ms(flops, precision)` | Estimated compute time |
| `memory_time_ms(bytes)` | Estimated memory time |
| `fits_in_memory(bytes)` | VRAM budget check |

## Determinism guarantee

Pure lookups + closed-form math. Same query → same answer. No I/O, no randomness.

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.