# neurax-ir

**10 analytical IR dialects for NEURAX.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. A 10-pass intermediate representation that transforms a neural architecture design into engineering metrics — FLOPs, VRAM, latency, cost, energy, carbon — in under 50 ms, fully deterministically, with zero GPU.

## The 10 passes

| Pass | Computed metrics |
|------|------------------|
| Architecture | Layer count, model type, global parameters |
| Graph | Topology validation, DAG structure, fan-in/fan-out |
| Tensor | Shape inference, dimension resolution, memory layout |
| Operator | FLOPs per op, parameter count, operation types |
| Compute | Total FLOPs, throughput, backward/optimizer overhead |
| Memory | Peak VRAM, activation/gradient memory, fragmentation |
| Parallelism | Tensor/pipeline/expert parallelism, efficiency scores |
| Hardware | GPU utilization, bandwidth, ridge point, latency |
| Cost | Training cost (USD), time (hours), energy (kWh), CO2 (kg) |
| Report | Consolidated metrics, diagnostics, recommendations |

## Usage

```rust
use neurax_ir::pipeline::compile;

let report = compile(model_json)?;
println!("{}", report.cost_usd);
```

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.