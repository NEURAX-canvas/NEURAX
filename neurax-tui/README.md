# neurax-tui

**Terminal User Interface for NEURAX IR model compilation visualization.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. A Ratatui-based terminal dashboard to visualize the 10-pass IR pipeline and its metrics in real time.

## Install

```bash
cargo install neurax-tui
```

## Usage

```bash
neurax-tui
```

## Features

- Live visualization of the 10-pass analytical IR pipeline
- Metric dashboards (FLOPs, VRAM, cost, energy, carbon)
- Built with [ratatui](https://crates.io/crates/ratatui) + crossterm

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.