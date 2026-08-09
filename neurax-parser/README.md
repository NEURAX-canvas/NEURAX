# neurax-parser

**JSON parsing for the NEURAX universal format.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. Parses the NEURAX JSON model format into a strongly-typed AST, with expression evaluation for parameterized blocks.

## Features

- Strongly-typed AST from NEURAX JSON
- 680+ configurable blocks, 88 reference templates
- Expression evaluation for dynamic parameters
- Serde-based, deterministic

## Usage

```rust
use neurax_parser::parse_model;

let model = parse_model(json_str)?;
```

## License

MIT — see the [NEURAX repository](https://github.com/rustnew/NEURAX) for details.