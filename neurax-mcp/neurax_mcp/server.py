"""NEURAX MCP Server - Model Context Protocol server for NEURAX neural architecture compiler."""

import asyncio
import json
import logging
import os
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurax-mcp")

# Default NEURAX service endpoint (can be overridden via env)
NEURAX_SERVICE_URL = os.environ.get("NEURAX_SERVICE_URL", "http://127.0.0.1:9098")

# Create MCP server
app = Server("neurax-mcp")


async def _call_backend(path: str, method: str = "GET", data: dict | None = None) -> dict | list | str:
    """Helper to call NEURAX backend API."""
    url = f"{NEURAX_SERVICE_URL}{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        if method == "GET":
            response = await client.get(url)
        elif method == "POST":
            response = await client.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        response.raise_for_status()
        return response.json()


def _format_metrics(results: dict) -> str:
    """Format analysis results into a readable summary."""
    metrics = results.get("metrics", results.get("report", {}).get("metrics", {}))
    if not metrics:
        return json.dumps(results, indent=2)

    lines = ["NEURAX Analysis Results:"]
    lines.append(f"- Total Parameters: {metrics.get('total_parameters', 0):,}")

    # Model size is the constraint most on-device work is actually held to
    # ("must fit in 1 MB"), and it was the one figure this summary omitted —
    # callers were left to infer it from the parameter count and the precision.
    lines.append(f"- Model Size: {format_size(metrics.get('parameter_memory_bytes', 0))}")

    lines.append(f"- Total FLOPs: {metrics.get('total_flops', 0):.2e}")

    peak_vram = metrics.get("peak_vram_bytes", 0)
    lines.append(f"- Peak VRAM: {format_size(peak_vram)}")

    latency = metrics.get("latency_ms", 0)
    if latency:
        lines.append(f"- Estimated Latency: {latency:.2f} ms")

    throughput = metrics.get("throughput_tokens_per_s", 0)
    if throughput:
        lines.append(f"- Throughput: {throughput:.2f} tok/s")

    # Training figures are meaningless for an inference-only design and come
    # back as zeros; reporting them anyway made every on-device analysis look
    # like it had a broken cost model.
    training = [
        ("training_cost_usd", "Training Cost", "${:,.2f}"),
        ("training_time_hours", "Training Time", "{:,.2f} hours"),
        ("energy_kwh", "Energy", "{:,.2f} kWh"),
        ("co2_kg", "CO2", "{:,.2f} kg"),
    ]
    reported = [(label, fmt.format(metrics[key]))
                for key, label, fmt in training
                if metrics.get(key, 0)]
    if reported:
        for label, value in reported:
            lines.append(f"- {label}: {value}")
    else:
        lines.append("- Training cost: not applicable (no training budget given)")

    return "\n".join(lines)


def format_size(num_bytes: float) -> str:
    """Human-readable size, with the byte count kept for exact comparisons."""
    if not num_bytes:
        return "0 B"
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {unit} ({num_bytes:,.0f} bytes)"
    return f"{num_bytes:,.0f} bytes"


def _format_budget_report(results: dict, budget: dict) -> str:
    """Check measured metrics against the caller's deployment budget.

    Each constraint is reported with the measured value and the headroom, so a
    design can be iterated toward a target rather than merely described.
    """
    metrics = results.get("metrics", results.get("report", {}).get("metrics", {}))
    if not metrics:
        return "No metrics returned; cannot check the budget."

    size_bytes = metrics.get("parameter_memory_bytes", 0)
    checks = [
        ("Model size", budget.get("max_size_mb"),
         size_bytes / (1024 ** 2), "MB", "{:.3f}"),
        ("Peak VRAM", budget.get("max_vram_gb"),
         metrics.get("peak_vram_bytes", 0) / (1024 ** 3), "GB", "{:.3f}"),
        ("Latency", budget.get("max_latency_ms"),
         metrics.get("latency_ms", 0), "ms", "{:.2f}"),
        ("Parameters", budget.get("max_parameters"),
         metrics.get("total_parameters", 0), "params", "{:,.0f}"),
    ]

    stated = [c for c in checks if c[1] is not None]
    if not stated:
        return ("No budget constraints given. Measured: "
                + _format_metrics(results))

    lines = ["NEURAX Budget Check:", ""]
    all_pass = True
    for label, limit, measured, unit, fmt in stated:
        fits = measured <= limit
        all_pass = all_pass and fits
        margin = limit - measured
        lines.append(
            f"[{'PASS' if fits else 'FAIL'}] {label}: "
            f"{fmt.format(measured)} {unit} against a limit of {fmt.format(limit)} {unit} "
            f"({'headroom' if fits else 'over by'} {fmt.format(abs(margin))} {unit})"
        )

    lines.append("")
    lines.append("VERDICT: the design fits every stated budget."
                 if all_pass else
                 "VERDICT: the design exceeds at least one budget and needs to be reduced.")
    if not all_pass:
        lines.append(
            "Levers, in rough order of effect: reduce hidden size (parameters scale "
            "with its square in the projections), cut layer count, shrink the "
            "vocabulary or embedding dimension, or move to a narrower dtype."
        )
    return "\n".join(lines)


# Map canvas/agent layer types to the compiler's LayerType vocabulary. Kept
# in sync with neurax-agent/budget_check.py's LAYER_TYPE_MAP by hand (this
# package installs standalone, with no dependency on neurax-agent) — this
# one had drifted to a fraction of that table's size (no MoE/GNN/SSM/LoRA
# entries at all), so any of those block types sent through this server
# reached the compiler as their own unrecognized name and failed to parse
# outright.
LAYER_TYPE_MAP = {
    "token_embedding": "embedding",
    "embedding": "embedding",
    "pos_embed": "positional_embed",
    "positional_encoding": "positional_embed",
    "rope": "positional_embed",
    "alibi": "positional_embed",
    "mha_attention": "attention",
    "mha": "attention",
    "attention": "attention",
    "gqa": "attention",
    "flash_attention": "attention",
    "self_attention": "attention",
    "cross_attention": "cross_attention",
    "bahdanau_attention": "attention",
    "ffn_standard": "mlp",
    "ffn_gated": "mlp",
    "ffn": "mlp",
    "mlp": "mlp",
    "swiglu": "mlp",
    "lm_head": "dense",
    "classification_head": "dense",
    "linear": "dense",
    "linear_projection": "dense",
    "dense": "dense",
    "output": "dense",
    "input": "embedding",
    "layer_norm": "normalization",
    "layernorm": "normalization",
    "rmsnorm": "normalization",
    "batchnorm": "normalization",
    "groupnorm": "normalization",
    "instancenorm": "normalization",
    "conv2d": "conv",
    "conv1d": "conv",
    "depthwise_conv2d": "conv",
    "conv_transpose2d": "conv",
    "max_pool": "pooling",
    "avg_pool": "pooling",
    "global_pool": "pooling",
    "moe_block": "moe",
    "expert": "moe",
    "mamba_block": "mamba_block",
    "s4_block": "s4_block",
    "lstm": "lstm",
    "gru": "gru",
    "gcn_conv": "graph_conv",
    "gat_conv": "graph_attention",
    "message_passing": "message_passing",
    "rgcn_conv": "rgcn_conv",
    "lora_linear": "lora_linear",
    "dora_linear": "dora_linear",
}


def _build_topology(arch: dict) -> dict:
    """Convert an MCP `architecture` argument (nodes/connections/hardware) into
    the wire-format topology NEURAX's /analyze and /sweep endpoints expect.
    Shared by analyze_architecture, check_budget, and find_optimal_hyperparameters
    so the layer-type mapping and schema quirks below live in exactly one place.
    """
    raw_nodes = arch.get("nodes", [])
    raw_connections = arch.get("connections", [])
    raw_hardware = arch.get("hardware", {})

    layers = []
    for n in raw_nodes:
        frontend_type = n.get("type", "unknown")
        backend_type = LAYER_TYPE_MAP.get(frontend_type, frontend_type)
        layer = {
            "id": n.get("id", f"layer_{len(layers)}"),
            "layer_type": backend_type,
            "params": n.get("params", {}),
        }
        if "input_shape" in n:
            layer["input_shape"] = n["input_shape"]
        if "output_shape" in n:
            layer["output_shape"] = n["output_shape"]
        layers.append(layer)

    gpu_name = raw_hardware.get("gpu_name", "H100-SXM")
    gpu_count = raw_hardware.get("gpu_count", 1)
    precision = raw_hardware.get("precision", "bf16")

    topology = {
        "schema_version": "1.0",
        "model": {
            "name": arch.get("name", "MCP Model"),
            "type": arch.get("family", "transformer"),
            "layers": layers,
        },
        "training": {
            "batch_size": raw_hardware.get("batch_size", 1),
            # The backend reads the storage width from `training.precision`.
            # Sending it only as `data.dtype` left every analysis on the
            # fp32 default, overstating model size fourfold for an int8
            # design — decisive against an on-device size budget.
            "precision": precision,
        },
        "hardware": {
            "gpus": [{"name": gpu_name, "count": gpu_count}],
        },
        "data": {
            "input_shape": arch.get("input_shape", [1, 128]),
            "dtype": precision,
        },
    }

    # Image-shaped families (cnn/vit/gan/diffusion) need their entry
    # shape stated here — the compiler's shape-inference engine reads
    # it from exactly these three fields, defaulting to a 224x224x3
    # placeholder image otherwise.
    for key in ("image_channels", "image_height", "image_width"):
        if key in arch:
            topology["data"][key] = arch[key]

    # `model.connections`, not `model.graph.edges` — the field the
    # compiler's wire schema actually defines (RawModel has `layers`/
    # `global_params`/`connections`, no `graph` key at all). Sending
    # `graph.edges` parsed without error (unknown keys are ignored)
    # but the edges never reached anything: the compiler fell back to
    # its positional-chain assumption regardless of what topology the
    # caller actually described.
    if raw_connections:
        topology["model"]["connections"] = [
            {"from": c["from"], "to": c["to"]}
            for c in raw_connections
            if "from" in c and "to" in c
        ]

    return topology


def _format_sweep_result(result: dict) -> str:
    """Format a /sweep response into a readable summary."""
    points = result.get("result", {}).get("points", [])
    best = result.get("result", {}).get("best")
    feasible_count = sum(1 for p in points if p.get("feasible"))

    if best is None:
        return (
            f"No feasible configuration found among {len(points)} candidates evaluated "
            "— this architecture doesn't fit the configured GPU's VRAM at any swept "
            "batch_size/zero_stage combination. Try a smaller batch range, a higher "
            "zero_stage ceiling, or more GPUs."
        )

    lines = [
        f"Evaluated {len(points)} configurations ({feasible_count} feasible).",
        "",
        "Best configuration:",
        f"  batch_size:  {best['batch_size']}",
        f"  zero_stage:  {best['zero_stage']}",
        f"  gpu_count:   {best['gpu_count']}",
        f"  precision:   {best['precision']}",
        "",
        f"  peak VRAM:        {best['peak_vram_gb']:.2f} GB",
        f"  throughput:       {best['throughput_tokens_per_s']:.1f} tok/s",
        f"  latency:          {best['latency_ms']:.2f} ms",
        f"  training cost:    ${best['training_cost_usd']:.2f}",
    ]
    return "\n".join(lines)


# ─── Tool Definitions ─────────────────────────────────────────────────

AVAILABLE_TOOLS = [
    Tool(
        name="analyze_architecture",
        description="Analyze a neural network architecture using NEURAX compiler. Returns comprehensive metrics including FLOPs, memory, latency, and cost estimates.",
        inputSchema={
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "object",
                    "description": "Architecture definition with nodes, connections, and hardware config",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of layer nodes with type and parameters"
                        },
                        "connections": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of connections between nodes, each {\"from\": node_id, \"to\": node_id}"
                        },
                        "image_channels": {
                            "type": "integer",
                            "description": "Required for cnn/vit/gan/diffusion families — input image channel count (e.g. 3 for RGB).",
                        },
                        "image_height": {
                            "type": "integer",
                            "description": "Required for cnn/vit/gan/diffusion families — input image height in pixels.",
                        },
                        "image_width": {
                            "type": "integer",
                            "description": "Required for cnn/vit/gan/diffusion families — input image width in pixels.",
                        },
                        "hardware": {
                            "type": "object",
                            "description": "Hardware configuration",
                            "properties": {
                                "gpu_name": {"type": "string"},
                                "gpu_count": {"type": "integer"},
                                "gpu_memory_gb": {"type": "number"},
                                "precision": {"type": "string"},
                                "batch_size": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            "required": ["architecture"]
        }
    ),
    Tool(
        name="check_budget",
        description=(
            "Analyze an architecture and check it against deployment budgets "
            "(model size, VRAM, latency). Use this when the user states a hard "
            "constraint such as 'must be under 1 MB' or 'must run in 20 ms on a "
            "phone': it reports pass/fail per constraint with the measured value "
            "and the headroom, so a design can be iterated until it actually fits."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "object",
                    "description": "Same shape as analyze_architecture.",
                },
                "budget": {
                    "type": "object",
                    "description": "Constraints to check. Omit any that do not apply.",
                    "properties": {
                        "max_size_mb": {
                            "type": "number",
                            "description": "Maximum model size on disk, in MB.",
                        },
                        "max_vram_gb": {
                            "type": "number",
                            "description": "Maximum peak VRAM, in GB.",
                        },
                        "max_latency_ms": {
                            "type": "number",
                            "description": "Maximum per-step latency, in ms.",
                        },
                        "max_parameters": {
                            "type": "number",
                            "description": "Maximum parameter count.",
                        },
                    },
                },
            },
            "required": ["architecture", "budget"],
        },
    ),
    Tool(
        name="find_optimal_hyperparameters",
        description=(
            "Search batch_size x zero_stage x gpu_count x precision for the fastest, "
            "cheapest, or largest-feasible-batch training configuration for an "
            "architecture, without running anything — each candidate is a full NEURAX "
            "analysis (no training, no GPU). Use this before committing to a training "
            "config, or as a fast pre-check before spending real compute on a change: "
            "infeasible (out-of-memory) candidates are never selected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "object",
                    "description": "Same shape as analyze_architecture.",
                },
                "objective": {
                    "type": "string",
                    "enum": ["max_throughput", "min_cost", "min_latency", "max_batch_size"],
                    "description": "What to optimize for among feasible candidates. Defaults to max_throughput.",
                },
                "candidates": {
                    "type": "object",
                    "description": (
                        "Candidate values to sweep. Any field left out defaults to a "
                        "standard batch/zero_stage range, anchored to the architecture's "
                        "own gpu_count/precision (the sweep doesn't second-guess hardware "
                        "or precision choices unless you ask it to)."
                    ),
                    "properties": {
                        "batch_sizes": {"type": "array", "items": {"type": "integer"}},
                        "zero_stages": {
                            "type": "array",
                            "items": {"type": "integer", "enum": [0, 1, 2, 3]},
                        },
                        "gpu_counts": {"type": "array", "items": {"type": "integer"}},
                        "precisions": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["architecture"],
        },
    ),
    Tool(
        name="get_hardware_list",
        description="List all available GPU hardware configurations supported by NEURAX.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="get_presets",
        description="List available architecture presets/templates in the NEURAX catalog.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="get_preset",
        description="Get full details of a specific architecture preset.",
        inputSchema={
            "type": "object",
            "properties": {
                "preset_id": {
                    "type": "string",
                    "description": "Preset ID (e.g., gpt2-small, llama-7b, resnet-50)"
                }
            },
            "required": ["preset_id"]
        }
    ),
    Tool(
        name="estimate_training_cost",
        description="Estimate training cost for a given architecture configuration.",
        inputSchema={
            "type": "object",
            "properties": {
                "parameters": {"type": "number", "description": "Total model parameters"},
                "tokens": {"type": "number", "description": "Total training tokens"},
                "gpu_type": {"type": "string", "description": "GPU type (e.g., A100, H100, V100)"},
                "gpu_count": {"type": "integer", "description": "Number of GPUs"},
                "hours": {"type": "number", "description": "Training hours (optional)"}
            },
            "required": ["parameters", "tokens", "gpu_type"]
        }
    ),
    Tool(
        name="get_compliance_config",
        description="Get compliance configuration and regulations (EU AI Act, CSRD, etc.).",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="get_credits",
        description="Get credit balance and usage for the current user.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="get_user_info",
        description="Get current user information and plan tier.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="health_check",
        description="Check if the NEURAX backend service is healthy.",
        inputSchema={"type": "object", "properties": {}}
    ),
]


@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    return AVAILABLE_TOOLS


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name in ("analyze_architecture", "check_budget"):
            arch = arguments.get("architecture", {})
            topology = _build_topology(arch)

            result = await _call_backend("/analyze", method="POST", data={"topology": topology})
            if name == "analyze_architecture":
                return [TextContent(type="text", text=_format_metrics(result))]
            return [TextContent(
                type="text",
                text=_format_budget_report(result, arguments.get("budget", {})),
            )]

        elif name == "find_optimal_hyperparameters":
            arch = arguments.get("architecture", {})
            topology = _build_topology(arch)
            payload: dict[str, Any] = {
                "topology": topology,
                "objective": arguments.get("objective", "max_throughput"),
            }
            candidates = arguments.get("candidates")
            if candidates:
                payload["candidates"] = candidates

            result = await _call_backend("/sweep", method="POST", data=payload)
            return [TextContent(type="text", text=_format_sweep_result(result))]

        elif name == "get_hardware_list":
            result = await _call_backend("/hardware")
            gpus = result if isinstance(result, list) else []
            text = "Available NEURAX Hardware:\n\n"
            for gpu in gpus:
                text += f"- {gpu.get('name', 'Unknown')}: {gpu.get('memory_gb', '?')}GB, "
                text += f"{gpu.get('tflops_fp16', '?')} TFLOPs (FP16), "
                text += f"{gpu.get('memory_bandwidth_gbs', '?')} GB/s\n"
            return [TextContent(type="text", text=text)]

        elif name == "get_presets":
            presets = await _call_backend("/presets")
            text = "Available Architecture Presets:\n\n"
            for p in presets:
                text += f"- {p.get('id', '?')}: {p.get('name', '?')} "
                text += f"({p.get('family', '?')}) — {p.get('description', '')[:100]}\n"
            return [TextContent(type="text", text=text)]

        elif name == "get_preset":
            preset_id = arguments.get("preset_id", "")
            try:
                result = await _call_backend(f"/presets/{preset_id}")
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return [TextContent(type="text", text=f"Preset '{preset_id}' not found.")]
                raise

        elif name == "estimate_training_cost":
            params = float(arguments.get("parameters", 0))
            tokens = float(arguments.get("tokens", 0))
            gpu_type = arguments.get("gpu_type", "A100")
            gpu_count = int(arguments.get("gpu_count", 1))
            hours = arguments.get("hours")

            gpu_hourly_cost = {
                "A100": 2.5, "H100": 4.0, "V100": 1.5, "T4": 0.5, "A10G": 1.0
            }.get(gpu_type, 2.0)

            if hours is None:
                flops = params * tokens * 2
                hours = flops / (gpu_count * 300e12 * 3600)

            cost = hours * gpu_count * gpu_hourly_cost

            text = f"""Training Cost Estimate:
- Model Size: {params / 1e9:.2f}B parameters
- Training Tokens: {tokens / 1e9:.2f}B tokens
- GPU: {gpu_type} x {gpu_count}
- Estimated Hours: {hours:.2f} hours
- Hourly Cost: ${gpu_hourly_cost:.2f}/GPU-hour
- Total Cost: ${cost:.2f}"""
            return [TextContent(type="text", text=text)]

        elif name == "get_compliance_config":
            result = await _call_backend("/compliance/config")
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_credits":
            result = await _call_backend("/credits")
            credits = result.get("credits", {})
            text = f"""Credits:
- Used: {credits.get('used', 0)}
- Limit: {credits.get('limit', 'unlimited')}
- Plan: {credits.get('plan', 'unknown')}
- Period: {credits.get('period_start', '?')} to {credits.get('period_end', '?')}"""
            return [TextContent(type="text", text=text)]

        elif name == "get_user_info":
            result = await _call_backend("/me")
            text = f"""User Info:
- User ID: {result.get('user_id', '?')}
- Plan: {result.get('plan', '?')}"""
            return [TextContent(type="text", text=text)]

        elif name == "health_check":
            result = await _call_backend("/health")
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except httpx.HTTPStatusError as e:
        logger.error(f"Backend error calling {name}: {e}")
        return [TextContent(type="text", text=f"Backend error ({e.response.status_code}): {e.response.text[:500]}")]
    except httpx.RequestError as e:
        logger.error(f"Connection error calling {name}: {e}")
        return [TextContent(type="text", text=f"Cannot reach NEURAX backend at {NEURAX_SERVICE_URL}. Is the server running?")]
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    logger.info("Starting NEURAX MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
