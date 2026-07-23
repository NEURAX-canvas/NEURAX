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
    fields = [
        ("total_parameters", "Total Parameters", "{:,}"),
        ("total_flops", "Total FLOPs", "{:.2e}"),
    ]
    for key, label, fmt in fields:
        val = metrics.get(key, 0)
        lines.append(f"- {label}: {fmt.format(val)}")

    peak_vram = metrics.get("peak_vram_bytes", 0)
    lines.append(f"- Peak VRAM: {peak_vram / 1e9:.2f} GB")

    latency = metrics.get("latency_ms", 0)
    if latency:
        lines.append(f"- Estimated Latency: {latency:.2f} ms")

    throughput = metrics.get("throughput_tokens_per_s", 0)
    if throughput:
        lines.append(f"- Throughput: {throughput:.2f} tok/s")

    cost = metrics.get("training_cost_usd", 0)
    lines.append(f"- Training Cost: ${cost:.2f}")
    hours = metrics.get("training_time_hours", 0)
    lines.append(f"- Training Time: {hours:.2f} hours")
    energy = metrics.get("energy_kwh", 0)
    lines.append(f"- Energy: {energy:.2f} kWh")
    co2 = metrics.get("co2_kg", 0)
    lines.append(f"- CO2: {co2:.2f} kg")

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
                            "description": "List of connections between nodes"
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
        if name == "analyze_architecture":
            arch = arguments.get("architecture", {})
            raw_nodes = arch.get("nodes", [])
            raw_connections = arch.get("connections", [])
            raw_hardware = arch.get("hardware", {})

            # Map frontend layer types to backend-compatible names
            LAYER_TYPE_MAP = {
                "token_embedding": "embedding",
                "embedding": "embedding",
                "pos_embed": "positional_embed",
                "mha_attention": "attention",
                "attention": "attention",
                "cross_attention": "cross_attention",
                "flash_attention": "attention",
                "ffn_standard": "ffn",
                "ffn_gated": "ffn",
                "mlp": "mlp",
                "layer_norm": "normalization",
                "rmsnorm": "normalization",
                "linear": "dense",
                "dense": "dense",
                "conv2d": "conv",
                "conv1d": "conv",
                "max_pool": "pooling",
                "avg_pool": "pooling",
                "input": "embedding",  # input layers mapped to embedding
                "output": "dense",     # output layers mapped to dense
            }

            # Build proper topology format required by NEURAX backend
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

            topology = {
                "schema_version": "1.0",
                "model": {
                    "name": arch.get("name", "MCP Model"),
                    "type": arch.get("family", "transformer"),
                    "layers": layers,
                },
                "training": {
                    "batch_size": raw_hardware.get("batch_size", 1),
                },
                "hardware": {
                    "gpus": [{"name": gpu_name, "count": gpu_count}],
                },
                "data": {
                    "input_shape": arch.get("input_shape", [1, 128]),
                    "dtype": raw_hardware.get("precision", "bfloat16"),
                },
            }

            # Add connections as graph edges
            if raw_connections:
                topology["model"]["graph"] = {
                    "edges": [
                        {"from": c["from"], "to": c["to"]}
                        for c in raw_connections
                        if "from" in c and "to" in c
                    ]
                }

            result = await _call_backend("/analyze", method="POST", data={"topology": topology})
            return [TextContent(type="text", text=_format_metrics(result))]

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
