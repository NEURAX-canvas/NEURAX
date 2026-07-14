"""NEURAX MCP Server - Model Context Protocol server for NEURAX neural architecture compiler."""

import asyncio
import json
import logging
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurax-mcp")

# Default NEURAX service endpoint (can be overridden via env)
NEURAX_SERVICE_URL = "http://localhost:9098"

# Create MCP server
app = Server("neurax-mcp")


@app.tool(
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
)
async def analyze_architecture(architecture: dict[str, Any]) -> list[TextContent]:
    """Analyze architecture and return compiler metrics."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{NEURAX_SERVICE_URL}/api/analyze",
                json=architecture
            )
            response.raise_for_status()
            result = response.json()
            
            # Format metrics summary
            metrics = result.get("metrics", {})
            summary = f"""
NEURAX Analysis Results:
- Total Parameters: {metrics.get('total_parameters', 0):,}
- Total FLOPs: {metrics.get('total_flops', 0):.2e}
- Peak VRAM: {metrics.get('peak_vram_bytes', 0) / 1e9:.2f} GB
- Estimated Latency: {metrics.get('latency_ms', 0):.2f} ms
- Throughput: {metrics.get('throughput_tokens_per_s', 0):.2f} tok/s
- Training Cost: ${metrics.get('training_cost_usd', 0):.2f}
- Training Time: {metrics.get('training_time_hours', 0):.2f} hours
- Energy: {metrics.get('energy_kwh', 0):.2f} kWh
- CO2: {metrics.get('co2_kg', 0):.2f} kg
"""
            
            return [TextContent(
                type="text",
                text=summary + "\n\nFull JSON:\n" + json.dumps(result, indent=2)
            )]
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return [TextContent(
            type="text",
            text=f"Error analyzing architecture: {str(e)}"
        )]


@app.tool(
    name="get_model_families",
    description="List available neural architecture families in NEURAX catalog (e.g., transformer, cnn, rnn, gnn).",
    inputSchema={"type": "object", "properties": {}}
)
async def get_model_families() -> list[TextContent]:
    """Get available model families from NEURAX catalog."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{NEURAX_SERVICE_URL}/api/catalog/families")
            response.raise_for_status()
            families = response.json()
            
            summary = "Available NEURAX Model Families:\n"
            for family in families:
                summary += f"- {family.get('name', 'Unknown')}: {family.get('description', 'No description')}\n"
            
            return [TextContent(type="text", text=summary)]
    except Exception as e:
        logger.error(f"Failed to get families: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@app.tool(
    name="get_layer_types",
    description="List available layer types for a given model family.",
    inputSchema={
        "type": "object",
        "properties": {
            "family": {
                "type": "string",
                "description": "Model family name (e.g., transformer, cnn)"
            }
        },
        "required": ["family"]
    }
)
async def get_layer_types(family: str) -> list[TextContent]:
    """Get available layer types for a model family."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{NEURAX_SERVICE_URL}/api/catalog/layers",
                params={"family": family}
            )
            response.raise_for_status()
            layers = response.json()
            
            summary = f"Layer Types for '{family}':\n"
            for layer in layers:
                summary += f"- {layer.get('type', 'Unknown')}: {layer.get('description', 'No description')}\n"
                if layer.get('params'):
                    summary += f"  Params: {', '.join(layer['params'])}\n"
            
            return [TextContent(type="text", text=summary)]
    except Exception as e:
        logger.error(f"Failed to get layer types: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@app.tool(
    name="estimate_training_cost",
    description="Estimate training cost for a given architecture configuration.",
    inputSchema={
        "type": "object",
        "properties": {
            "parameters": {"type": "number", "description": "Total model parameters"},
            "tokens": {"type": "number", "description": "Total training tokens"},
            "gpu_type": {"type": "string", "description": "GPU type (e.g., A100, H100)"},
            "gpu_count": {"type": "integer", "description": "Number of GPUs"},
            "hours": {"type": "number", "description": "Training hours"}
        },
        "required": ["parameters", "tokens", "gpu_type"]
    }
)
async def estimate_training_cost(
    parameters: float,
    tokens: float,
    gpu_type: str,
    gpu_count: int = 1,
    hours: float | None = None
) -> list[TextContent]:
    """Estimate training cost using NEURAX cost model."""
    try:
        # Simple cost estimation (in production, call NEURAX service)
        # Approximate: $1-3 per GPU-hour depending on region and GPU type
        gpu_hourly_cost = {
            "A100": 2.5,
            "H100": 4.0,
            "V100": 1.5,
            "T4": 0.5,
            "A10G": 1.0
        }.get(gpu_type, 2.0)
        
        if hours is None:
            # Rough estimate: 1e18 FLOPs per 1B params per 1B tokens
            flops = parameters * tokens * 2  # forward + backward
            # A100 ~ 300 TFLOPs
            hours = flops / (gpu_count * 300e12 * 3600)
        
        cost = hours * gpu_count * gpu_hourly_cost
        
        summary = f"""
Training Cost Estimate:
- Model Size: {parameters / 1e9:.2f}B parameters
- Training Tokens: {tokens / 1e9:.2f}B tokens
- GPU: {gpu_type} x {gpu_count}
- Estimated Hours: {hours:.2f} hours
- Hourly Cost: ${gpu_hourly_cost:.2f}/GPU-hour
- Total Cost: ${cost:.2f}
"""
        return [TextContent(type="text", text=summary)]
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@app.tool(
    name="optimize_memory_layout",
    description="Get memory optimization recommendations for a given architecture.",
    inputSchema={
        "type": "object",
        "properties": {
            "architecture": {
                "type": "object",
                "description": "Architecture definition"
            }
        },
        "required": ["architecture"]
    }
)
async def optimize_memory_layout(architecture: dict[str, Any]) -> list[TextContent]:
    """Get memory optimization recommendations."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{NEURAX_SERVICE_URL}/api/optimize/memory",
                json=architecture
            )
            response.raise_for_status()
            result = response.json()
            
            recommendations = result.get("recommendations", [])
            summary = "Memory Optimization Recommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                summary += f"{i}. {rec.get('title', 'Unknown')}\n"
                summary += f"   {rec.get('description', '')}\n"
                summary += f"   Impact: {rec.get('impact', 'Unknown')}\n\n"
            
            return [TextContent(type="text", text=summary)]
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@app.tool(
    name="get_hardware_profile",
    description="Get hardware profile and capabilities for a specific GPU.",
    inputSchema={
        "type": "object",
        "properties": {
            "gpu_name": {
                "type": "string",
                "description": "GPU name (e.g., A100, H100, V100)"
            }
        },
        "required": ["gpu_name"]
    }
)
async def get_hardware_profile(gpu_name: str) -> list[TextContent]:
    """Get hardware profile for a GPU."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{NEURAX_SERVICE_URL}/api/hardware/{gpu_name}"
            )
            response.raise_for_status()
            profile = response.json()
            
            summary = f"Hardware Profile: {gpu_name}\n"
            summary += f"- VRAM: {profile.get('vram_gb', 0)} GB\n"
            summary += f"- Peak TFLOPs (FP16): {profile.get('peak_tflops', 0)} TFLOPs\n"
            summary += f"- Memory Bandwidth: {profile.get('memory_bandwidth_gbs', 0)} GB/s\n"
            summary += f"- Tensor Cores: {'Yes' if profile.get('has_tensor_cores') else 'No'}\n"
            summary += f"- Interconnect: {profile.get('interconnect', 'N/A')}\n"
            
            return [TextContent(type="text", text=summary)]
    except Exception as e:
        logger.error(f"Failed to get hardware profile: {e}")
        # Fallback to static data
        fallback_profiles = {
            "A100": {"vram_gb": 80, "peak_tflops": 312, "memory_bandwidth_gbs": 1935, "has_tensor_cores": True},
            "H100": {"vram_gb": 80, "peak_tflops": 989, "memory_bandwidth_gbs": 3350, "has_tensor_cores": True},
            "V100": {"vram_gb": 32, "peak_tflops": 125, "memory_bandwidth_gbs": 900, "has_tensor_cores": True},
            "T4": {"vram_gb": 16, "peak_tflops": 65, "memory_bandwidth_gbs": 320, "has_tensor_cores": False},
        }
        if gpu_name in fallback_profiles:
            p = fallback_profiles[gpu_name]
            summary = f"Hardware Profile: {gpu_name} (cached)\n"
            summary += f"- VRAM: {p['vram_gb']} GB\n"
            summary += f"- Peak TFLOPs (FP16): {p['peak_tflops']} TFLOPs\n"
            summary += f"- Memory Bandwidth: {p['memory_bandwidth_gbs']} GB/s\n"
            summary += f"- Tensor Cores: {'Yes' if p['has_tensor_cores'] else 'No'}\n"
            return [TextContent(type="text", text=summary)]
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
