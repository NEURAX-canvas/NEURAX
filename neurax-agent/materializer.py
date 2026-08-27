"""
Materializer — Phase 3 of the declarative agent pipeline.

Converts a validated ArchSpec + positions map into an ordered stream of
canvas tool call dicts. The sequence is deterministic and ordered so that
all `add_node` calls precede `connect` calls (no forward reference issues).
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from topology_validator import ArchSpec
from layout_engine import assign_positions

logger = logging.getLogger(__name__)


def _tool(name: str, args: dict) -> dict:
    return {"name": name, "args": args}


_ATTENTION_TYPES = {"mha", "gqa", "mqa", "mla"}


def _first_param(spec: ArchSpec, *keys: str) -> Optional[int]:
    """First positive value found for any of `keys`, checked across every
    node in plan order — the value a design already states beats a guess."""
    for node in spec.nodes:
        for key in keys:
            value = node.params.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return int(value)
    return None


def _sequence_family_hw_defaults(spec: ArchSpec) -> dict[str, object]:
    """seqLen/numLayers/vocabSize the canvas requires before Run Analysis will
    run for transformer/moe (HardwareContext.tsx's MANDATORY_FIELDS) — derived
    from the design itself rather than a flat guess, since numLayers in
    particular scales total_parameters directly and getting it wrong would
    misreport the very design it describes.

    numLayers counts real attention blocks rather than reading a global
    `num_layers`: this pipeline's designs are individually-typed nodes, not a
    `layer_stack` placeholder repeated N times, so the attention-block count
    already *is* the depth.
    """
    defaults: dict[str, object] = {
        "seqLen": _first_param(spec, "max_len", "max_position_embeddings", "sequence_length") or 2048,
        "numLayers": sum(1 for n in spec.nodes if n.type in _ATTENTION_TYPES) or 1,
        "vocabSize": _first_param(spec, "vocab_size") or 32000,
    }
    if spec.family == "moe":
        defaults["numExperts"] = _first_param(spec, "num_experts") or 8
        defaults["topK"] = _first_param(spec, "top_k", "topk") or 2
    return defaults


async def materialize(
    spec: ArchSpec,
    positions: dict[str, tuple[float, float]] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Yield canvas tool call dicts for a validated ArchSpec.

    Tool call order:
        1. clear_canvas
        2. set_family
        3. add_node × N   (all nodes first, in topological order)
        4. set_node_params × N  (params for each node that has them)
        5. connect × M    (all edges after all nodes exist)
        6. done

    Args:
        spec:      Validated ArchSpec
        positions: Optional pre-computed positions. If None, layout_engine is called.

    Yields:
        dict with keys "name" and "args" — compatible with snapshot_ops._apply_tool_to_snapshot
    """
    if positions is None:
        positions = assign_positions(spec)

    logger.info(f"🔧 Materializing: {len(spec.nodes)} nodes, {len(spec.edges)} edges")

    # 1. Clear canvas
    yield _tool("clear_canvas", {})

    # 2. Set family
    yield _tool("set_family", {"family": spec.family})

    # Platform settings before the graph: the canvas derives block defaults from
    # them, and they decide the model's cost as much as the blocks do.
    hw_config = dict(getattr(spec, "hw_config", None) or {})

    # The canvas gates Run Analysis on a family-specific set of hw_config
    # fields (HardwareContext.tsx's MANDATORY_FIELDS) that has nothing to do
    # with what any individual node's own params carry — a conv node can
    # have a perfectly correct in_channels and the build is still stuck
    # forever on "Missing required field: inChannels" if this *global* field
    # was never set. Nothing upstream (the planning prompt, `set_hw_config`)
    # reliably sets these for every family, so backstop them here
    # deterministically rather than hope the plan happened to include them —
    # this is the same fallback-of-last-resort pattern as
    # `initialize_hyperparams` below, just for the platform config instead
    # of the training hyperparameters.
    _HW_MANDATORY_DEFAULTS: dict[str, dict[str, object]] = {
        "cnn": {"inChannels": 3, "imgHeight": 224, "imgWidth": 224},
        "vit": {"inChannels": 3, "imgHeight": 224, "imgWidth": 224},
        "diffusion": {"inChannels": 3, "imgHeight": 64, "imgWidth": 64, "numDenoisingSteps": 1000},
        "gnn": {"numNodes": 1000, "numEdges": 5000, "nodeFeatDim": 64},
        "ssm": {"dState": 16},
    }
    family_defaults = (
        _sequence_family_hw_defaults(spec)
        if spec.family in ("transformer", "moe")
        else _HW_MANDATORY_DEFAULTS.get(spec.family, {})
    )
    for key, default in family_defaults.items():
        hw_config.setdefault(key, default)

    if hw_config:
        yield _tool("set_hw_config", {"updates": hw_config})

    # 3. Add all nodes (in topological order from positions keys)
    # Sort nodes so input comes first, output comes last
    def _node_sort_key(node):
        if node.type == "input":
            return (0, node.id)
        if node.type == "output":
            return (2, node.id)
        return (1, positions.get(node.id, (0, 0))[0])  # sort by x position

    sorted_nodes = sorted(spec.nodes, key=_node_sort_key)

    for node in sorted_nodes:
        x, y = positions.get(node.id, (0.0, 0.0))
        yield _tool("add_node", {
            "layer_type": node.type,
            "node_id": node.id,
            "x": x,
            "y": y,
        })

        # 4. Set params immediately after adding node
        if node.params:
            yield _tool("set_node_params", {
                "node_id": node.id,
                "updates": node.params,
            })

    # 5. Connect all edges
    for edge in spec.edges:
        yield _tool("connect", {
            "from_id": edge.from_id,
            "to_id": edge.to_id,
        })

    # 6. Done
    logger.info(f"✅ Materialization complete")
    # Hyperparameters last, once the graph and hardware are in place: the
    # defaults are derived from the layer count and the hardware config, so they
    # are only meaningful after both exist.
    yield _tool("initialize_hyperparams", {})

    hyperparams = getattr(spec, "hyperparams", None) or {}
    if hyperparams:
        yield _tool("set_hyperparams", {"updates": hyperparams})

    yield _tool("done", {})
