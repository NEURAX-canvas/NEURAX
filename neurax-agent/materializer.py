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


def _existing_index(
    existing_snapshot: dict,
) -> tuple[dict[str, dict], dict[tuple[str, str], bool]]:
    """Index the caller-supplied snapshot's own nodes/connections by id, the
    same lookup shape `snapshot_ops._apply_tool_to_snapshot` uses internally,
    so a diff is computed against what the canvas actually holds rather than
    what the plan assumes it holds."""
    nodes_by_id: dict[str, dict] = {}
    for n in existing_snapshot.get("nodes") or []:
        if isinstance(n, dict) and n.get("id"):
            nodes_by_id[str(n["id"])] = n
    edges: dict[tuple[str, str], bool] = {}
    for c in existing_snapshot.get("connections") or []:
        if not isinstance(c, dict):
            continue
        f = str(c.get("from") or c.get("from_id") or "")
        t = str(c.get("to") or c.get("to_id") or "")
        if f and t:
            edges[(f, t)] = True
    return nodes_by_id, edges


async def diff_to_tool_calls(
    spec: ArchSpec,
    existing_snapshot: dict,
    positions: dict[str, tuple[float, float]] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Yield canvas tool calls that turn the *current* canvas (`existing_snapshot`)
    into `spec`, without a `clear_canvas` — the counterpart to `materialize()`
    for editing an architecture that already has nodes on it, instead of
    always wiping and rebuilding from nothing.

    Only emits a call where something actually differs: a node present in both
    with identical params produces no `set_node_params`, a connection present
    in both produces no `connect`/`disconnect`. A node whose `type` changed is
    treated as delete-then-add — there is no "retype a node in place" tool on
    the canvas, matching how a human editing the canvas would have to do it
    (remove the block, add the right one).

    Yields dicts in the same `{"name", "args"}` shape `materialize()` yields,
    compatible with `snapshot_ops._apply_tool_to_snapshot`.
    """
    existing_nodes, existing_edges = _existing_index(existing_snapshot)
    target_ids = {n.id for n in spec.nodes}

    logger.info(
        f"🔧 Diff-materializing: {len(existing_nodes)} existing -> {len(spec.nodes)} target nodes"
    )

    current_family = str(existing_snapshot.get("family") or "")
    if spec.family and spec.family != current_family:
        yield _tool("set_family", {"family": spec.family})

    existing_hw = dict(existing_snapshot.get("hw_config") or {})
    target_hw = dict(getattr(spec, "hw_config", None) or {})
    hw_diff = {k: v for k, v in target_hw.items() if existing_hw.get(k) != v}
    if hw_diff:
        yield _tool("set_hw_config", {"updates": hw_diff})

    # Nodes the target no longer wants — delete first, so a later `connect`
    # can never target an id about to disappear.
    for node_id in list(existing_nodes.keys()):
        if node_id not in target_ids:
            yield _tool("delete_node", {"node_id": node_id})

    # Rewiring: a connection the current canvas has but the target doesn't,
    # where both endpoints survive (an endpoint being deleted above already
    # takes its incident connections with it — see snapshot_ops.py's
    # `delete_node` case — so re-disconnecting it here would be a no-op sent
    # for a connection that no longer exists by the time this runs).
    target_edges = {(e.from_id, e.to_id) for e in spec.edges}
    for (f, t) in existing_edges:
        if (f, t) not in target_edges and f in target_ids and t in target_ids:
            yield _tool("disconnect", {"from_id": f, "to_id": t})

    # New nodes need a position; a node that already exists keeps the one the
    # user (or a prior run) already gave it rather than being silently moved.
    computed_positions: dict[str, tuple[float, float]] | None = None

    def _position_for(node_id: str) -> tuple[float, float]:
        nonlocal computed_positions
        existing = existing_nodes.get(node_id)
        if existing is not None:
            return (float(existing.get("x") or 0), float(existing.get("y") or 0))
        if positions is not None and node_id in positions:
            return positions[node_id]
        if computed_positions is None:
            computed_positions = assign_positions(spec)
        return computed_positions.get(node_id, (0.0, 0.0))

    for node in spec.nodes:
        prior = existing_nodes.get(node.id)
        if prior is None:
            x, y = _position_for(node.id)
            yield _tool("add_node", {
                "layer_type": node.type,
                "node_id": node.id,
                "x": x,
                "y": y,
            })
            if node.params:
                yield _tool("set_node_params", {"node_id": node.id, "updates": node.params})
            continue

        if str(prior.get("type") or "") != node.type:
            # Same id, different type: the canvas has no "retype" tool, so
            # this is a delete+add even though the id survives — the delete
            # above only covered ids the target dropped entirely.
            yield _tool("delete_node", {"node_id": node.id})
            x, y = _position_for(node.id)
            yield _tool("add_node", {
                "layer_type": node.type,
                "node_id": node.id,
                "x": x,
                "y": y,
            })
            if node.params:
                yield _tool("set_node_params", {"node_id": node.id, "updates": node.params})
            continue

        prior_params = prior.get("params") if isinstance(prior.get("params"), dict) else {}
        params_diff = {k: v for k, v in node.params.items() if prior_params.get(k) != v}
        if params_diff:
            yield _tool("set_node_params", {"node_id": node.id, "updates": params_diff})

    for edge in spec.edges:
        if (edge.from_id, edge.to_id) not in existing_edges:
            yield _tool("connect", {"from_id": edge.from_id, "to_id": edge.to_id})

    hyperparams = getattr(spec, "hyperparams", None) or {}
    existing_hyperparams = existing_snapshot.get("hyperparams") or {}
    hp_diff = {
        k: v for k, v in hyperparams.items() if existing_hyperparams.get(k) != v
    }
    if hp_diff:
        yield _tool("set_hyperparams", {"updates": hp_diff})

    logger.info("✅ Diff-materialization complete")
    yield _tool("done", {})
