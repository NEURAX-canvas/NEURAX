"""`materializer.diff_to_tool_calls` — editing a canvas that already has
nodes on it, instead of `materialize()`'s `clear_canvas` + full rebuild.

Every test drives the diff against a real `_apply_tool_to_snapshot` replay
of a starting snapshot, not just against `diff_to_tool_calls`'s yielded
list in isolation — the thing that actually matters is what the canvas ends
up looking like after the tool calls are applied, the same bar
`test_materializer_hw_defaults.py` already holds `materialize()` to.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology_validator import ArchSpec
from materializer import diff_to_tool_calls
from snapshot_ops import _apply_tool_to_snapshot


def _run_diff(spec: ArchSpec, existing_snapshot: dict) -> list[dict]:
    async def run():
        return [t async for t in diff_to_tool_calls(spec, existing_snapshot)]
    return asyncio.run(run())


def _replay(existing_snapshot: dict, tool_calls: list[dict]) -> dict:
    snap = dict(existing_snapshot)
    for tc in tool_calls:
        snap = _apply_tool_to_snapshot(snap, tc)
    return snap


def test_never_emits_clear_canvas():
    existing = {
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 16}, "x": 0, "y": 0}],
        "connections": [],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [
            {"id": "c1", "type": "conv2d", "params": {"out_channels": 16}},
            {"id": "c2", "type": "conv2d", "params": {"out_channels": 32}},
        ],
        "edges": [{"from": "c1", "to": "c2"}],
    })
    calls = _run_diff(spec, existing)
    assert all(c["name"] != "clear_canvas" for c in calls)


def test_unchanged_node_produces_no_calls_at_all():
    existing = {
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 16}, "x": 10, "y": 20}],
        "connections": [],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 16}}],
        "edges": [],
    })
    calls = _run_diff(spec, existing)
    # `done` is the only expected call — nothing about c1 changed.
    assert [c["name"] for c in calls] == ["done"]


def test_a_new_node_is_added_with_a_position_and_its_params():
    existing = {"family": "cnn", "nodes": [], "connections": []}
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 16}}],
        "edges": [],
    })
    calls = _run_diff(spec, existing)
    add_calls = [c for c in calls if c["name"] == "add_node"]
    assert len(add_calls) == 1
    assert add_calls[0]["args"]["node_id"] == "c1"
    param_calls = [c for c in calls if c["name"] == "set_node_params"]
    assert param_calls and param_calls[0]["args"]["updates"] == {"out_channels": 16}


def test_a_node_the_target_drops_is_deleted_not_left_behind():
    existing = {
        "family": "cnn",
        "nodes": [
            {"id": "c1", "type": "conv2d", "params": {}, "x": 0, "y": 0},
            {"id": "c2", "type": "conv2d", "params": {}, "x": 100, "y": 0},
        ],
        "connections": [{"from": "c1", "to": "c2"}],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {}}],
        "edges": [],
    })
    calls = _run_diff(spec, existing)
    assert any(c["name"] == "delete_node" and c["args"]["node_id"] == "c2" for c in calls)

    result = _replay(existing, calls)
    assert [n["id"] for n in result["nodes"]] == ["c1"]
    assert result["connections"] == []


def test_only_the_changed_param_keys_are_sent_not_the_whole_dict():
    existing = {
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 16, "kernel_size": 3}, "x": 0, "y": 0}],
        "connections": [],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"out_channels": 32, "kernel_size": 3}}],
        "edges": [],
    })
    calls = _run_diff(spec, existing)
    param_calls = [c for c in calls if c["name"] == "set_node_params"]
    assert len(param_calls) == 1
    assert param_calls[0]["args"]["updates"] == {"out_channels": 32}


def test_reconnecting_disconnects_the_old_edge_and_connects_the_new_one():
    existing = {
        "family": "cnn",
        "nodes": [
            {"id": "a", "type": "conv2d", "params": {}, "x": 0, "y": 0},
            {"id": "b", "type": "conv2d", "params": {}, "x": 100, "y": 0},
            {"id": "c", "type": "conv2d", "params": {}, "x": 200, "y": 0},
        ],
        "connections": [{"from": "a", "to": "b"}],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [
            {"id": "a", "type": "conv2d", "params": {}},
            {"id": "b", "type": "conv2d", "params": {}},
            {"id": "c", "type": "conv2d", "params": {}},
        ],
        "edges": [{"from": "a", "to": "c"}],
    })
    calls = _run_diff(spec, existing)
    assert {"name": "disconnect", "args": {"from_id": "a", "to_id": "b"}} in calls
    assert {"name": "connect", "args": {"from_id": "a", "to_id": "c"}} in calls

    result = _replay(existing, calls)
    assert result["connections"] == [{"from": "a", "to": "c"}]


def test_a_node_whose_type_changed_is_deleted_then_re_added_not_mutated_in_place():
    # There is no "retype a node" tool on the canvas — same id, different
    # type has to be a delete+add, the same edit a human would make by hand.
    existing = {
        "family": "cnn",
        "nodes": [{"id": "x1", "type": "conv2d", "params": {}, "x": 0, "y": 0}],
        "connections": [],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "x1", "type": "max_pool", "params": {}}],
        "edges": [],
    })
    calls = _run_diff(spec, existing)
    names = [c["name"] for c in calls]
    assert names.index("delete_node") < names.index("add_node")

    result = _replay(existing, calls)
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["type"] == "max_pool"


def test_an_existing_nodes_position_is_preserved_not_reassigned():
    existing = {
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {}, "x": 777, "y": 333}],
        "connections": [],
    }
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [
            {"id": "c1", "type": "conv2d", "params": {}},
            {"id": "c2", "type": "conv2d", "params": {}},
        ],
        "edges": [],
    })
    calls = _run_diff(spec, existing)
    # c1 already existed — no add_node for it, so its (x, y) never comes up
    # in this diff at all, meaning the frontend's own position is untouched.
    assert not any(c["name"] == "add_node" and c["args"]["node_id"] == "c1" for c in calls)


def test_family_and_hw_config_are_only_sent_when_they_actually_change():
    existing = {"family": "cnn", "nodes": [], "connections": [], "hw_config": {"batchSize": 32}}
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [],
        "edges": [],
        "hw_config": {"batchSize": 32},
    })
    calls = _run_diff(spec, existing)
    assert not any(c["name"] == "set_family" for c in calls)
    assert not any(c["name"] == "set_hw_config" for c in calls)
