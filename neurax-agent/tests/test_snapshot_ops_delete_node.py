"""`delete_node` on the server's own snapshot copy.

The frontend (`Index.tsx::handleAgentToolEvent`) has handled `delete_node`,
`navigate_to`, `run_analysis` and `select_node` since before this repo had a
step-by-step loop to emit them — only `snapshot_ops._apply_tool_to_snapshot`,
the server's in-memory view used to build each next prompt, was missing
`delete_node` entirely and had no explicit case for the other three. A loop
that deletes a node and then asks the model what to do next must not still
show that node as present.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_ops import _apply_tool_to_snapshot


def _snapshot(nodes, connections):
    return {"nodes": nodes, "connections": connections}


def test_delete_node_removes_the_node():
    snap = _snapshot(
        nodes=[{"id": "a", "type": "conv2d"}, {"id": "b", "type": "relu"}],
        connections=[],
    )
    out = _apply_tool_to_snapshot(snap, {"name": "delete_node", "args": {"node_id": "a"}})
    assert [n["id"] for n in out["nodes"]] == ["b"]


def test_delete_node_also_removes_its_incident_connections():
    snap = _snapshot(
        nodes=[{"id": "a", "type": "conv2d"}, {"id": "b", "type": "relu"}, {"id": "c", "type": "pool"}],
        connections=[{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
    )
    out = _apply_tool_to_snapshot(snap, {"name": "delete_node", "args": {"node_id": "b"}})
    # Both edges touching 'b' are gone; a stray edge referencing a
    # since-removed id would let a later `connect` reason about a node that
    # no longer exists.
    assert out["connections"] == []
    assert [n["id"] for n in out["nodes"]] == ["a", "c"]


def test_delete_node_on_an_unknown_id_is_a_harmless_no_op():
    snap = _snapshot(nodes=[{"id": "a", "type": "conv2d"}], connections=[])
    out = _apply_tool_to_snapshot(snap, {"name": "delete_node", "args": {"node_id": "does-not-exist"}})
    assert [n["id"] for n in out["nodes"]] == ["a"]


def test_navigate_to_run_analysis_select_node_do_not_change_the_snapshot():
    # None of these three have any canvas-*structure* effect (they switch a
    # tab, trigger a compile, or highlight a node on the frontend) — the
    # server's own copy of nodes/connections must be untouched, and none of
    # them should be silently swallowed as an "unknown tool".
    snap = _snapshot(
        nodes=[{"id": "a", "type": "conv2d"}, {"id": "b", "type": "relu"}],
        connections=[{"from": "a", "to": "b"}],
    )
    for tool_name, args in (
        ("navigate_to", {"tab": "simulation"}),
        ("run_analysis", {}),
        ("select_node", {"node_id": "a"}),
    ):
        before = (list(snap["nodes"]), list(snap["connections"]))
        out = _apply_tool_to_snapshot(snap, {"name": tool_name, "args": args})
        assert (out["nodes"], out["connections"]) == before
