"""`_run_agent` must edit a canvas that already has nodes on it in place,
not wipe it with `clear_canvas` and rebuild from nothing — the gap
`materializer.diff_to_tool_calls` and this routing decision close.

Every LLM-touching boundary function is monkeypatched (same style
`test_credentials.py` uses for `make_chat_model`) so this test never makes a
real model call; `allowed_families` is left empty in the snapshot so
`select_family` — which *would* need a real/faked LLM call — is never
invoked at all, matching `_run_agent`'s own `if allowed_families:` guard.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_runner
from topology_validator import ArchSpec, ValidationResult
from budget_check import BudgetReport, SweepReport


async def _fake_plan_strategy(**kwargs):
    return []


def _make_fake_plan_architecture(spec: ArchSpec):
    async def fake(**kwargs):
        return spec
    return fake


def _fake_measure_and_check_ok():
    async def fake(spec, budget, hw_config):
        return BudgetReport(fits=True)
    return fake


async def _fake_optimize_hyperparameters(spec, hw_config):
    return SweepReport(points_evaluated=0)


async def _drive(monkeypatch, snapshot: dict, spec: ArchSpec) -> list[dict]:
    monkeypatch.setattr(agent_runner, "plan_strategy", _fake_plan_strategy)
    monkeypatch.setattr(agent_runner, "plan_architecture", _make_fake_plan_architecture(spec))
    monkeypatch.setattr(agent_runner, "validate_arch_spec", lambda *a, **k: ValidationResult(valid=True, errors=[]))
    monkeypatch.setattr(agent_runner, "measure_and_check", _fake_measure_and_check_ok())
    monkeypatch.setattr(agent_runner, "optimize_hyperparameters", _fake_optimize_hyperparameters)

    q: "asyncio.Queue[dict]" = asyncio.Queue()
    await agent_runner._run_agent("test-run", q, "build it", snapshot)

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def test_a_fresh_empty_canvas_still_uses_the_wipe_and_rebuild_path(monkeypatch):
    snapshot = {"nodes": [], "connections": [], "allowed_families": [], "family": "cnn"}
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {}}],
        "edges": [],
    })

    events = asyncio.run(_drive(monkeypatch, snapshot, spec))

    tool_names = [e["data"]["name"] for e in events if e["event"] == "tool"]
    assert "clear_canvas" in tool_names, (
        "a from-scratch build (nothing on the canvas yet) must still use the "
        "existing, well-tested wipe-and-rebuild path"
    )


def test_a_canvas_with_existing_nodes_is_edited_in_place_no_clear_canvas(monkeypatch):
    snapshot = {
        "nodes": [{"id": "c1", "type": "conv2d", "params": {}, "x": 0, "y": 0}],
        "connections": [],
        "allowed_families": [],
        "family": "cnn",
    }
    # The target spec: keep c1, add c2 — what a real plan_architecture call
    # would produce for an edit request ("add another conv layer").
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [
            {"id": "c1", "type": "conv2d", "params": {}},
            {"id": "c2", "type": "conv2d", "params": {}},
        ],
        "edges": [{"from": "c1", "to": "c2"}],
    })

    events = asyncio.run(_drive(monkeypatch, snapshot, spec))

    tool_names = [e["data"]["name"] for e in events if e["event"] == "tool"]
    assert "clear_canvas" not in tool_names, (
        "editing an existing canvas must never wipe it — this is the whole "
        "point of diff_to_tool_calls"
    )
    assert any(n == "add_node" for n in tool_names), "c2 is new and must still be added"
