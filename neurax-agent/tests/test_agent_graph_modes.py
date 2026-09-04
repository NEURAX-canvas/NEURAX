"""Phase 3 — the four modes' least-privilege tool grants.

Two layers are tested separately, matching how they're actually enforced:
1. `langchain_runner._build_tools_section` / the catalogue-skip logic — the
   *prompt* the model sees never mentions an ungranted tool (token
   efficiency and good behavior, not the security boundary).
2. `agent_graph.execute_tool`'s own check against `state["allowed_tools"]`
   — the *real* boundary: a hallucinated call to an ungranted tool is
   refused here even if the model was never told about it, exactly
   mirroring the reasoning behind OpenClaw's "tools simply don't exist in
   restricted modes" permission model researched for this plan.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph
import analysis_tools
import langchain_runner


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _run_one_step(monkeypatch, snapshot, tool, mode, **kwargs):
    """Fake the planner into proposing exactly `tool` on step 1, `done` on
    every step after — returns the drained SSE events."""
    calls = {"n": 0}

    async def fake_once(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"assistant": "step", "tool": tool}
        return {"assistant": "done", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_once)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph("test-run", q, "do it", snapshot, mode=mode, **kwargs)
        return _drain(q)

    return asyncio.run(go())


# ─── MODE_TOOL_GRANTS sanity ────────────────────────────────────────────────

def test_every_mode_grant_names_only_real_described_tools():
    for mode, grant in agent_graph.MODE_TOOL_GRANTS.items():
        unknown = grant - set(langchain_runner.ALL_TOOL_DESCRIPTIONS)
        assert not unknown, f"mode '{mode}' grants undescribed tools: {unknown}"


def test_all_four_modes_are_present():
    assert set(agent_graph.MODE_TOOL_GRANTS) == {"creation", "optimization", "research", "explanation"}


def test_optimization_mode_cannot_add_remove_or_rewire_blocks():
    grant = agent_graph.MODE_TOOL_GRANTS["optimization"]
    assert not (grant & {"add_node", "delete_node", "connect", "disconnect", "set_family"})
    # But it can still tune what's already there.
    assert {"set_node_params", "set_hw_config", "set_hyperparams"} <= grant


def test_explanation_mode_grants_no_canvas_mutation_tool_at_all():
    canvas_mutation_tools = {
        "set_family", "add_node", "connect", "disconnect", "delete_node",
        "set_node_params", "set_hw_config", "initialize_hyperparams", "set_hyperparams",
    }
    grant = agent_graph.MODE_TOOL_GRANTS["explanation"]
    assert not (grant & canvas_mutation_tools)


# ─── Prompt-level filtering (langchain_runner) ─────────────────────────────

def test_the_tools_section_only_lists_granted_tools():
    section = langchain_runner._build_tools_section(frozenset({"analyze_architecture"}))
    assert "analyze_architecture" in section
    assert "add_node" not in section
    assert "find_optimal_hyperparameters" not in section
    # 'done' is always present regardless of what's explicitly granted.
    assert "`done`" in section


def test_no_restriction_lists_every_tool():
    section = langchain_runner._build_tools_section(None)
    for name in langchain_runner.ALL_TOOL_DESCRIPTIONS:
        assert f"`{name}`" in section, name


def test_a_mode_without_add_node_skips_the_block_catalogue_entirely(monkeypatch):
    """Token-efficiency check: a mode that can't place a new block has no
    use for ~100 lines of catalogue text, resent every single step."""
    captured = {}

    class FakeStructured:
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            class Out:
                assistant = "ok"
                class tool:
                    name = "done"
                    args = {}
            return Out()

    monkeypatch.setattr(langchain_runner, "make_chat_model", lambda **kw: type(
        "F", (), {"with_structured_output": lambda self, m: FakeStructured()}
    )())

    snapshot = {
        "family": "cnn",
        "allowed_families": ["cnn"],
        "catalogue": [{"type": "conv2d", "name": "Conv2D", "defaultParams": {"out_channels": 16}}],
        "nodes": [], "connections": [], "hw_config": {}, "missing_mandatory_fields": [], "analysis_warnings": [],
    }

    asyncio.run(langchain_runner.run_controller_step(
        user_message="explain", snapshot=snapshot, history=[],
        allowed_tools=agent_graph.MODE_TOOL_GRANTS["explanation"], mode="explanation",
    ))
    explanation_prompt = captured["system"]
    assert "Block Catalogue" not in explanation_prompt

    asyncio.run(langchain_runner.run_controller_step(
        user_message="build", snapshot=snapshot, history=[],
        allowed_tools=agent_graph.MODE_TOOL_GRANTS["creation"], mode="creation",
    ))
    creation_prompt = captured["system"]
    assert "Block Catalogue" in creation_prompt
    assert "conv2d" in creation_prompt


# ─── Execution-level structural gate (agent_graph.execute_tool) ───────────

def test_a_canvas_tool_outside_the_mode_grant_is_refused_not_executed(monkeypatch):
    # `_apply_tool_to_snapshot` is called synchronously (no `await`) in the
    # real code — the spy must be a plain function too. An `async def` spy
    # would silently "pass" this test even with a broken gate: calling an
    # async function without awaiting it creates a coroutine object without
    # ever running its body, so `called['n']` would stay 0 either way.
    # The fake planner's second step (after the refusal) proposes `done`,
    # which legitimately does reach `_apply_tool_to_snapshot` too (that is
    # normal, pre-existing behavior, unrelated to this test) — so the spy
    # records *which* tool names it was called with, rather than just a
    # call count, and the assertion below checks specifically for add_node.
    calls_with: list[str] = []

    def spy_apply(snapshot, tool):
        calls_with.append(str(tool.get("name")))
        return snapshot

    monkeypatch.setattr(agent_graph, "_apply_tool_to_snapshot", spy_apply)

    snapshot = {"nodes": [], "connections": []}
    events = _run_one_step(
        monkeypatch, snapshot,
        {"name": "add_node", "args": {"layer_type": "conv2d", "node_id": "n1", "x": 0, "y": 0}},
        mode="optimization",  # optimization does not grant add_node
    )

    assert "add_node" not in calls_with, "add_node must never reach _apply_tool_to_snapshot in optimization mode"
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("not available in 'optimization' mode" in e["data"]["content"] for e in result_events)


def test_an_analysis_tool_outside_the_mode_grant_is_refused_not_dispatched(monkeypatch):
    called = {"n": 0}

    async def spy_dispatch(name, args, snapshot):
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr(analysis_tools, "dispatch", spy_dispatch)

    events = _run_one_step(
        monkeypatch, {"nodes": [], "connections": []},
        {"name": "find_optimal_hyperparameters", "args": {}},
        mode="creation",  # creation does not grant find_optimal_hyperparameters
    )

    assert called["n"] == 0
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("not available in 'creation' mode" in e["data"]["content"] for e in result_events)


def test_done_is_always_allowed_regardless_of_mode(monkeypatch):
    events = _run_one_step(
        monkeypatch, {"nodes": [], "connections": []},
        {"name": "done", "args": {}},
        mode="explanation",
    )
    tool_events = [e for e in events if e["event"] == "tool"]
    assert tool_events[-1]["data"]["name"] == "done"
    assert events[-1]["event"] == "done"


def test_a_granted_tool_still_runs_normally(monkeypatch):
    """The gate must not be so strict it blocks what a mode actually grants."""
    snapshot = {"nodes": [], "connections": []}
    events = _run_one_step(
        monkeypatch, snapshot,
        {"name": "add_node", "args": {"layer_type": "conv2d", "node_id": "n1", "x": 0, "y": 0}},
        mode="creation",  # creation does grant add_node
    )
    tool_events = [e for e in events if e["event"] == "tool"]
    assert tool_events[0]["data"]["name"] == "add_node"
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert not any("not available" in e["data"]["content"] for e in result_events)


# ─── explain_layer_type (explanation mode's own lookup tool) ──────────────

def test_explain_layer_type_reads_the_snapshots_own_catalogue(monkeypatch):
    snapshot = {
        "nodes": [], "connections": [],
        "catalogue": [{"type": "rmsnorm", "name": "RMSNorm", "description": "Root mean square normalization"}],
    }
    events = _run_one_step(
        monkeypatch, snapshot,
        {"name": "explain_layer_type", "args": {"layer_type": "rmsnorm"}},
        mode="explanation",
    )
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("Root mean square normalization" in e["data"]["content"] for e in result_events)


def test_explain_layer_type_on_an_unknown_type_says_so_plainly(monkeypatch):
    snapshot = {"nodes": [], "connections": [], "catalogue": []}
    events = _run_one_step(
        monkeypatch, snapshot,
        {"name": "explain_layer_type", "args": {"layer_type": "not_a_real_block"}},
        mode="explanation",
    )
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("was not found" in e["data"]["content"] for e in result_events)


def test_explain_layer_type_never_touches_the_snapshot(monkeypatch):
    original_nodes = [{"id": "a", "type": "conv2d"}]
    snapshot = {"nodes": list(original_nodes), "connections": []}
    _run_one_step(
        monkeypatch, snapshot,
        {"name": "explain_layer_type", "args": {"layer_type": "conv2d"}},
        mode="explanation",
    )
    assert snapshot["nodes"] == original_nodes


# ─── app.py: mode replaces creativity ──────────────────────────────────────

def test_run_request_has_mode_not_creativity():
    import app
    fields = app.RunRequest.model_fields
    assert "mode" in fields
    assert "creativity" not in fields
    assert not fields["mode"].is_required()  # defaults, an older client sending none still works


def test_run_request_mode_defaults_to_creation():
    import app
    req = app.RunRequest(user_message="hi", snapshot=app.CanvasSnapshot())
    assert req.mode == "creation"


def test_run_request_rejects_an_unknown_mode():
    import app
    import pydantic
    try:
        app.RunRequest(user_message="hi", snapshot=app.CanvasSnapshot(), mode="not_a_real_mode")
        assert False, "an unrecognized mode should be rejected at the request boundary"
    except pydantic.ValidationError:
        pass
