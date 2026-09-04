"""`agent_graph.py` — the step-by-step LangGraph loop.

Every test monkeypatches `agent_graph.run_controller_step` rather than
calling a real model — the same "fake the client, check what got passed
to/read from it" style `test_credentials.py` already uses for
`make_chat_model`. No test here makes a real LLM call.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph


def _drain(q: "asyncio.Queue") -> list[dict]:
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _run(user_message, snapshot, *, credentials=None, history=None, max_steps=10, timeout_seconds=60.0):
    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "test-run",
            q,
            user_message,
            snapshot,
            credentials=credentials,
            conversation_history=history,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
        return _drain(q)
    return asyncio.run(go())


def test_calling_done_stops_the_loop_after_one_step(monkeypatch):
    async def fake(**kwargs):
        return {"assistant": "Nothing to build.", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    events = _run("do nothing", {"nodes": [], "connections": []})

    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["name"] == "done"
    # The stream must always end in `done` — `app.py`'s SSE consumer breaks
    # its loop on exactly this event, same contract `_run_agent` upholds.
    assert events[-1]["event"] == "done"


def test_the_loop_stops_at_the_step_limit_not_before_or_after(monkeypatch):
    calls = {"n": 0}

    async def fake(**kwargs):
        calls["n"] += 1
        return {
            "assistant": f"adding node {calls['n']}",
            "tool": {"name": "add_node", "args": {"layer_type": "conv2d", "node_id": f"n{calls['n']}", "x": 0, "y": 0}},
        }

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    events = _run("build forever", {"nodes": [], "connections": []}, max_steps=3, timeout_seconds=60.0)

    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 3
    assistant_texts = [e["data"].get("content", "") for e in events if e["event"] == "assistant"]
    assert any("3 steps" in t for t in assistant_texts), assistant_texts


def test_the_loop_stops_at_the_wall_clock_ceiling(monkeypatch):
    async def fake(**kwargs):
        return {
            "assistant": "still going",
            "tool": {"name": "add_node", "args": {"layer_type": "conv2d", "node_id": "n", "x": 0, "y": 0}},
        }

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    # A zero-second ceiling means "stop after exactly one step" — deterministic,
    # no real sleeping required to exercise this path.
    events = _run("build forever", {"nodes": [], "connections": []}, max_steps=1000, timeout_seconds=0.0)

    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 1
    assistant_texts = [e["data"].get("content", "") for e in events if e["event"] == "assistant"]
    assert any("time limit" in t for t in assistant_texts), assistant_texts


def test_credentials_reach_run_controller_step(monkeypatch):
    captured = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    creds = {"provider": "anthropic", "api_key": "caller-key"}
    _run("build", {"nodes": [], "connections": []}, credentials=creds)

    assert captured["credentials"] == creds


def test_a_rejected_tool_calls_reason_is_fed_back_into_the_next_steps_history(monkeypatch):
    seen_histories = []

    async def fake(**kwargs):
        seen_histories.append(list(kwargs["history"]))
        if len(seen_histories) == 1:
            # 'a' connecting to itself always hits `_apply_tool_to_snapshot`'s
            # "invalid" rejection path, regardless of block_constraints.json
            # content — a deterministic way to trigger a real rejection.
            return {"assistant": "connecting a to a", "tool": {"name": "connect", "args": {"from_id": "a", "to_id": "a"}}}
        return {"assistant": "giving up cleanly", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    # Coherent on purpose (not the "a" node the self-loop attempt names —
    # the rejection fires on from_id == to_id regardless of whether that id
    # is a real node, see snapshot_ops.py's connect handling): an incoherent
    # starting snapshot would make the second step's `done` call get
    # refused by the coherence check `execute_tool` now runs, which is a
    # second, unrelated rejection this test isn't about.
    snapshot = {
        "nodes": [{"id": "input", "type": "input"}, {"id": "output", "type": "output"}],
        "connections": [{"from": "input", "to": "output"}],
    }
    _run("connect a to a", snapshot)

    assert len(seen_histories) == 2, "the second plan_step call must see the first step's outcome"
    second_call_history = seen_histories[1]
    assert any(
        turn.get("role") == "system" and "rejected" in turn.get("content", "")
        for turn in second_call_history
    ), second_call_history


def test_conversation_history_seeds_the_first_step(monkeypatch):
    captured = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    prior = [{"role": "user", "content": "make it use GQA instead"}]
    _run("continue", {"nodes": [], "connections": []}, history=prior)

    assert captured["history"][0] == prior[0]


def test_a_step_that_raises_still_ends_the_stream_in_done(monkeypatch):
    async def fake(**kwargs):
        raise RuntimeError("the model provider is unreachable")

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    events = _run("build", {"nodes": [], "connections": []})

    assert events[-1]["event"] == "done"
    assert any(e["event"] == "error" for e in events)
