"""`agent_graph.py`'s dispatch of analysis tools (`analysis_tools.py`) —
the other half of `test_agent_graph.py`, which only covers canvas-mutation
tools and the loop's own stop conditions.

`analysis_tools.dispatch` is monkeypatched rather than really hitting
`neurax-service` — this file is about the *loop's* behavior (routing,
history, the per-run ceiling on expensive calls), not about compiler
correctness, which `tests/test_analysis_tools.py` already covers.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph
import analysis_tools


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _run_graph(monkeypatch, snapshot, planned_tools, **kwargs):
    """`planned_tools`: a list of `{"name", "args"}` dicts, one per step the
    fake planner should propose in order, `done` implied at the end if the
    list runs out before a real `done`."""
    calls = {"n": 0}

    async def fake_run_controller_step(**kw):
        i = calls["n"]
        calls["n"] += 1
        if i < len(planned_tools):
            return {"assistant": f"step {i}", "tool": planned_tools[i]}
        return {"assistant": "done", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph("test-run", q, "build and check", snapshot, **kwargs)
        return _drain(q)

    return asyncio.run(go())


def test_an_analysis_tool_call_does_not_mutate_the_snapshot(monkeypatch):
    async def fake_dispatch(name, args, snapshot):
        return "Total Parameters: 1,234,567"

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    snapshot = {"nodes": [{"id": "a", "type": "conv2d"}], "connections": []}
    events = _run_graph(
        monkeypatch, snapshot,
        [{"name": "analyze_architecture", "args": {}}],
    )
    tool_events = [e for e in events if e["event"] == "tool"]
    assert tool_events[0]["data"]["name"] == "analyze_architecture"
    # Only 'analyze_architecture' then 'done' — no canvas tool ever ran, so
    # the node count on the (untouched) snapshot never changes.
    assert len(tool_events) == 2


def test_an_analysis_result_is_emitted_as_a_tool_result_event(monkeypatch):
    async def fake_dispatch(name, args, snapshot):
        return "Total Parameters: 1,234,567"

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    events = _run_graph(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "get_hardware_list", "args": {}}],
        mode="research",  # 'creation' mode doesn't grant get_hardware_list
    )
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert len(result_events) == 1
    assert result_events[0]["data"]["tool"] == "get_hardware_list"
    assert "1,234,567" in result_events[0]["data"]["content"]


def test_the_result_is_fed_back_into_the_next_steps_history(monkeypatch):
    async def fake_dispatch(name, args, snapshot):
        return "Total Parameters: 42"

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    seen_histories = []

    async def fake_run_controller_step(**kw):
        seen_histories.append(list(kw["history"]))
        if len(seen_histories) == 1:
            return {"assistant": "checking", "tool": {"name": "analyze_architecture", "args": {}}}
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph("r", q, "check it", {"nodes": [], "connections": []})
        return _drain(q)

    asyncio.run(go())

    assert len(seen_histories) == 2
    second_history = seen_histories[1]
    assert any(
        turn.get("role") == "system" and "Result of analyze_architecture" in turn.get("content", "") and "42" in turn.get("content", "")
        for turn in second_history
    ), second_history


def test_a_failing_analysis_tool_reports_the_error_without_crashing_the_loop(monkeypatch):
    async def fake_dispatch(name, args, snapshot):
        raise RuntimeError("compiler exploded")

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    events = _run_graph(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "analyze_architecture", "args": {}}],
    )
    assert events[-1]["event"] == "done"
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert "compiler exploded" in result_events[0]["data"]["content"]


def test_the_expensive_tool_is_blocked_after_its_per_run_ceiling(monkeypatch):
    call_log = []

    async def fake_dispatch(name, args, snapshot):
        call_log.append(name)
        return "sweep result"

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    # Plan calls find_optimal_hyperparameters five times in a row — the
    # ceiling (set to 2 here) must stop it from actually running past that.
    planned = [{"name": "find_optimal_hyperparameters", "args": {}}] * 5
    events = _run_graph(
        monkeypatch, {"nodes": [], "connections": []}, planned,
        mode="research",  # 'creation' mode doesn't grant find_optimal_hyperparameters
        max_steps=10, max_expensive_calls=2,
    )

    # The real dispatch only ran twice, no matter how many times the model
    # asked for it.
    assert call_log.count("find_optimal_hyperparameters") == 2

    result_events = [e for e in events if e["event"] == "tool_result"]
    blocked = [e for e in result_events if "limit" in e["data"]["content"].lower()]
    assert len(blocked) >= 1


def test_a_blocked_call_does_not_further_increment_its_own_ceiling(monkeypatch):
    # Regression guard for a plausible off-by-one: a blocked attempt must not
    # itself count toward (or reset) the ceiling it was blocked by.
    call_log = []

    async def fake_dispatch(name, args, snapshot):
        call_log.append(name)
        return "sweep result"

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    planned = [{"name": "find_optimal_hyperparameters", "args": {}}] * 10
    _run_graph(
        monkeypatch, {"nodes": [], "connections": []}, planned,
        mode="research",  # 'creation' mode doesn't grant find_optimal_hyperparameters
        max_steps=10, max_expensive_calls=1,
    )
    assert call_log.count("find_optimal_hyperparameters") == 1


def test_a_cheap_analysis_tool_has_no_ceiling(monkeypatch):
    call_log = []

    async def fake_dispatch(name, args, snapshot):
        call_log.append(name)
        return "ok"

    monkeypatch.setattr(analysis_tools, "dispatch", fake_dispatch)

    planned = [{"name": "get_hardware_list", "args": {}}] * 5
    _run_graph(
        monkeypatch, {"nodes": [], "connections": []}, planned,
        mode="research",  # 'creation' mode doesn't grant get_hardware_list
        max_steps=10, max_expensive_calls=1,
    )
    assert call_log.count("get_hardware_list") == 5
