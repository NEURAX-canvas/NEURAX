"""The upfront roadmap (`langchain_runner.plan_run_strategy`) and its
binding enforcement in `agent_graph.py`.

Explicitly asked for: NEURAX must not just *display* a plan, it must
*follow* it — `done` is refused while any roadmap step isn't marked
complete, the same enforcement shape the coherence gate already uses.
Every test here overrides the `conftest.py` autouse default (which fakes
"no plan" for every other test in this suite) to exercise a real plan.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph
import langchain_runner


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _fake_plan(items):
    async def fake(**kwargs):
        return [dict(item) for item in items]
    return fake


def _run(monkeypatch, planned_tools, mode="creation", plan_items=None, snapshot=None, max_steps=10):
    monkeypatch.setattr(
        agent_graph, "plan_run_strategy",
        _fake_plan(plan_items if plan_items is not None else [{"id": "1", "text": "Do the one thing"}]),
    )
    calls = {"n": 0}

    async def fake_step(**kw):
        i = calls["n"]
        calls["n"] += 1
        if i < len(planned_tools):
            return {"assistant": f"step {i}", "tool": planned_tools[i]}
        return {"assistant": "wrapping up", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "do it", snapshot or {"nodes": [], "connections": []}, mode=mode, max_steps=max_steps,
        )
        return _drain(q)

    return asyncio.run(go())


# ─── Generation + the plan event ───────────────────────────────────────

def test_a_generated_plan_is_emitted_as_a_plan_event(monkeypatch):
    events = _run(
        monkeypatch, [{"name": "advance_plan_step", "args": {}}],
        plan_items=[{"id": "1", "text": "Add the backbone"}, {"id": "2", "text": "Add the head"}],
    )
    plan_events = [e for e in events if e["event"] == "plan"]
    assert plan_events, "a generated plan must reach the frontend as a plan event"
    first = plan_events[0]["data"]["items"]
    assert first[0]["status"] == "in_progress"
    assert first[1]["status"] == "pending"


def test_no_plan_generated_means_no_plan_event_and_no_enforcement(monkeypatch):
    events = _run(monkeypatch, [{"name": "done", "args": {}}], plan_items=[])
    assert not [e for e in events if e["event"] == "plan"]
    # done succeeds immediately — nothing to enforce.
    tool_events = [e for e in events if e["event"] == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0]["data"]["name"] == "done"


def test_plan_generation_failure_degrades_to_no_plan_not_a_crashed_run(monkeypatch):
    async def failing_plan(**kwargs):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(agent_graph, "plan_run_strategy", failing_plan)

    async def fake_step(**kw):
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph("r", q, "hi", {"nodes": [], "connections": []}, mode="creation")
        return _drain(q)

    events = asyncio.run(go())
    # plan_run_strategy raising must not crash the whole run — done fires.
    assert events[-1]["event"] == "done"
    assert not [e for e in events if e["event"] == "plan"]


# ─── Strict enforcement on `done` ───────────────────────────────────────

def test_done_is_refused_while_a_roadmap_step_is_not_complete(monkeypatch):
    events = _run(monkeypatch, [{"name": "done", "args": {}}], plan_items=[{"id": "1", "text": "Add the backbone"}])
    refusals = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert refusals
    assert "roadmap isn't complete" in refusals[0]["data"]["content"]
    assert "Add the backbone" in refusals[0]["data"]["content"]


def test_done_succeeds_once_every_step_is_advanced(monkeypatch):
    events = _run(
        monkeypatch,
        [{"name": "advance_plan_step", "args": {}}, {"name": "done", "args": {}}],
        plan_items=[{"id": "1", "text": "Only step"}],
    )
    tool_events = [e for e in events if e["event"] == "tool"]
    assert [e["data"]["name"] for e in tool_events] == ["advance_plan_step", "done"]
    refusals = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert not refusals, "done must not be refused once the roadmap is complete"


def test_a_multi_step_plan_requires_advancing_every_item(monkeypatch):
    plan = [{"id": "1", "text": "First"}, {"id": "2", "text": "Second"}]
    events = _run(
        monkeypatch,
        [
            {"name": "done", "args": {}},                    # refused: nothing advanced yet
            {"name": "advance_plan_step", "args": {}},        # completes step 1
            {"name": "done", "args": {}},                     # refused: step 2 still pending
            {"name": "advance_plan_step", "args": {}},        # completes step 2
            {"name": "done", "args": {}},                     # succeeds
        ],
        plan_items=plan,
        max_steps=10,
    )
    refusals = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert len(refusals) == 2
    tool_events = [e for e in events if e["event"] == "tool"]
    assert [e["data"]["name"] for e in tool_events] == ["advance_plan_step", "advance_plan_step", "done"]


def test_advance_plan_step_moves_status_forward_correctly(monkeypatch):
    events = _run(
        monkeypatch,
        [{"name": "advance_plan_step", "args": {}}],
        plan_items=[{"id": "1", "text": "First"}, {"id": "2", "text": "Second"}],
    )
    plan_events = [e for e in events if e["event"] == "plan"]
    # One at start (item 1 in_progress), one after advancing (item 1 done, item 2 in_progress).
    assert len(plan_events) == 2
    after = plan_events[1]["data"]["items"]
    assert after[0]["status"] == "done"
    assert after[1]["status"] == "in_progress"


def test_advancing_with_nothing_in_progress_is_a_harmless_no_op(monkeypatch):
    # Two advances in a row on a one-item plan: the second has nothing left
    # to advance and must not crash or desync the state.
    events = _run(
        monkeypatch,
        [{"name": "advance_plan_step", "args": {}}, {"name": "advance_plan_step", "args": {}}, {"name": "done", "args": {}}],
        plan_items=[{"id": "1", "text": "Only step"}],
    )
    result_events = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "advance_plan_step"]
    assert "no in-progress" in result_events[1]["data"]["content"].lower()
    assert events[-1]["event"] == "done"


def test_advance_plan_step_is_available_in_every_mode_when_a_plan_exists(monkeypatch):
    for mode in ("creation", "optimization", "research", "explanation"):
        events = _run(
            monkeypatch,
            [{"name": "advance_plan_step", "args": {}}, {"name": "done", "args": {}}],
            plan_items=[{"id": "1", "text": "Only step"}],
            mode=mode,
        )
        tool_events = [e for e in events if e["event"] == "tool"]
        names = [e["data"]["name"] for e in tool_events]
        assert names == ["advance_plan_step", "done"], f"mode={mode}: {names}"


def test_the_step_ceiling_still_applies_if_the_roadmap_is_never_advanced(monkeypatch):
    events = _run(
        monkeypatch, [{"name": "done", "args": {}}],
        plan_items=[{"id": "1", "text": "Never advanced"}],
        max_steps=3,
    )
    refusals = [e for e in events if e["event"] == "tool_result" and e["data"]["tool"] == "done"]
    assert len(refusals) == 3
    assert events[-1]["event"] == "done"


# ─── The plan is genuinely mode-aware, not one hard-coded prompt ──────────

def test_plan_run_strategy_frames_the_prompt_by_mode(monkeypatch):
    captured = {}

    class FakeStructured:
        async def ainvoke(self, prompt):
            captured["prompt"] = prompt
            class Out:
                items = []
            return Out()

    monkeypatch.setattr(langchain_runner, "make_chat_model", lambda **kw: type(
        "F", (), {"with_structured_output": lambda self, m: FakeStructured()}
    )())

    asyncio.run(langchain_runner.plan_run_strategy(
        user_message="tune this for a phone", mode="optimization", snapshot={"family": "cnn"},
    ))
    assert "optimizing the existing" in captured["prompt"]

    asyncio.run(langchain_runner.plan_run_strategy(
        user_message="what does this do", mode="explanation", snapshot={"family": "cnn"},
    ))
    assert "explaining the existing" in captured["prompt"]
