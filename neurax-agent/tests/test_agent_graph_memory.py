"""Phase 5 — agentic memory wired into `agent_graph.py`'s loop.

`memory_tools` is monkeypatched throughout (matching `analysis_tools`'s and
`web_search_tools`'s own test style) — this file is about the *loop's*
behavior (grants, recall-at-start, persistence-at-end, graceful
degradation with no project), not about `neurax-service`'s real Supabase
REST calls, which `tests/test_memory_tools.py` already covers against a
faked HTTP client.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph
import memory_tools


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _run_scripted(monkeypatch, snapshot, planned_tools, mode, project_id=None, **kwargs):
    """`planned_tools`: a list of `{"name","args"}` dicts proposed in order;
    `done` implied once the list runs out."""
    calls = {"n": 0}

    async def fake(**kw):
        i = calls["n"]
        calls["n"] += 1
        if i < len(planned_tools):
            return {"assistant": f"step {i}", "tool": planned_tools[i]}
        return {"assistant": "wrapping up", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "test-run", q, "do it", snapshot, mode=mode, project_id=project_id, **kwargs
        )
        return _drain(q)

    return asyncio.run(go())


# ─── remember_preference ───────────────────────────────────────────────

def test_remember_preference_saves_with_a_project_id(monkeypatch):
    captured = {}

    async def fake_add(project_id, preference):
        captured["project_id"] = project_id
        captured["preference"] = preference
        return True

    monkeypatch.setattr(memory_tools, "add_core_preference", fake_add)
    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)

    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "remember_preference", "args": {"preference": "prefers GQA over MHA"}}],
        mode="creation", project_id="proj-1",
    )

    assert captured == {"project_id": "proj-1", "preference": "prefers GQA over MHA"}
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("Remembered: prefers GQA over MHA" in e["data"]["content"] for e in result_events)


def test_remember_preference_without_a_project_id_is_refused_gracefully(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(memory_tools, "add_core_preference", spy)

    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "remember_preference", "args": {"preference": "x"}}],
        mode="creation", project_id=None,
    )

    assert called["n"] == 0
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("isn't saved as a project" in e["data"]["content"] for e in result_events)


def test_remember_preference_is_not_granted_in_explanation_mode(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(memory_tools, "add_core_preference", spy)
    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)

    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "remember_preference", "args": {"preference": "x"}}],
        mode="explanation", project_id="proj-1",
    )

    assert called["n"] == 0
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("not available in 'explanation' mode" in e["data"]["content"] for e in result_events)


# ─── search_past_designs ────────────────────────────────────────────────

def test_search_past_designs_returns_real_entries(monkeypatch):
    async def fake_search(project_id, query, limit=5):
        assert project_id == "proj-1"
        assert query == "moe design"
        return ["Built an 8x7B MoE for cost reasons."]

    monkeypatch.setattr(memory_tools, "search_past_designs", fake_search)
    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)
    monkeypatch.setattr(memory_tools, "add_archival_entry", _noop_bool)

    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "search_past_designs", "args": {"query": "moe design"}}],
        mode="research", project_id="proj-1",
    )

    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("Built an 8x7B MoE for cost reasons." in e["data"]["content"] for e in result_events)


def test_search_past_designs_distinguishes_a_service_failure_from_genuinely_empty(monkeypatch):
    # None (memory_tools failed) must produce a different message than []
    # (the project genuinely has no past designs) — telling the model "no
    # past designs" when the real cause was an unreachable service is a
    # false fact, not a graceful degradation.
    async def failing_search(project_id, query, limit=5):
        return None

    monkeypatch.setattr(memory_tools, "search_past_designs", failing_search)
    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)

    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "search_past_designs", "args": {"query": "x"}}],
        mode="research", project_id="proj-1",
    )
    result_events = [e for e in events if e["event"] == "tool_result"]
    content = next(e["data"]["content"] for e in result_events if e["data"]["tool"] == "search_past_designs")
    assert "unavailable" in content.lower()
    assert "no past designs found" not in content.lower()


def test_search_past_designs_without_a_project_id_says_so(monkeypatch):
    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "search_past_designs", "args": {"query": "x"}}],
        mode="research", project_id=None,
    )
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("isn't saved as a project" in e["data"]["content"] for e in result_events)


def test_search_past_designs_is_not_granted_in_optimization_mode(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(memory_tools, "search_past_designs", spy)
    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)

    events = _run_scripted(
        monkeypatch, {"nodes": [], "connections": []},
        [{"name": "search_past_designs", "args": {"query": "x"}}],
        mode="optimization", project_id="proj-1",
    )
    assert called["n"] == 0
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("not available in 'optimization' mode" in e["data"]["content"] for e in result_events)


# ─── Core memory recall at run start ────────────────────────────────────

def test_core_memory_is_fetched_and_reaches_the_first_plan_step_call(monkeypatch):
    async def fake_get_core(project_id):
        assert project_id == "proj-1"
        return ["prefers int8", "target: mobile"]

    monkeypatch.setattr(memory_tools, "get_core_preferences", fake_get_core)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)

    captured = {}

    async def fake_run_controller_step(**kw):
        captured.update(kw)
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "hi", {"nodes": [], "connections": []}, mode="creation", project_id="proj-1"
        )

    asyncio.run(go())
    assert captured["core_memory"] == ["prefers int8", "target: mobile"]


def test_recalled_conversation_is_prepended_to_history(monkeypatch):
    async def fake_get_recent(project_id, limit=8):
        return [{"role": "user", "content": "earlier turn"}]

    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", fake_get_recent)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)

    captured = {}

    async def fake_run_controller_step(**kw):
        captured.update(kw)
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "continue", {"nodes": [], "connections": []},
            mode="creation", project_id="proj-1",
            conversation_history=[{"role": "user", "content": "this request"}],
        )

    asyncio.run(go())
    assert captured["history"][0] == {"role": "user", "content": "earlier turn"}
    assert captured["history"][1] == {"role": "user", "content": "this request"}


def test_cancellation_during_memory_recall_still_emits_done(monkeypatch):
    # Regression guard: an earlier version fetched recall *before* the
    # try/finally that guarantees `done` fires — a cancellation landing
    # during that fetch (a real network call) would have propagated
    # CancelledError out of run_agent_graph with `done` never queued at
    # all, breaking the one invariant every other test in this file
    # depends on, just from an angle none of them exercised.
    async def cancel_during_fetch(project_id):
        raise asyncio.CancelledError()

    monkeypatch.setattr(memory_tools, "get_core_preferences", cancel_during_fetch)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        try:
            await agent_graph.run_agent_graph(
                "r", q, "hi", {"nodes": [], "connections": []}, mode="creation", project_id="proj-1"
            )
        except asyncio.CancelledError:
            pass
        return _drain(q)

    events = asyncio.run(go())
    assert events, "the queue must have received events even though the run was cancelled"
    assert events[-1]["event"] == "done"


def test_with_no_project_id_no_recall_call_is_made(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(memory_tools, "get_core_preferences", spy)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", spy)

    _run_scripted(monkeypatch, {"nodes": [], "connections": []}, [], mode="creation", project_id=None)
    assert called["n"] == 0


# ─── Persistence at the end of a run ────────────────────────────────────

def test_this_runs_turn_is_appended_to_conversation_memory(monkeypatch):
    captured = {}

    async def fake_append(project_id, turns):
        captured["project_id"] = project_id
        captured["turns"] = turns
        return True

    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", fake_append)

    async def fake_run_controller_step(**kw):
        return {"assistant": "built it", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "build a small cnn", {"nodes": [], "connections": []},
            mode="creation", project_id="proj-1",
        )

    asyncio.run(go())
    assert captured["project_id"] == "proj-1"
    assert captured["turns"][0] == {"role": "user", "content": "build a small cnn"}
    assert captured["turns"][1]["role"] == "assistant"
    assert "built it" in captured["turns"][1]["content"]


def test_a_successful_creation_run_saves_an_archival_summary(monkeypatch):
    captured = {}

    async def fake_add_archival(project_id, content):
        captured["project_id"] = project_id
        captured["content"] = content
        return True

    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)
    monkeypatch.setattr(memory_tools, "add_archival_entry", fake_add_archival)

    async def fake_run_controller_step(**kw):
        return {"assistant": "Added a ResNet backbone for image input.", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "build a resnet", {"nodes": [], "connections": []},
            mode="creation", project_id="proj-1",
        )

    asyncio.run(go())
    assert captured["project_id"] == "proj-1"
    assert "ResNet backbone" in captured["content"]


def test_no_archival_entry_is_saved_in_optimization_mode(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)
    monkeypatch.setattr(memory_tools, "add_archival_entry", spy)

    async def fake_run_controller_step(**kw):
        return {"assistant": "tuned the batch size", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "optimize it", {"nodes": [], "connections": []},
            mode="optimization", project_id="proj-1",
        )

    asyncio.run(go())
    assert called["n"] == 0, "optimization mode tunes a design, it doesn't produce a new one worth archiving"


def test_no_archival_entry_is_saved_when_the_run_stops_on_the_step_limit(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", _noop_bool)
    monkeypatch.setattr(memory_tools, "add_archival_entry", spy)

    async def fake_run_controller_step(**kw):
        return {
            "assistant": "still going",
            "tool": {"name": "add_node", "args": {"layer_type": "conv2d", "node_id": "n", "x": 0, "y": 0}},
        }

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "build forever", {"nodes": [], "connections": []},
            mode="creation", project_id="proj-1", max_steps=2,
        )

    asyncio.run(go())
    assert called["n"] == 0, "an incomplete design (stopped on the ceiling, not `done`) isn't worth archiving as a finished one"


def test_a_memory_persistence_failure_never_blocks_the_done_event(monkeypatch):
    async def failing_append(*a, **kw):
        raise RuntimeError("neurax-service unreachable")

    monkeypatch.setattr(memory_tools, "get_core_preferences", _empty_list)
    monkeypatch.setattr(memory_tools, "get_recent_conversation", _empty_list)
    monkeypatch.setattr(memory_tools, "append_conversation_turns", failing_append)

    async def fake_run_controller_step(**kw):
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_run_controller_step)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "r", q, "hi", {"nodes": [], "connections": []}, mode="creation", project_id="proj-1"
        )
        return _drain(q)

    events = asyncio.run(go())
    assert events[-1]["event"] == "done", "the done event must fire even if memory persistence blows up"


# ─── app.py: project_id reaches the request ────────────────────────────

def test_run_request_accepts_an_optional_project_id():
    import app
    req = app.RunRequest(user_message="hi", snapshot=app.CanvasSnapshot(), project_id="proj-1")
    assert req.project_id == "proj-1"


def test_run_request_project_id_defaults_to_none():
    import app
    req = app.RunRequest(user_message="hi", snapshot=app.CanvasSnapshot())
    assert req.project_id is None


def test_create_run_passes_project_id_through_to_the_graph(monkeypatch):
    import app as app_module
    import httpx as _httpx

    captured = {}

    async def spy(*a, **kw):
        captured.update(kw)
        return None

    monkeypatch.setattr(app_module, "run_agent_graph", spy)

    async def go():
        transport = _httpx.ASGITransport(app=app_module.app)
        async with _httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/runs", json={
                "user_message": "hi", "snapshot": {}, "project_id": "proj-42",
            })

    resp = asyncio.run(go())
    assert resp.status_code == 200
    assert captured["project_id"] == "proj-42"


# ─── shared fakes ───────────────────────────────────────────────────────

async def _empty_list(*a, **kw):
    return []


async def _noop_bool(*a, **kw):
    return True
