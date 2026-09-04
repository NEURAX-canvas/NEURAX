"""`web_search` in `agent_graph.py`'s loop — the dynamic half of its grant.

Unlike every other tool, `web_search` is never in `MODE_TOOL_GRANTS["research"]`
itself; `run_agent_graph` adds it only for research mode *and* only when the
caller's credentials actually carry a `search_api_key` (confirmed BYOK — see
the plan document). These tests exercise exactly that resolution, plus the
structural gate that follows from it, mirroring `test_agent_graph_modes.py`'s
style for the other tools' gating.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_graph
import web_search_tools


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _run_one_step(monkeypatch, snapshot, tool, mode, credentials=None, **kwargs):
    calls = {"n": 0}

    async def fake_once(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"assistant": "step", "tool": tool}
        return {"assistant": "done", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_once)

    async def go():
        q: asyncio.Queue = asyncio.Queue()
        await agent_graph.run_agent_graph(
            "test-run", q, "research it", snapshot, mode=mode, credentials=credentials, **kwargs
        )
        return _drain(q)

    return asyncio.run(go())


def test_web_search_is_not_granted_in_research_mode_without_a_key(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr(web_search_tools, "web_search", spy)

    events = _run_one_step(
        monkeypatch, {"nodes": [], "connections": []},
        {"name": "web_search", "args": {"query": "mamba"}},
        mode="research", credentials=None,
    )

    assert called["n"] == 0
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("not available" in e["data"]["content"] for e in result_events)


def test_web_search_is_granted_in_research_mode_with_a_key(monkeypatch):
    captured = {}

    async def spy(query, api_key, **kw):
        captured["query"] = query
        captured["api_key"] = api_key
        return f"{web_search_tools.UNTRUSTED_LABEL}: fake result for {query}"

    monkeypatch.setattr(web_search_tools, "web_search", spy)

    events = _run_one_step(
        monkeypatch, {"nodes": [], "connections": []},
        {"name": "web_search", "args": {"query": "jamba hybrid architecture"}},
        mode="research", credentials={"api_key": "llm-key", "search_api_key": "tvly-key"},
    )

    assert captured["query"] == "jamba hybrid architecture"
    assert captured["api_key"] == "tvly-key"
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("fake result for jamba hybrid architecture" in e["data"]["content"] for e in result_events)


def test_web_search_is_never_granted_outside_research_mode_even_with_a_key(monkeypatch):
    called = {"n": 0}

    async def spy(*a, **kw):
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr(web_search_tools, "web_search", spy)

    events = _run_one_step(
        monkeypatch, {"nodes": [], "connections": []},
        {"name": "web_search", "args": {"query": "mamba"}},
        mode="creation", credentials={"api_key": "llm-key", "search_api_key": "tvly-key"},
    )

    assert called["n"] == 0
    result_events = [e for e in events if e["event"] == "tool_result"]
    assert any("not available in 'creation' mode" in e["data"]["content"] for e in result_events)


def test_web_search_never_touches_the_snapshot(monkeypatch):
    async def spy(*a, **kw):
        return "some result"

    monkeypatch.setattr(web_search_tools, "web_search", spy)

    original_nodes = [{"id": "a", "type": "conv2d"}]
    snapshot = {"nodes": list(original_nodes), "connections": []}
    _run_one_step(
        monkeypatch, snapshot,
        {"name": "web_search", "args": {"query": "q"}},
        mode="research", credentials={"search_api_key": "tvly-key"},
    )
    assert snapshot["nodes"] == original_nodes


def test_app_credentials_accept_an_optional_search_api_key():
    import app
    creds = app.LlmCredentials(api_key="k", search_api_key="tvly-key")
    assert creds.search_api_key == "tvly-key"
    creds_without = app.LlmCredentials(api_key="k")
    assert creds_without.search_api_key is None
