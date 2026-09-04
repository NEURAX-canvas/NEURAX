"""Phase 6 — `POST /ask`, the always-available explain/consult path.

Not a second orchestration mechanism (the plan document's own words for
why this shouldn't exist as one): this drives the exact same
`run_agent_graph` every other entry point uses, in explanation mode, with
`enable_plan=False` and a small step ceiling — collected synchronously
instead of streamed, since a one-off question has no chat drawer polling
`/runs/{id}/events` for it.

`run_controller_step` is monkeypatched throughout — no real LLM call, same
style every other agent_graph-level test in this suite already uses.
`plan_run_strategy` needs no monkeypatching here: `enable_plan=False`
means it is never even called (verified directly), which is itself part
of what this file checks.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import agent_graph


def _client(app_module):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_module.app),
        base_url="http://testserver",
    )


def test_ask_never_calls_plan_run_strategy(monkeypatch):
    called = {"n": 0}

    async def spy(**kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(agent_graph, "plan_run_strategy", spy)

    async def fake_step(**kw):
        return {"assistant": "It's a normalization layer.", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    import app as app_module

    async def go():
        async with _client(app_module) as client:
            return await client.post("/ask", json={
                "question": "what does layernorm do?",
                "snapshot": {"nodes": [], "connections": []},
            })

    resp = asyncio.run(go())
    assert resp.status_code == 200
    assert called["n"] == 0, "enable_plan=False must skip plan_run_strategy entirely"


def test_ask_returns_the_models_narration_as_the_answer(monkeypatch):
    async def fake_step(**kw):
        return {"assistant": "LayerNorm normalizes across the feature dimension.", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    import app as app_module

    async def go():
        async with _client(app_module) as client:
            return await client.post("/ask", json={
                "question": "what does layernorm do?",
                "snapshot": {"nodes": [], "connections": []},
            })

    resp = asyncio.run(go())
    assert resp.status_code == 200
    body = resp.json()
    assert "LayerNorm normalizes across the feature dimension." in body["answer"]


def test_ask_collects_tool_results_from_a_multi_step_answer(monkeypatch):
    calls = {"n": 0}

    async def fake_step(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"assistant": "Let me check the catalogue.", "tool": {"name": "explain_layer_type", "args": {"layer_type": "rmsnorm"}}}
        return {"assistant": "It's RMSNorm, a normalization layer.", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    import app as app_module

    async def go():
        async with _client(app_module) as client:
            return await client.post("/ask", json={
                "question": "what is rmsnorm?",
                "snapshot": {
                    "nodes": [], "connections": [],
                    "catalogue": [{"type": "rmsnorm", "description": "Root mean square normalization"}],
                },
            })

    resp = asyncio.run(go())
    body = resp.json()
    assert any("Root mean square normalization" in tr["content"] for tr in body["tool_results"])
    assert "It's RMSNorm" in body["answer"]


def test_ask_runs_in_explanation_mode_no_canvas_mutation_even_if_asked(monkeypatch):
    async def fake_step(**kw):
        # A misbehaving/hallucinating model tries to build something anyway.
        return {"assistant": "adding a layer", "tool": {"name": "add_node", "args": {"layer_type": "conv2d", "node_id": "n", "x": 0, "y": 0}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    import app as app_module

    async def go():
        async with _client(app_module) as client:
            return await client.post("/ask", json={
                "question": "add a conv layer",
                "snapshot": {"nodes": [], "connections": []},
            })

    resp = asyncio.run(go())
    body = resp.json()
    assert any("not available in 'explanation' mode" in tr["content"] for tr in body["tool_results"])


def test_ask_respects_its_own_small_step_ceiling(monkeypatch):
    async def fake_step(**kw):
        # Never calls done — would run forever without a ceiling.
        return {"assistant": "still looking", "tool": {"name": "explain_layer_type", "args": {"layer_type": "x"}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    import app as app_module
    monkeypatch.setattr(app_module, "_ASK_MAX_STEPS", 3)

    async def go():
        async with _client(app_module) as client:
            return await client.post("/ask", json={
                "question": "keep going forever",
                "snapshot": {"nodes": [], "connections": []},
            })

    resp = asyncio.run(go())
    assert resp.status_code == 200  # returns an answer, doesn't hang or error
    body = resp.json()
    assert len(body["tool_results"]) == 3


def test_ask_is_rate_limited_like_runs(monkeypatch):
    import app as app_module
    import config

    async def fake_step(**kw):
        return {"assistant": "ok", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 1)

    async def go():
        async with _client(app_module) as client:
            first = await client.post("/ask", json={"question": "q1", "snapshot": {"nodes": [], "connections": []}})
            second = await client.post("/ask", json={"question": "q2", "snapshot": {"nodes": [], "connections": []}})
            return first, second

    first, second = asyncio.run(go())
    assert first.status_code == 200
    assert second.status_code == 429


def test_ask_with_no_answer_produced_says_so_plainly(monkeypatch):
    async def fake_step(**kw):
        return {"assistant": "", "tool": {"name": "done", "args": {}}}

    monkeypatch.setattr(agent_graph, "run_controller_step", fake_step)

    import app as app_module

    async def go():
        async with _client(app_module) as client:
            return await client.post("/ask", json={
                "question": "hi", "snapshot": {"nodes": [], "connections": []},
            })

    resp = asyncio.run(go())
    body = resp.json()
    assert body["answer"] == "I don't have an answer for that."
