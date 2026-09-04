"""`AGENT_MAX_STEPS`/`AGENT_TIMEOUT_SECONDS`/`AGENT_MAX_EXPENSIVE_CALLS` env
vars actually reaching `run_agent_graph`.

`AGENT_MAX_STEPS` already existed in real `.env` files (this repo's own
included) before this — nothing read it, so a run's step ceiling was always
`agent_graph`'s hardcoded default regardless of what an operator set.
Verified here by checking what `app.py` actually passes to
`run_agent_graph`, monkeypatched to a spy, rather than trusting the module
constants alone (those could be read correctly and still not be threaded
through to the call).
"""
import asyncio
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx


def test_app_reads_agent_max_steps_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("AGENT_MAX_EXPENSIVE_CALLS", "1")

    import app as app_module
    importlib.reload(app_module)  # module-level constants are read at import time

    try:
        captured = {}

        async def spy(*a, **kw):
            captured.update(kw)
            return None

        monkeypatch.setattr(app_module, "run_agent_graph", spy)

        body = {"user_message": "hi", "snapshot": {}}

        async def go():
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/runs", json=body)

        resp = asyncio.run(go())
        assert resp.status_code == 200
        assert captured["max_steps"] == 7
        assert captured["timeout_seconds"] == 42.0
        assert captured["max_expensive_calls"] == 1
    finally:
        # Other test modules import `app` too — leave it as it was found,
        # not reloaded with this test's env still baked into its constants.
        for var in ("AGENT_MAX_STEPS", "AGENT_TIMEOUT_SECONDS", "AGENT_MAX_EXPENSIVE_CALLS"):
            monkeypatch.delenv(var, raising=False)
        importlib.reload(app_module)


def test_app_falls_back_to_agent_graphs_own_defaults_when_unset(monkeypatch):
    # This repo's own real .env sets AGENT_MAX_STEPS=20 — `app.py`'s
    # `load_dotenv()` (called at import time, every reload included) would
    # silently repopulate `os.environ` from that file the moment `delenv`
    # cleared it, making "unset" unreachable while that file exists on
    # disk. Patching `load_dotenv` itself to a no-op is what actually
    # isolates this test from that file, not clearing the environment alone.
    for var in ("AGENT_MAX_STEPS", "AGENT_TIMEOUT_SECONDS", "AGENT_MAX_EXPENSIVE_CALLS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)

    import agent_graph
    import app as app_module
    importlib.reload(app_module)

    assert app_module._AGENT_MAX_STEPS == agent_graph.DEFAULT_MAX_STEPS
    assert app_module._AGENT_TIMEOUT_SECONDS == agent_graph.DEFAULT_TIMEOUT_SECONDS
    assert app_module._AGENT_MAX_EXPENSIVE_CALLS == agent_graph.DEFAULT_MAX_EXPENSIVE_CALLS

    monkeypatch.undo()
    importlib.reload(app_module)
