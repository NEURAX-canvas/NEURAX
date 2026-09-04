"""Rate limiting in front of `POST /runs`.

Nothing throttled how often a caller can *start* a run anywhere in this
stack before this — `agent_graph.py`'s per-run step/timeout ceilings (and
the old pipeline's `MAX_ATTEMPTS`) bound how long one run can go, not how
many runs a caller can start back to back, each one real, billed LLM and
compiler work.

`config.check_rate_limit` is tested directly (pure function, explicit `now`
— same style `test_run_lifecycle.py` already uses for `_sweep_expired_runs`'s
boundary), then the actual `/runs` endpoint is checked end to end by driving
the ASGI app directly with `httpx.AsyncClient(transport=ASGITransport(...))`
— not `fastapi.testclient.TestClient`, whose bundled Starlette version
doesn't speak to this repo's installed `httpx` (`Client.__init__() got an
unexpected keyword argument 'app'`, a real version mismatch in this
environment, not a hypothetical). `run_agent_graph` is monkeypatched to a no-op
so the test never makes a real LLM call.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import config


# ─── config.check_rate_limit (pure, no HTTP) ───────────────────────────────

def test_the_first_max_runs_within_the_window_are_all_allowed():
    config._run_starts.clear()
    now = 1000.0
    for _ in range(5):
        assert config.check_rate_limit("client-a", now=now) is True


def test_exceeding_the_limit_is_rejected_within_the_same_window(monkeypatch):
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 3)
    now = 1000.0
    assert config.check_rate_limit("client-a", now=now) is True
    assert config.check_rate_limit("client-a", now=now) is True
    assert config.check_rate_limit("client-a", now=now) is True
    assert config.check_rate_limit("client-a", now=now) is False


def test_a_rejected_attempt_does_not_itself_count_toward_the_window(monkeypatch):
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 1)
    now = 1000.0
    assert config.check_rate_limit("client-a", now=now) is True
    # Rejected five times in a row — none of these should ever "use up" a
    # slot that a later, legitimate window could otherwise have granted.
    for _ in range(5):
        assert config.check_rate_limit("client-a", now=now) is False
    assert len(config._run_starts["client-a"]) == 1


def test_the_window_slides_old_attempts_age_out(monkeypatch):
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 2)
    monkeypatch.setattr(config, "_RATE_LIMIT_WINDOW_SECONDS", 60.0)
    assert config.check_rate_limit("client-a", now=0.0) is True
    assert config.check_rate_limit("client-a", now=1.0) is True
    assert config.check_rate_limit("client-a", now=2.0) is False  # still within the window

    # 61s after the first attempt: that first attempt has aged out, so one
    # slot is free again even though the second attempt (at t=1) has not.
    assert config.check_rate_limit("client-a", now=61.5) is True


def test_different_clients_have_independent_windows(monkeypatch):
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 1)
    now = 1000.0
    assert config.check_rate_limit("client-a", now=now) is True
    assert config.check_rate_limit("client-a", now=now) is False
    # A different client's own window is untouched by client-a's usage.
    assert config.check_rate_limit("client-b", now=now) is True


# ─── POST /runs end to end (TestClient, no real LLM call) ─────────────────

def _client(app_module):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_module.app),
        base_url="http://testserver",
    )


def test_post_runs_returns_429_once_the_limit_is_exceeded(monkeypatch):
    import app as app_module

    async def _noop_run_agent(*a, **kw):
        return None

    monkeypatch.setattr(app_module, "run_agent_graph", _noop_run_agent)
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 2)

    body = {"user_message": "build something", "snapshot": {}}

    async def go():
        async with _client(app_module) as client:
            r1 = await client.post("/runs", json=body)
            r2 = await client.post("/runs", json=body)
            r3 = await client.post("/runs", json=body)
            return r1, r2, r3

    r1, r2, r3 = asyncio.run(go())

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "Rate limit" in r3.json()["detail"]


def test_post_runs_rate_limits_by_api_key_not_just_ip(monkeypatch):
    """Two callers behind the same IP (a shared corporate proxy, a NAT) must
    not share one rate-limit bucket if they present different API keys."""
    import app as app_module

    async def _noop_run_agent(*a, **kw):
        return None

    monkeypatch.setattr(app_module, "run_agent_graph", _noop_run_agent)
    config._run_starts.clear()
    monkeypatch.setattr(config, "_RATE_LIMIT_MAX_RUNS", 1)

    body_a = {"user_message": "hi", "snapshot": {}, "credentials": {"api_key": "key-a"}}
    body_b = {"user_message": "hi", "snapshot": {}, "credentials": {"api_key": "key-b"}}

    async def go():
        async with _client(app_module) as client:
            first = await client.post("/runs", json=body_a)
            second = await client.post("/runs", json=body_a)
            # Different key, same connection/"IP" — must still be allowed.
            third = await client.post("/runs", json=body_b)
            return first, second, third

    first, second, third = asyncio.run(go())

    assert first.status_code == 200
    assert second.status_code == 429
    assert third.status_code == 200
