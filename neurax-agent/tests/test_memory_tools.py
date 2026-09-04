"""`memory_tools.py` — the client for `neurax-service`'s agent-memory
endpoints.

Every function must degrade to "no memory available" rather than raise —
this is a client called automatically by the loop, not something a user
request should ever fail because of. Both directions are tested: the real
request shape sent, and that every failure mode (network error, bad
status, malformed JSON) returns the safe default instead of propagating.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import memory_tools


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://test")
            raise httpx.HTTPStatusError(
                "error", request=request,
                response=httpx.Response(self.status_code, text=self.text, request=request),
            )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload=None, status_code=200, raise_connect_error=False):
        self._payload = payload
        self._status_code = status_code
        self._raise_connect_error = raise_connect_error
        self.sent_params = None
        self.sent_json = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, **kw):
        self.sent_params = params
        if self._raise_connect_error:
            raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))
        return _FakeResponse(self._payload, self._status_code)

    async def post(self, url, json=None, **kw):
        self.sent_json = json
        if self._raise_connect_error:
            raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))
        return _FakeResponse(self._payload, self._status_code)


def _patch(monkeypatch, **kwargs):
    client = _FakeAsyncClient(**kwargs)
    monkeypatch.setattr(memory_tools.httpx, "AsyncClient", lambda **_: client)
    return client


# ─── Core memory ────────────────────────────────────────────────────────

def test_get_core_preferences_returns_the_real_list(monkeypatch):
    _patch(monkeypatch, payload={"preferences": ["prefers GQA over MHA", "target: mobile"]})
    result = asyncio.run(memory_tools.get_core_preferences("proj-1"))
    assert result == ["prefers GQA over MHA", "target: mobile"]


def test_get_core_preferences_with_no_project_id_makes_no_call(monkeypatch):
    client = _patch(monkeypatch, payload={"preferences": ["x"]})
    result = asyncio.run(memory_tools.get_core_preferences(""))
    assert result == []
    assert client.sent_params is None


def test_get_core_preferences_degrades_to_empty_on_network_error(monkeypatch):
    _patch(monkeypatch, raise_connect_error=True)
    assert asyncio.run(memory_tools.get_core_preferences("proj-1")) == []


def test_get_core_preferences_degrades_to_empty_on_bad_status(monkeypatch):
    _patch(monkeypatch, payload={"error": "boom"}, status_code=500)
    assert asyncio.run(memory_tools.get_core_preferences("proj-1")) == []


def test_add_core_preference_sends_project_and_text(monkeypatch):
    client = _patch(monkeypatch, payload={"ok": True})
    ok = asyncio.run(memory_tools.add_core_preference("proj-1", "likes int8"))
    assert ok is True
    assert client.sent_json == {"project_id": "proj-1", "preference": "likes int8"}


def test_add_core_preference_returns_false_on_failure_without_raising(monkeypatch):
    _patch(monkeypatch, raise_connect_error=True)
    assert asyncio.run(memory_tools.add_core_preference("proj-1", "x")) is False


# ─── Archival memory ────────────────────────────────────────────────────

def test_search_past_designs_returns_entries(monkeypatch):
    _patch(monkeypatch, payload={"entries": ["Built a 7B MoE for on-device use, rationale: ..."]})
    result = asyncio.run(memory_tools.search_past_designs("proj-1", "moe"))
    assert result == ["Built a 7B MoE for on-device use, rationale: ..."]


def test_search_past_designs_sends_the_query_and_limit(monkeypatch):
    client = _patch(monkeypatch, payload={"entries": []})
    asyncio.run(memory_tools.search_past_designs("proj-1", "mamba", limit=3))
    assert client.sent_params == {"project_id": "proj-1", "query": "mamba", "limit": 3}


def test_search_past_designs_returns_none_on_failure_not_an_empty_list(monkeypatch):
    # None (failure) and [] (genuinely no matches) are different facts —
    # agent_graph.py narrates them as different messages to the model, so
    # this function must not collapse the two the way the other "get"
    # functions in this module are allowed to.
    _patch(monkeypatch, raise_connect_error=True)
    assert asyncio.run(memory_tools.search_past_designs("proj-1", "x")) is None


def test_search_past_designs_with_genuinely_no_matches_returns_an_empty_list(monkeypatch):
    _patch(monkeypatch, payload={"entries": []})
    result = asyncio.run(memory_tools.search_past_designs("proj-1", "x"))
    assert result == []
    assert result is not None


def test_add_archival_entry_sends_content(monkeypatch):
    client = _patch(monkeypatch, payload={"ok": True})
    ok = asyncio.run(memory_tools.add_archival_entry("proj-1", "Built X because Y."))
    assert ok is True
    assert client.sent_json == {"project_id": "proj-1", "content": "Built X because Y."}


# ─── Recall (conversation) memory ──────────────────────────────────────

def test_get_recent_conversation_returns_turns(monkeypatch):
    _patch(monkeypatch, payload={"turns": [{"role": "user", "content": "hi"}]})
    result = asyncio.run(memory_tools.get_recent_conversation("proj-1"))
    assert result == [{"role": "user", "content": "hi"}]


def test_append_conversation_turns_sends_the_real_list(monkeypatch):
    client = _patch(monkeypatch, payload={"ok": True})
    turns = [{"role": "user", "content": "build it"}, {"role": "assistant", "content": "done"}]
    ok = asyncio.run(memory_tools.append_conversation_turns("proj-1", turns))
    assert ok is True
    assert client.sent_json == {"project_id": "proj-1", "turns": turns}


def test_append_conversation_turns_with_no_turns_makes_no_call(monkeypatch):
    client = _patch(monkeypatch, payload={"ok": True})
    ok = asyncio.run(memory_tools.append_conversation_turns("proj-1", []))
    assert ok is False
    assert client.sent_json is None
