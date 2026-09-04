"""`web_search_tools.py` — research mode's BYOK Tavily tool.

No real network call in any test here — Tavily is mocked via a faked
`httpx.AsyncClient`, matching `test_analysis_tools.py`'s style for the
standalone HTTP tools. What's actually being checked: the tainting label is
present on every code path (success, empty, and every error), never only
the happy path — a search result is untrusted content regardless of how it
arrived.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import web_search_tools


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test")
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        self.sent = json
        if self._raise_connect_error:
            raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))
        return _FakeResponse(self._payload, self._status_code)


def _patch_client(monkeypatch, **kwargs):
    client = _FakeAsyncClient(**kwargs)
    monkeypatch.setattr(web_search_tools.httpx, "AsyncClient", lambda **_: client)
    return client


def test_no_api_key_returns_a_clear_message_without_any_http_call(monkeypatch):
    called = {"n": 0}

    class ExplodingClient:
        async def __aenter__(self):
            called["n"] += 1
            raise AssertionError("must not construct an HTTP client with no api_key")

    monkeypatch.setattr(web_search_tools.httpx, "AsyncClient", lambda **_: ExplodingClient())

    text = asyncio.run(web_search_tools.web_search("mamba architecture", api_key=""))
    assert called["n"] == 0
    assert "no search API key" in text
    assert web_search_tools.UNTRUSTED_LABEL in text


def test_a_successful_search_is_labeled_and_lists_every_result(monkeypatch):
    _patch_client(monkeypatch, payload={
        "results": [
            {"title": "Mamba: Linear-Time Sequence Modeling", "url": "https://arxiv.org/abs/2312.00752", "content": "A state space model..."},
            {"title": "State Space Models Explained", "url": "https://example.com/ssm", "content": "An overview..."},
        ]
    })
    text = asyncio.run(web_search_tools.web_search("mamba architecture", api_key="tvly-real-key"))
    assert web_search_tools.UNTRUSTED_LABEL in text
    assert "Mamba: Linear-Time Sequence Modeling" in text
    assert "https://arxiv.org/abs/2312.00752" in text
    assert "State Space Models Explained" in text


def test_the_real_key_and_query_are_actually_sent(monkeypatch):
    client = _patch_client(monkeypatch, payload={"results": []})
    asyncio.run(web_search_tools.web_search("jamba hybrid attention", api_key="tvly-secret", max_results=3))
    assert client.sent["api_key"] == "tvly-secret"
    assert client.sent["query"] == "jamba hybrid attention"
    assert client.sent["max_results"] == 3


def test_max_results_is_clamped_to_a_sane_range(monkeypatch):
    client = _patch_client(monkeypatch, payload={"results": []})
    asyncio.run(web_search_tools.web_search("x", api_key="k", max_results=999))
    assert client.sent["max_results"] == 10

    asyncio.run(web_search_tools.web_search("x", api_key="k", max_results=0))
    assert client.sent["max_results"] == 1


def test_no_results_still_carries_the_untrusted_label():
    text = web_search_tools._format_results("obscure query", [])
    assert web_search_tools.UNTRUSTED_LABEL in text
    assert "no results" in text.lower()


def test_a_long_snippet_is_truncated():
    long_content = "x" * 1000
    text = web_search_tools._format_results("q", [{"title": "T", "url": "u", "content": long_content}])
    assert "..." in text
    assert len(text) < 1000 + 200  # not the full 1000 chars verbatim


def test_a_connection_error_is_reported_but_still_labeled(monkeypatch):
    _patch_client(monkeypatch, raise_connect_error=True)
    text = asyncio.run(web_search_tools.web_search("q", api_key="k"))
    assert web_search_tools.UNTRUSTED_LABEL in text
    assert "could not reach" in text.lower()


def test_a_backend_error_status_is_reported_but_still_labeled(monkeypatch):
    _patch_client(monkeypatch, payload={"detail": "invalid key"}, status_code=401)
    text = asyncio.run(web_search_tools.web_search("q", api_key="bad-key"))
    assert web_search_tools.UNTRUSTED_LABEL in text
    assert "401" in text
