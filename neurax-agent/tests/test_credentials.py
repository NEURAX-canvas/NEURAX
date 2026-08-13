"""Each caller's own key must be the one that is used.

The studio collects an API key per user and, until this was wired, never sent
it: every run was billed to whatever key the agent had in its environment. A
bring-your-own-key product that silently uses the operator's key is both a
false promise and an unbounded bill on a public deployment.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import langchain_runner


def test_caller_key_is_preferred_over_the_server_key(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'server-key')
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr('langchain_openai.ChatOpenAI', FakeChat)
    langchain_runner.make_chat_model(credentials={'api_key': 'caller-key', 'provider': 'openai'})
    assert captured['openai_api_key'] == 'caller-key'


def test_server_key_is_used_when_the_caller_supplies_none(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'server-key')
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr('langchain_openai.ChatOpenAI', FakeChat)
    langchain_runner.make_chat_model()
    assert captured['openai_api_key'] == 'server-key'


def test_a_caller_key_never_leaks_into_the_process_environment(monkeypatch):
    # It would outlive the request and serve whoever calls next.
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('LLM_API_KEY', raising=False)

    class FakeChat:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr('langchain_openai.ChatOpenAI', FakeChat)
    langchain_runner.make_chat_model(credentials={'api_key': 'secret-caller-key'})
    assert os.environ.get('OPENAI_API_KEY') != 'secret-caller-key'


def test_the_caller_may_also_choose_the_model_and_endpoint(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr('langchain_openai.ChatOpenAI', FakeChat)
    langchain_runner.make_chat_model(credentials={
        'api_key': 'k', 'provider': 'openai',
        'model': 'gpt-4o', 'base_url': 'https://example.invalid/v1',
    })
    assert captured['model'] == 'gpt-4o'
    assert 'example.invalid' in str(captured['base_url'])


def test_the_run_request_carries_credentials():
    import app
    fields = app.RunRequest.model_fields
    assert 'credentials' in fields, 'the agent must accept the caller key'
    # Optional, so a deployment with its own key keeps working.
    assert not fields['credentials'].is_required()
