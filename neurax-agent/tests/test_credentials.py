"""Each caller's own key must be the one that is used.

The studio collects an API key per user and, until this was wired, never sent
it: every run was billed to whatever key the agent had in its environment. A
bring-your-own-key product that silently uses the operator's key is both a
false promise and an unbounded bill on a public deployment.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
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


# ─── Provider reach: Anthropic, and gateways in front of it ────────────

def _fake_anthropic(monkeypatch, captured):
    """Stand in for `langchain_anthropic.ChatAnthropic`."""
    import types

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = types.ModuleType('langchain_anthropic')
    module.ChatAnthropic = FakeChatAnthropic
    monkeypatch.setitem(sys.modules, 'langchain_anthropic', module)


def test_anthropic_is_selected_and_uses_the_callers_key(monkeypatch):
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    langchain_runner.make_chat_model(
        credentials={'provider': 'anthropic', 'api_key': 'caller-anthropic-key'}
    )

    assert captured['anthropic_api_key'] == 'caller-anthropic-key'


def test_the_default_claude_model_is_current(monkeypatch):
    """A stale default silently gives every client an older model.

    The agent defaulted to `claude-3-5-sonnet-20240620` while the studio
    offered `claude-sonnet-4-20250514`, so which model a client got depended on
    which of the two happened to fill the blank.
    """
    captured = {}
    _fake_anthropic(monkeypatch, captured)
    monkeypatch.delenv('LLM_MODEL', raising=False)
    monkeypatch.delenv('LLAMA_MODEL', raising=False)

    langchain_runner.make_chat_model(credentials={'provider': 'anthropic', 'api_key': 'k'})

    assert captured['model'] == langchain_runner.DEFAULT_ANTHROPIC_MODEL
    assert captured['model'] == 'claude-sonnet-5'


def test_an_anthropic_gateway_is_honoured(monkeypatch):
    """A client behind a corporate proxy or LiteLLM must be able to point at it.

    Only the OpenAI path honoured `base_url`, so these callers were accepted
    and then sent to api.anthropic.com regardless.
    """
    captured = {}
    _fake_anthropic(monkeypatch, captured)

    langchain_runner.make_chat_model(
        credentials={
            'provider': 'anthropic',
            'api_key': 'k',
            'base_url': 'https://llm.example.internal/anthropic/',
        }
    )

    assert captured['anthropic_api_url'] == 'https://llm.example.internal/anthropic'


def test_the_official_endpoint_is_not_passed_as_an_override(monkeypatch):
    """Naming the real endpoint should behave exactly like naming none."""
    captured = {}
    _fake_anthropic(monkeypatch, captured)

    langchain_runner.make_chat_model(
        credentials={
            'provider': 'anthropic',
            'api_key': 'k',
            'base_url': 'https://api.anthropic.com/v1',
        }
    )

    assert 'anthropic_api_url' not in captured


def test_a_custom_provider_reaches_its_own_endpoint(monkeypatch):
    """`custom` is any OpenAI-compatible server: vLLM, Ollama, a gateway."""
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import types
    module = types.ModuleType('langchain_openai')
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, 'langchain_openai', module)

    langchain_runner.make_chat_model(
        credentials={
            'provider': 'custom',
            'api_key': 'k',
            'base_url': 'http://192.168.1.50:8000/v1/',
            'model': 'qwen2.5-72b-instruct',
        }
    )

    assert captured['base_url'] == 'http://192.168.1.50:8000/v1'
    assert captured['model'] == 'qwen2.5-72b-instruct'
    assert captured['openai_api_key'] == 'k'


# ─── Google/Gemini: the one other provider with its own client ────────────

def _fake_google(monkeypatch, captured):
    import types

    class FakeChatGoogle:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = types.ModuleType('langchain_google_genai')
    module.ChatGoogleGenerativeAI = FakeChatGoogle
    monkeypatch.setitem(sys.modules, 'langchain_google_genai', module)


def test_google_is_selected_and_uses_the_callers_key(monkeypatch):
    captured = {}
    _fake_google(monkeypatch, captured)
    # A real .env (loaded once, process-wide, whenever anything imports
    # `app`) sets LLM_MODEL for the OpenAI path elsewhere in this test
    # session — cleared here so this test sees the same blank slate it
    # would in a real request with no model named.
    monkeypatch.delenv('LLM_MODEL', raising=False)
    monkeypatch.delenv('LLAMA_MODEL', raising=False)

    langchain_runner.make_chat_model(
        credentials={'provider': 'google', 'api_key': 'caller-google-key'}
    )

    assert captured['google_api_key'] == 'caller-google-key'
    assert captured['model'] == langchain_runner.DEFAULT_GOOGLE_MODEL


def test_google_never_goes_through_the_openai_client(monkeypatch):
    """Gemini's API isn't OpenAI-shaped — sending it there fails every call.

    This is the exact regression this file exists to catch: before Google had
    its own branch, selecting it silently fell through to ChatOpenAI, which
    sent the caller's Gemini key to api.openai.com and failed authentication
    on every single request.
    """
    captured_google = {}
    captured_openai = {}
    _fake_google(monkeypatch, captured_google)

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured_openai.update(kwargs)

    import types
    module = types.ModuleType('langchain_openai')
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, 'langchain_openai', module)

    langchain_runner.make_chat_model(credentials={'provider': 'google', 'api_key': 'k'})

    assert captured_google, 'Google should have been constructed'
    assert not captured_openai, 'the OpenAI client should never have been touched'


# ─── OpenAI-compatible clones: Mistral, Fireworks, DeepSeek, GLM ──────────

def _fake_openai(monkeypatch, captured):
    import types

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = types.ModuleType('langchain_openai')
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, 'langchain_openai', module)


@pytest.mark.parametrize('provider', ['mistral', 'fireworks', 'deepseek', 'glm'])
def test_each_openai_compatible_clone_reaches_its_own_endpoint_by_default(monkeypatch, provider):
    """Each clone must get its own base_url and model, not OpenAI's.

    Before OPENAI_COMPATIBLE_DEFAULTS existed, only 'custom' (which requires
    the caller to supply base_url by hand) worked — selecting 'mistral' with
    no base_url sent the caller's Mistral key straight to api.openai.com.
    """
    captured = {}
    _fake_openai(monkeypatch, captured)
    monkeypatch.delenv('LLM_MODEL', raising=False)
    monkeypatch.delenv('LLAMA_MODEL', raising=False)

    langchain_runner.make_chat_model(
        credentials={'provider': provider, 'api_key': f'caller-{provider}-key'}
    )

    expected = langchain_runner.OPENAI_COMPATIBLE_DEFAULTS[provider]
    assert captured['base_url'] == expected['base_url']
    assert captured['model'] == expected['model']
    assert captured['openai_api_key'] == f'caller-{provider}-key'
    # Never silently defaults to OpenAI's own endpoint.
    assert 'api.openai.com' not in (captured['base_url'] or '')


@pytest.mark.parametrize('provider', ['mistral', 'fireworks', 'deepseek', 'glm'])
def test_an_openai_compatible_clone_honours_an_explicit_gateway(monkeypatch, provider):
    """A caller behind their own proxy must still be able to override it."""
    captured = {}
    _fake_openai(monkeypatch, captured)

    langchain_runner.make_chat_model(
        credentials={
            'provider': provider,
            'api_key': 'k',
            'base_url': 'https://llm.example.internal/v1',
            'model': 'a-custom-model-name',
        }
    )

    assert captured['base_url'] == 'https://llm.example.internal/v1'
    assert captured['model'] == 'a-custom-model-name'


def test_glm_model_name_prefix_is_auto_detected(monkeypatch):
    """Naming a glm- model alone, with no explicit provider, must pick GLM."""
    captured = {}
    _fake_openai(monkeypatch, captured)

    langchain_runner.make_chat_model(credentials={'api_key': 'k', 'model': 'glm-4-plus'})

    assert captured['base_url'] == langchain_runner.OPENAI_COMPATIBLE_DEFAULTS['glm']['base_url']


def test_deepseek_model_name_prefix_is_auto_detected(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, captured)

    langchain_runner.make_chat_model(credentials={'api_key': 'k', 'model': 'deepseek-chat'})

    assert captured['base_url'] == langchain_runner.OPENAI_COMPATIBLE_DEFAULTS['deepseek']['base_url']


def test_fireworks_model_name_prefix_is_auto_detected(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, captured)

    langchain_runner.make_chat_model(
        credentials={'api_key': 'k', 'model': 'accounts/fireworks/models/llama-v3p1-70b-instruct'}
    )

    assert captured['base_url'] == langchain_runner.OPENAI_COMPATIBLE_DEFAULTS['fireworks']['base_url']
