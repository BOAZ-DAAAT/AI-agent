from __future__ import annotations

import DATA_Analyst_Assistant_Agent.shared.llm as llm_module
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model, get_model_name


def test_get_model_name_prefers_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CHART_READER_MODEL", "env-chart-model")

    assert get_model_name("CHART_READER_MODEL", "fallback-model") == "env-chart-model"


def test_get_model_name_falls_back_to_llm_model_for_role_env(monkeypatch) -> None:
    monkeypatch.delenv("CHART_READER_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "env-default-model")

    assert get_model_name("CHART_READER_MODEL") == "env-default-model"


def test_get_model_name_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert get_model_name("LLM_MODEL", "fallback-model") == "fallback-model"


def test_get_model_name_requires_env_without_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)

    try:
        get_model_name("LLM_MODEL")
    except ValueError as exc:
        assert "LLM_MODEL" in str(exc)
    else:
        raise AssertionError("expected missing model env to raise ValueError")


def test_get_chat_model_sets_default_request_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "12.5")

    model = get_chat_model(model="openai/gpt-test")

    assert isinstance(model, FakeChatOpenAI)
    assert captured["request_timeout"] == 12.5


def test_get_chat_model_caps_default_max_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    model = get_chat_model(model="openai/gpt-test")

    assert isinstance(model, FakeChatOpenAI)
    assert captured["max_tokens"] == 12288


def test_get_chat_model_respects_explicit_max_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")

    model = get_chat_model(model="openai/gpt-test", max_tokens=512)

    assert isinstance(model, FakeChatOpenAI)
    assert captured["max_tokens"] == 512
