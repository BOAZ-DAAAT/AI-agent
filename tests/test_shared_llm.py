from __future__ import annotations

import DATA_Analyst_Assistant_Agent.shared.llm as llm_module
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model, get_model_name


def test_get_model_name_prefers_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CHART_READER_MODEL", "openai/gpt-5")

    assert get_model_name("CHART_READER_MODEL", "fallback-model") == "openai/gpt-5"


def test_get_model_name_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert get_model_name("LLM_MODEL", "fallback-model") == "fallback-model"


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
