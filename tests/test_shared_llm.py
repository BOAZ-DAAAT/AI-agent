from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.llm import get_model_name


def test_get_model_name_prefers_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CHART_READER_MODEL", "openai/gpt-5")

    assert get_model_name("CHART_READER_MODEL", "fallback-model") == "openai/gpt-5"


def test_get_model_name_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert get_model_name("LLM_MODEL", "fallback-model") == "fallback-model"
