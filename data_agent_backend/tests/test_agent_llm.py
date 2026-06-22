from __future__ import annotations

import pytest

import data_agent_backend.agent.llm as llm
from data_agent_backend.models.common import BackendError


def _clear_agent_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "REPORT_AGENT_MODEL",
        "REPORT_AGENT_TEMPERATURE",
        "REPORT_AGENT_TIMEOUT",
        "REPORT_AGENT_MAX_RETRIES",
        "SQL_AGENT_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_build_chat_model_uses_agent_specific_env_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_model_env(monkeypatch)
    monkeypatch.setattr(llm, "find_dotenv", lambda usecwd=True: "")
    monkeypatch.setenv("REPORT_AGENT_MODEL", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("REPORT_AGENT_TEMPERATURE", "0.2")
    monkeypatch.setenv("REPORT_AGENT_TIMEOUT", "45")
    monkeypatch.setenv("REPORT_AGENT_MAX_RETRIES", "4")

    received = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        received["model_name"] = model_name
        received["kwargs"] = kwargs
        return {"model": model_name, "kwargs": kwargs}

    monkeypatch.setattr(llm, "init_chat_model", fake_init_chat_model)

    model = llm.build_chat_model_for_agent("report")

    assert model["model"] == "anthropic:claude-sonnet-4-5"
    assert received["kwargs"] == {"temperature": 0.2, "timeout": 45, "max_retries": 4}
    assert "api_key" not in received["kwargs"]


def test_build_chat_model_requires_agent_specific_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_model_env(monkeypatch)
    monkeypatch.setattr(llm, "find_dotenv", lambda usecwd=True: "")

    with pytest.raises(BackendError) as exc_info:
        llm.build_chat_model_for_agent("sql")

    assert exc_info.value.code == "LLM_CONFIG_REQUIRED"
    assert exc_info.value.details["env"] == "SQL_AGENT_MODEL"


def test_build_chat_model_requires_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_agent_model_env(monkeypatch)
    monkeypatch.setattr(llm, "find_dotenv", lambda usecwd=True: "")

    with pytest.raises(BackendError) as exc_info:
        llm.build_chat_model_for_agent("report")

    assert exc_info.value.code == "LLM_CONFIG_REQUIRED"
    assert exc_info.value.details["env"] == "REPORT_AGENT_MODEL"


def test_build_chat_model_rejects_api_key_kwargs() -> None:
    with pytest.raises(BackendError) as exc_info:
        llm.build_chat_model("openai:gpt-4.1-mini", api_key="secret")

    assert exc_info.value.code == "LLM_SECRET_NOT_ALLOWED"
