from __future__ import annotations

import os
import re

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import Field

from data_agent_backend.models.common import BackendError, BackendModel


SENSITIVE_MODEL_KWARGS = {"api_key", "apikey", "token", "secret", "password", "credential", "credentials"}


class AgentModelConfig(BackendModel):
    model_name: str
    temperature: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    max_tokens: int | None = None
    extra_params: dict[str, object] = Field(default_factory=dict)


def _agent_env_prefix(agent_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", agent_name).strip("_")
    if not normalized:
        raise BackendError("VALIDATION_ERROR", "agent_name is required.")
    return f"{normalized.upper()}_AGENT"


def _env_value(key: str) -> str | None:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_float(key: str, default: float) -> float:
    value = _env_value(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise BackendError("LLM_CONFIG_INVALID", f"{key} must be a number.", {"env": key}) from exc


def _env_int(key: str, default: int) -> int:
    value = _env_value(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BackendError("LLM_CONFIG_INVALID", f"{key} must be an integer.", {"env": key}) from exc
    if parsed <= 0:
        raise BackendError("LLM_CONFIG_INVALID", f"{key} must be positive.", {"env": key})
    return parsed


def _resolve_model_config(agent_name: str) -> AgentModelConfig:
    prefix = _agent_env_prefix(agent_name)
    model_env_key = f"{prefix}_MODEL"
    model_name = _env_value(model_env_key)
    if model_name is None:
        raise BackendError(
            "LLM_CONFIG_REQUIRED",
            f"{model_env_key} is required.",
            {"agent_name": agent_name, "env": model_env_key},
        )

    temperature_key = f"{prefix}_TEMPERATURE"
    timeout_key = f"{prefix}_TIMEOUT"
    retries_key = f"{prefix}_MAX_RETRIES"
    max_tokens_key = f"{prefix}_MAX_TOKENS"

    max_tokens_value = _env_value(max_tokens_key)
    return AgentModelConfig(
        model_name=model_name,
        temperature=_env_float(temperature_key, 0.0),
        timeout=_env_int(timeout_key, 30),
        max_retries=_env_int(retries_key, 3),
        max_tokens=_env_int(max_tokens_key, 1) if max_tokens_value is not None else None,
    )


def build_chat_model(config: AgentModelConfig | str, **overrides: object):
    model_config = AgentModelConfig(model_name=config) if isinstance(config, str) else config
    sensitive_keys = {str(key).lower() for key in model_config.extra_params}
    sensitive_keys |= {str(key).lower() for key in overrides}
    blocked = sorted(sensitive_keys & SENSITIVE_MODEL_KWARGS)
    if blocked:
        raise BackendError(
            "LLM_SECRET_NOT_ALLOWED",
            "Model credentials must be provided through environment variables, not code parameters.",
            {"keys": blocked},
        )
    params = {
        "temperature": model_config.temperature,
        "timeout": model_config.timeout,
        "max_retries": model_config.max_retries,
        **model_config.extra_params,
        **overrides,
    }
    if model_config.max_tokens is not None:
        params["max_tokens"] = model_config.max_tokens
    return init_chat_model(model_config.model_name, **params)


def build_chat_model_for_agent(agent_name: str, **overrides: object):
    load_dotenv(find_dotenv(usecwd=True), override=False)
    return build_chat_model(_resolve_model_config(agent_name), **overrides)
