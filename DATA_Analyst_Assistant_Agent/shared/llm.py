from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore[assignment]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "~openai/gpt-5"


def get_model_name(env_name: str = "LLM_MODEL", default: str = DEFAULT_LLM_MODEL) -> str:
    return os.getenv(env_name) or default


def get_chat_model(
    *,
    model: str | None = None,
    model_env: str = "LLM_MODEL",
    default_model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0,
    **kwargs: Any,
) -> Any:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    api_key = openrouter_key or openai_key
    if not api_key:
        if google_key:
            return ChatGoogleGenerativeAI(
                model=model or get_model_name(model_env, default_model),
                temperature=temperature,
                google_api_key=google_key,
                **kwargs,
            )
        raise ValueError("OPENROUTER_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY is required for LLM calls.")

    base_url = os.getenv("OPENROUTER_BASE_URL")
    if openrouter_key:
        base_url = base_url or OPENROUTER_BASE_URL

    headers = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "")
    if os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-OpenRouter-Title"] = os.getenv("OPENROUTER_APP_TITLE", "")

    params: dict[str, Any] = {
        "model": model or get_model_name(model_env, default_model),
        "temperature": temperature,
        "api_key": api_key,
        **kwargs,
    }
    if base_url:
        params["base_url"] = base_url
    if headers:
        params["default_headers"] = headers
    if ChatOpenAI is None:
        raise ModuleNotFoundError(
            "langchain_openai 패키지가 없어 OpenAI/OpenRouter 모델을 초기화할 수 없습니다. "
            "GOOGLE_API_KEY를 사용하거나 langchain_openai를 설치하세요."
        )
    return ChatOpenAI(**params)
