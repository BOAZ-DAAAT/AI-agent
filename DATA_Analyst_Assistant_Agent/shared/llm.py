from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "~openai/gpt-latest"


def get_model_name(env_name: str = "LLM_MODEL", default: str = DEFAULT_OPENROUTER_MODEL) -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_MODEL") or default
    return os.getenv(env_name) or os.getenv("AGENT_MODEL") or default


def get_chat_model(
    *,
    model: str | None = None,
    model_env: str = "LLM_MODEL",
    default_model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = 0,
    **kwargs: Any,
) -> ChatOpenAI:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    api_key = openrouter_key or openai_key
    if not api_key:
        if google_key:
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.5-flash",
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
    return ChatOpenAI(**params)
