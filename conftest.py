"""Repository-wide pytest safety defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-real-llm",
        action="store_true",
        default=False,
        help="Allow explicitly opt-in live LLM smoke tests.",
    )


def pytest_configure(config) -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    if config.getoption("--run-real-llm"):
        os.environ["RUN_REAL_LLM"] = "1"
        return

    os.environ["RUN_REAL_LLM"] = ""
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "PINECONE_API_KEY",
    ):
        os.environ[name] = ""
