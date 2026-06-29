from __future__ import annotations

from typing import Any


def increase_retry(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "error": "",
        "terminal_reason": "retrying",
    }
