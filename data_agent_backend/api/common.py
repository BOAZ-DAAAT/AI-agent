from __future__ import annotations

from typing import Any

from data_agent_backend.models.common import BackendModel
from data_agent_backend.models.tool_results import ToolResult


ContextPayload = dict[str, Any] | None


class EmptyRequest(BackendModel):
    context: ContextPayload = None


def dump_result(result: ToolResult) -> dict[str, Any]:
    return result.model_dump(mode="json")

