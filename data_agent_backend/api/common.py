from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data_agent_backend.models.common import BackendModel
from data_agent_backend.models.common import to_jsonable
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.tool_results import ToolResult


ContextPayload = dict[str, Any] | None


class EmptyRequest(BackendModel):
    context: ContextPayload = None


def dump_result(result: ToolResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def context_from(value: dict[str, Any] | PolicyContext | None, tool_name: str | None = None) -> PolicyContext:
    if isinstance(value, PolicyContext):
        context = value
    elif value is None:
        context = PolicyContext()
    else:
        context = PolicyContext(**value)
    return context.with_tool(tool_name) if tool_name else context


def result_wrap(fn: Callable[[], Any]) -> ToolResult:
    try:
        return ToolResult.success(to_jsonable(fn()))
    except Exception as exc:
        return ToolResult.from_exception(exc)

