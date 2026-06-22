from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data_agent_backend.models.common import to_jsonable
from data_agent_backend.models.contexts import PolicyContext
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices, create_backend_services

_SERVICES: BackendServices | None = None


def get_services() -> BackendServices:
    global _SERVICES
    if _SERVICES is None:
        _SERVICES = create_backend_services()
    return _SERVICES


def set_services(services: BackendServices | None) -> None:
    global _SERVICES
    _SERVICES = services


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

