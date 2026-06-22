from __future__ import annotations

from typing import Any

from data_agent_backend.mcp.deps import context_from, get_services, result_wrap
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices


def policy_evaluate(
    action: str,
    resource: str,
    payload: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.policy_engine.evaluate(action, resource, payload or {}, context_from(context, "policy_evaluate")))

