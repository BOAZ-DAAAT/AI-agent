from __future__ import annotations

from typing import Any

from data_agent_backend.mcp.deps import context_from, get_services, result_wrap
from data_agent_backend.models.approvals import ApprovalDecision
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices


def approval_list_pending(context: dict[str, Any] | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.approval_store.list_pending_approvals(context_from(context, "approval_list_pending")))


def approval_get(approval_id: str, context: dict[str, Any] | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.approval_store.get_approval_request(approval_id))


def approval_resolve(
    approval_id: str,
    decision: str,
    edited_payload: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    ctx = context_from(context, "approval_resolve")
    return result_wrap(lambda: services.approval_store.resolve_approval_request(approval_id, ApprovalDecision(decision), edited_payload, ctx.user_id))

