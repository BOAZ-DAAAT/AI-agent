from __future__ import annotations

from typing import Any

from data_agent_backend.mcp.deps import context_from, get_services, result_wrap
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices


def run_create(
    run_id: str | None = None,
    thread_id: str | None = None,
    project_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.run_service.create_run(run_id, thread_id, project_id, metadata, context_from(context, "run_create")))


def run_get(run_id: str, context: dict[str, Any] | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.run_service.get_run(run_id, context_from(context, "run_get")))


def run_list(
    thread_id: str | None = None,
    project_id: str | None = None,
    status: str | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.run_service.list_runs(thread_id, project_id, status, context_from(context, "run_list")))


def run_update_status(
    run_id: str,
    status: str,
    metadata: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.run_service.update_status(run_id, status, metadata, context_from(context, "run_update_status")))


def run_append_event(
    run_id: str,
    event_type: str,
    message: str,
    node_name: str | None = None,
    tool_name: str | None = None,
    artifact_ids: list[str] | None = None,
    approval_id: str | None = None,
    memory_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    return result_wrap(
        lambda: services.run_service.append_event(
            run_id,
            event_type,
            message,
            node_name,
            tool_name,
            artifact_ids,
            approval_id,
            memory_ids,
            metadata,
            context_from(context, "run_append_event"),
        )
    )


def run_list_events(run_id: str, context: dict[str, Any] | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.run_service.list_events(run_id, context_from(context, "run_list_events")))


def run_summary(run_id: str, context: dict[str, Any] | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.run_service.get_summary(run_id, context_from(context, "run_summary")))
