from __future__ import annotations

from typing import Any

from data_agent_backend.mcp.deps import context_from, get_services, result_wrap
from data_agent_backend.models.artifacts import ArtifactRegisterRequest
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices


def artifact_register(payload: dict[str, Any], context: dict[str, Any] | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.artifact_registry.register_artifact(ArtifactRegisterRequest(**payload), context_from(context, "artifact_register")))


def artifact_get(artifact_id: str, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.artifact_registry.get_artifact(artifact_id))


def artifact_list(run_id: str | None = None, type: str | None = None, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.artifact_registry.list_artifacts(run_id=run_id, type=type))


def artifact_preview(artifact_id: str, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.artifact_registry.get_preview(artifact_id))


def artifact_lineage(artifact_id: str, services: BackendServices | None = None) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.artifact_registry.get_lineage(artifact_id))

