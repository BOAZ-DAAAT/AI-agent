from __future__ import annotations

from typing import Any

from data_agent_backend.mcp.deps import context_from, get_services, result_wrap
from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType
from data_agent_backend.models.execution import ExecutionLimits
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices


def sql_run_query(
    query: str,
    run_id: str,
    connection_id: str | None = None,
    row_limit: int = 1000,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    return result_wrap(lambda: services.sql_executor.run_sql_query(query, run_id, connection_id, context_from(context, "sql_run_query"), row_limit))


def sandbox_run_python(
    code: str,
    run_id: str,
    input_artifact_ids: list[str] | None = None,
    context: dict[str, Any] | None = None,
    services: BackendServices | None = None,
) -> ToolResult:
    services = services or get_services()
    ctx = context_from(context, "sandbox_run_python").model_copy(update={"run_id": context_from(context).run_id or run_id})
    inputs = [ArtifactRef(artifact_id=item, type=ArtifactType.dataset) for item in (input_artifact_ids or [])]
    return result_wrap(lambda: services.sandbox_executor.run_python(code, inputs, ExecutionLimits(), ctx))

