from __future__ import annotations

from typing import Any

from data_agent_backend.mcp.tools_approvals import approval_get, approval_list_pending, approval_resolve
from data_agent_backend.mcp.tools_artifacts import artifact_get, artifact_lineage, artifact_list, artifact_preview, artifact_register
from data_agent_backend.mcp.tools_catalog import catalog_get, catalog_list
from data_agent_backend.mcp.tools_execution import sandbox_run_python, sql_run_query
from data_agent_backend.mcp.tools_exports import export_create
from data_agent_backend.mcp.tools_memory import memory_get, memory_list, memory_propose, memory_search
from data_agent_backend.mcp.tools_policy import policy_evaluate
from data_agent_backend.mcp.tools_runs import run_append_event, run_create, run_get, run_list, run_list_events, run_summary, run_update_status
from data_agent_backend.mcp.tools_workspace import workspace_list, workspace_preview, workspace_read_text, workspace_write_text


def create_mcp_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - mcp가 설치되지 않았을 때만 실행됨
        raise RuntimeError("The mcp package is required to run the MCP server.") from exc

    server = FastMCP("data-agent-backend")
    for fn in [
        workspace_list,
        workspace_read_text,
        workspace_write_text,
        workspace_preview,
        run_create,
        run_get,
        run_list,
        run_update_status,
        run_append_event,
        run_list_events,
        run_summary,
        artifact_register,
        artifact_get,
        artifact_list,
        artifact_preview,
        artifact_lineage,
        memory_propose,
        memory_list,
        memory_get,
        memory_search,
        approval_list_pending,
        approval_get,
        approval_resolve,
        policy_evaluate,
        sql_run_query,
        sandbox_run_python,
        catalog_list,
        catalog_get,
        export_create,
    ]:
        server.tool()(fn)
    return server


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
