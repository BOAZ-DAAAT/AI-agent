from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AnalysisPlan,
    OrchestrationState,
    SupervisorTerminalState,
)


AgentName = Literal["sql_agent", "eda_agent", "analysis_agent", "report_agent"]
AgentStatusValue = Literal["success", "warning", "failed", "approval_required"]
NextAction = Literal[
    "clarify",
    "create_plan",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "call_report_agent",
    "finalize",
    "fail",
]


class ArtifactSummary(BaseModel):
    artifact_id: str
    type: str = ""
    kind: str = ""
    summary: str = ""
    uri: str | None = None


class AgentCompactResult(BaseModel):
    agent: AgentName
    status: AgentStatusValue
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    retryable: bool = False
    error: str = ""


class StepSummary(BaseModel):
    step: str
    agent: AgentName | None = None
    action: str
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    next_action: str = ""


class PendingApproval(BaseModel):
    approval_id: str
    agent: AgentName
    reason: str
    approval_type: str


class SupervisorState(TypedDict, total=False):
    thread_id: str
    current_run_id: str
    run_ids: list[str]
    project_id: str | None
    datasource_id: str | None
    catalog_summary: dict[str, Any] | None
    latest_user_query: str
    user_turns: list[dict[str, str]]
    clarified_query: str
    needs_clarification: bool
    clarification_question: str
    analysis_plan: dict[str, Any]
    current_step: str
    next_action: NextAction
    terminal_state: str
    final_answer: str
    agent_results: list[dict[str, Any]]
    artifacts: dict[str, list[dict[str, Any]]]
    validation_results: list[dict[str, Any]]
    step_summaries: list[dict[str, Any]]
    completed_agents: list[str]
    failed_agents: list[str]
    pending_approval: dict[str, Any] | None
    generated_sql: str
    error_state: dict[str, Any]
    retry_counts: dict[str, int]
    max_retry_per_agent: int


def empty_supervisor_state(
    *,
    thread_id: str,
    run_id: str,
    user_query: str,
    datasource_id: str | None,
    project_id: str | None = None,
    catalog_summary: dict[str, Any] | None = None,
) -> SupervisorState:
    return {
        "thread_id": thread_id,
        "current_run_id": run_id,
        "run_ids": [run_id],
        "project_id": project_id,
        "datasource_id": datasource_id,
        "catalog_summary": catalog_summary,
        "latest_user_query": user_query,
        "user_turns": [{"run_id": run_id, "query": user_query}],
        "clarified_query": "",
        "needs_clarification": False,
        "clarification_question": "",
        "analysis_plan": {},
        "current_step": "created",
        "next_action": "create_plan",
        "terminal_state": "running",
        "final_answer": "",
        "agent_results": [],
        "artifacts": {},
        "validation_results": [],
        "step_summaries": [],
        "completed_agents": [],
        "failed_agents": [],
        "pending_approval": None,
        "generated_sql": "",
        "error_state": {},
        "retry_counts": {},
        "max_retry_per_agent": 1,
    }


def merge_agent_result(state: SupervisorState, result: AgentCompactResult) -> SupervisorState:
    agent_results = list(state.get("agent_results", []))
    agent_results.append(result.model_dump(mode="json"))

    artifacts = {agent: list(items) for agent, items in state.get("artifacts", {}).items()}
    artifacts.setdefault(result.agent, [])
    artifacts[result.agent].extend(artifact.model_dump(mode="json") for artifact in result.artifacts)

    completed_agents = list(state.get("completed_agents", []))
    failed_agents = list(state.get("failed_agents", []))
    if result.status in {"success", "warning", "approval_required"} and result.agent not in completed_agents:
        completed_agents.append(result.agent)
    if result.status == "failed" and result.agent not in failed_agents:
        failed_agents.append(result.agent)

    merged: SupervisorState = {
        **state,
        "agent_results": agent_results,
        "artifacts": artifacts,
        "completed_agents": completed_agents,
        "failed_agents": failed_agents,
    }
    if result.error:
        merged["error_state"] = {
            "agent": result.agent,
            "message": result.error,
            "retryable": result.retryable,
        }
    return merged


def artifact_ids_by_agent(state: SupervisorState) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    for result in state.get("agent_results", []):
        agent = str(result.get("agent", ""))
        if not agent:
            continue
        ids.setdefault(agent, [])
        for artifact_id in result.get("artifact_ids", []):
            artifact_id_value = str(artifact_id)
            if artifact_id_value not in ids[agent]:
                ids[agent].append(artifact_id_value)
    return ids


def to_orchestration_state(state: SupervisorState) -> OrchestrationState:
    terminal_value = state.get("terminal_state")
    terminal_state = None
    if terminal_value and terminal_value != "running":
        terminal_state = SupervisorTerminalState(str(terminal_value))

    plan_payload = state.get("analysis_plan") or {}
    planner_mode = "llm" if plan_payload.get("planner_mode") == "llm" else "deterministic"
    generated_sql = state.get("generated_sql") or "SELECT 1 AS sample_value"
    plan = AnalysisPlan(
        goal=str(plan_payload.get("goal") or state.get("latest_user_query") or ""),
        datasource_id=state.get("datasource_id"),
        catalog_summary=state.get("catalog_summary"),
        retry_context=state.get("retry_counts"),
        planner_mode=planner_mode,
        route_kind=str(plan_payload.get("route_kind") or "comprehensive"),
        generated_sql=generated_sql,
        source_sql=generated_sql,
    )

    return OrchestrationState(
        run_id=state["current_run_id"],
        thread_id=state.get("thread_id"),
        datasource_id=state.get("datasource_id"),
        catalog_summary=state.get("catalog_summary"),
        user_query=state.get("clarified_query") or state.get("latest_user_query", ""),
        goal=plan.goal,
        plan=plan,
        retry_context=state.get("retry_counts"),
        current_step=state.get("current_step", "created"),
        artifact_ids=artifact_ids_by_agent(state),
        error_state=dict(state.get("error_state", {})),
        terminal_state=terminal_state,
        remaining_agents=[],
        completed_agents=list(state.get("completed_agents", [])),
        last_agent=(state.get("completed_agents") or [None])[-1],
        route_kind=plan.route_kind,
        planner_mode=plan.planner_mode,
        generated_sql=state.get("generated_sql", ""),
        retry_counts=dict(state.get("retry_counts", {})),
        max_retry_per_agent=int(state.get("max_retry_per_agent", 1)),
    )
