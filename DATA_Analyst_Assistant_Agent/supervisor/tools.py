from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AgentStatus, OrchestrationState
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    ArtifactSummary,
    SupervisorState,
    to_orchestration_state,
)


class RunnableAgent(Protocol):
    name: str

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        ...


class AgentToolResult(BaseModel):
    agent_result: AgentCompactResult
    state_updates: dict[str, Any] = Field(default_factory=dict)


def default_agents() -> dict[AgentName, RunnableAgent]:
    from DATA_Analyst_Assistant_Agent.agents.analysis.agent import AnalysisAgent
    from DATA_Analyst_Assistant_Agent.agents.eda.agent import EDAAgent
    from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent

    return {
        "sql_agent": SQLAgent(),
        "eda_agent": EDAAgent(),
        "analysis_agent": AnalysisAgent(),
    }


def compact_agent_envelope(
    envelope: AgentEnvelope,
    backend_adapter: BackendAdapter,
) -> AgentCompactResult:
    artifact_ids = envelope.artifact_ids()
    validation_errors = [
        check.detail
        for check in envelope.validation.local_checks
        if check.severity == "error" and not check.passed
    ]
    validation_warnings = [
        check.detail
        for check in envelope.validation.local_checks
        if check.severity == "warning" and not check.passed
    ]
    validation_errors.extend(
        flag.message for flag in envelope.validation.business_flags if flag.severity == "error"
    )
    validation_warnings.extend(
        flag.message for flag in envelope.validation.business_flags if flag.severity == "warning"
    )

    return AgentCompactResult(
        agent=envelope.agent_name,
        status=_status_value(envelope),
        summary=envelope.summary,
        artifact_ids=artifact_ids,
        artifacts=[_artifact_summary(backend_adapter, artifact_id) for artifact_id in artifact_ids],
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        findings=envelope.validation.normalized_findings(),
        retry_hint=envelope.retry_hint,
        approval=envelope.approval,
        fallback_used=envelope.fallback_used,
        retryable=envelope.retry_hint.retryable,
        error=_error_message(envelope),
    )


def _artifact_summary(backend_adapter: BackendAdapter, artifact_id: str) -> ArtifactSummary:
    try:
        artifact = backend_adapter.get_artifact(artifact_id)
    except Exception:
        return ArtifactSummary(artifact_id=artifact_id)

    metadata = getattr(artifact, "metadata", None) or {}
    preview = getattr(artifact, "preview", None) or {}
    return ArtifactSummary(
        artifact_id=artifact_id,
        type=str(getattr(artifact, "type", "") or ""),
        kind=str(metadata.get("kind", "")),
        summary=_preview_summary(preview),
        uri=getattr(artifact, "uri", None),
        run_id=str(getattr(artifact, "run_id", "") or ""),
        content_hash=getattr(artifact, "content_hash", None),
        parent_ids=list(getattr(artifact, "parent_ids", None) or []),
        metadata=dict(metadata),
        preview=dict(preview) if isinstance(preview, dict) else {},
    )


def _status_value(envelope: AgentEnvelope) -> str:
    if isinstance(envelope.status, AgentStatus):
        return envelope.status.value
    status = str(envelope.status)
    valid_statuses = {item.value for item in AgentStatus}
    if status not in valid_statuses:
        raise ValueError(f"Invalid status from {envelope.agent_name}: {status}")
    return status


def _error_message(envelope: AgentEnvelope) -> str:
    if envelope.status != AgentStatus.failed:
        return ""
    if envelope.retry_hint.reason_code and envelope.retry_hint.reason_code != "none":
        return f"{envelope.summary} ({envelope.retry_hint.reason_code})"
    return envelope.summary


def _preview_summary(preview: Any) -> str:
    if not isinstance(preview, dict):
        return ""
    parts = [
        f"{key}={_truncate_text(str(value), 240)}"
        for key, value in list(preview.items())[:3]
    ]
    return _truncate_text(", ".join(parts), 1000)


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "…"


class SubAgentAdapter:
    def __init__(
        self,
        *,
        backend_adapter: BackendAdapter,
        agents: dict[str, RunnableAgent] | None = None,
    ) -> None:
        self.backend_adapter = backend_adapter
        self.runtime = AgentRuntime(adapter=backend_adapter)
        self.agents = agents if agents is not None else default_agents()

    def call(self, agent_name: AgentName, state: SupervisorState) -> AgentToolResult:
        if agent_name not in self.agents:
            raise ValueError(f"Unknown supervisor sub-agent: {agent_name}")

        orchestration_state = to_orchestration_state(state)
        envelope = self.agents[agent_name].run(orchestration_state, self.runtime)
        return AgentToolResult(
            agent_result=compact_agent_envelope(envelope, self.backend_adapter),
            state_updates=self._state_updates(orchestration_state),
        )

    def _compact_envelope(self, envelope: AgentEnvelope) -> AgentCompactResult:
        return compact_agent_envelope(envelope, self.backend_adapter)

    def _artifact_summary(self, artifact_id: str) -> ArtifactSummary:
        return _artifact_summary(self.backend_adapter, artifact_id)

    @staticmethod
    def _status_value(envelope: AgentEnvelope) -> str:
        return _status_value(envelope)

    @staticmethod
    def _error_message(envelope: AgentEnvelope) -> str:
        return _error_message(envelope)

    @staticmethod
    def _preview_summary(preview: Any) -> str:
        return _preview_summary(preview)

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        return _truncate_text(value, limit)

    @staticmethod
    def _state_updates(state: OrchestrationState) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "planner_mode": state.planner_mode,
            "generated_sql": state.generated_sql,
            "error_state": state.error_state,
        }
        if state.plan is not None:
            updates["analysis_plan"] = {
                "generated_sql": state.plan.generated_sql,
                "source_sql": state.plan.source_sql,
                "planner_mode": state.plan.planner_mode,
                "route_kind": state.plan.route_kind,
                "goal": state.plan.goal,
                "target_table": state.plan.target_table,
            }
        return updates
