from __future__ import annotations

import inspect
from typing import Any, Protocol

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisSelectionResponse,
    ReviewRequest,
)
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    OrchestrationState,
    RetryHint,
)
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


class AgentContractError(ValueError):
    pass


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
    return envelope.error


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
        chart_reader: Any | None = None,
    ) -> None:
        self.backend_adapter = backend_adapter
        self.runtime = AgentRuntime(adapter=backend_adapter)
        self.agents = agents if agents is not None else default_agents()
        self.chart_reader = chart_reader

    def call(self, agent_name: AgentName, state: SupervisorState) -> AgentToolResult:
        if agent_name not in self.agents:
            raise ValueError(f"Unknown supervisor sub-agent: {agent_name}")

        orchestration_state = to_orchestration_state(state)
        agent = self.agents[agent_name]
        run_kwargs = self._agent_run_kwargs(agent_name, agent, state)
        try:
            envelope = agent.run(
                orchestration_state,
                self.runtime,
                **run_kwargs,
            )
        except Exception as exc:
            error = str(exc)
            retry_hint = RetryHint(
                retryable=True,
                reason_code="agent_execution_exception",
                details={
                    "failure_reason": error,
                    "exception_type": exc.__class__.__name__,
                },
            )
            return AgentToolResult(
                agent_result=AgentCompactResult(
                    agent=agent_name,
                    status="failed",
                    summary=f"{agent_name} 실행 중 예외가 발생했습니다.",
                    retry_hint=retry_hint,
                    retryable=True,
                    error=error,
                ),
                state_updates={},
            )

        if envelope.agent_name != agent_name:
            raise AgentContractError(
                "Sub-agent envelope agent_name이 요청한 agent와 일치하지 않습니다: "
                f"requested={agent_name}, returned={envelope.agent_name}"
            )

        return AgentToolResult(
            agent_result=compact_agent_envelope(envelope, self.backend_adapter),
            state_updates=self._state_updates(orchestration_state),
        )

    def _agent_run_kwargs(
        self,
        agent_name: AgentName,
        agent: RunnableAgent,
        raw_state: SupervisorState,
    ) -> dict[str, Any]:
        if agent_name != "analysis_agent":
            return {}

        signature = inspect.signature(agent.run)
        parameters = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        kwargs: dict[str, Any] = {}
        selection_payload = raw_state.get("analysis_selection_response")
        review_payload = raw_state.get("analysis_selection_review_request")
        if selection_payload is not None and (accepts_kwargs or "selection_response" in parameters):
            kwargs["selection_response"] = AnalysisSelectionResponse.model_validate(selection_payload)
        if review_payload is not None and (accepts_kwargs or "review_request" in parameters):
            kwargs["review_request"] = ReviewRequest.model_validate(review_payload)
        if accepts_kwargs or "chart_artifact_loader" in parameters:
            kwargs["chart_artifact_loader"] = self._chart_artifact_loader
        if self.chart_reader is not None and (accepts_kwargs or "chart_reader" in parameters):
            kwargs["chart_reader"] = self.chart_reader
        return kwargs

    def _chart_artifact_loader(self, artifact_id: str) -> bytes:
        if hasattr(self.backend_adapter, "read_artifact_bytes"):
            return self.backend_adapter.read_artifact_bytes(artifact_id)
        services = getattr(self.backend_adapter, "services", None)
        artifact_store = getattr(services, "artifact_store", None)
        if artifact_store is not None and hasattr(artifact_store, "read_bytes"):
            return artifact_store.read_bytes(artifact_id)
        raise RuntimeError("Backend adapter does not support artifact byte reads.")

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
            plan_updates = {
                "generated_sql": state.plan.generated_sql,
                "source_sql": state.plan.source_sql,
                "planner_mode": state.plan.planner_mode,
                "route_kind": state.plan.route_kind,
                "goal": state.plan.goal,
                "target_table": state.plan.target_table,
                "source_tables": state.plan.source_tables,
                "business_grain": state.plan.business_grain,
                "mart_design": state.plan.mart_design,
                "analysis_data_contract": state.plan.analysis_data_contract,
            }
            if state.plan.sql_generation_source is not None:
                plan_updates["sql_generation_source"] = state.plan.sql_generation_source
            if state.plan.sql_template_id is not None:
                plan_updates["sql_template_id"] = state.plan.sql_template_id.value
            updates["analysis_plan"] = plan_updates
        return updates
