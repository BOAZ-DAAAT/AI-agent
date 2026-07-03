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
    from DATA_Analyst_Assistant_Agent.agents.report.agent import ReportAgent
    from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent

    return {
        "sql_agent": SQLAgent(),
        "eda_agent": EDAAgent(),
        "analysis_agent": AnalysisAgent(),
        "report_agent": ReportAgent(),
    }


class SubAgentAdapter:
    def __init__(
        self,
        *,
        backend_adapter: BackendAdapter,
        agents: dict[str, RunnableAgent] | None = None,
    ) -> None:
        self.backend_adapter = backend_adapter
        self.runtime = AgentRuntime(adapter=backend_adapter)
        self.agents = agents or default_agents()

    def call(self, agent_name: AgentName, state: SupervisorState) -> AgentToolResult:
        if agent_name not in self.agents:
            raise ValueError(f"Unknown supervisor sub-agent: {agent_name}")

        orchestration_state = to_orchestration_state(state)
        envelope = self.agents[agent_name].run(orchestration_state, self.runtime)
        return AgentToolResult(
            agent_result=self._compact_envelope(envelope),
            state_updates={
                "generated_sql": orchestration_state.generated_sql,
                "error_state": orchestration_state.error_state,
            },
        )

    def _compact_envelope(self, envelope: AgentEnvelope) -> AgentCompactResult:
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

        return AgentCompactResult(
            agent=envelope.agent_name,
            status=self._status_value(envelope.status),
            summary=envelope.summary,
            artifact_ids=artifact_ids,
            artifacts=[self._artifact_summary(artifact_id) for artifact_id in artifact_ids],
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            retryable=envelope.retry_hint.retryable,
            error=self._error_message(envelope),
        )

    def _artifact_summary(self, artifact_id: str) -> ArtifactSummary:
        try:
            artifact = self.backend_adapter.get_artifact(artifact_id)
        except Exception:
            return ArtifactSummary(artifact_id=artifact_id)

        metadata = getattr(artifact, "metadata", None) or {}
        preview = getattr(artifact, "preview", None) or {}
        return ArtifactSummary(
            artifact_id=artifact_id,
            type=str(getattr(artifact, "type", "") or ""),
            kind=str(metadata.get("kind", "")),
            summary=self._preview_summary(preview),
            uri=getattr(artifact, "uri", None),
        )

    @staticmethod
    def _status_value(status: AgentStatus | str) -> str:
        return status.value if isinstance(status, AgentStatus) else str(status)

    @staticmethod
    def _error_message(envelope: AgentEnvelope) -> str:
        if envelope.status != AgentStatus.failed:
            return ""
        if envelope.retry_hint.reason_code and envelope.retry_hint.reason_code != "none":
            return f"{envelope.summary} ({envelope.retry_hint.reason_code})"
        return envelope.summary

    @staticmethod
    def _preview_summary(preview: Any) -> str:
        if not isinstance(preview, dict):
            return ""
        parts = [f"{key}={value}" for key, value in list(preview.items())[:3]]
        return ", ".join(parts)
