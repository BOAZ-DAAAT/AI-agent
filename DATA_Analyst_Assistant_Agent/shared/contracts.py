from __future__ import annotations

from data_agent_backend.models.common import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from data_agent_backend.models.artifacts import ArtifactRef


class AgentStatus(StrEnum):
    success = "success"
    warning = "warning"
    failed = "failed"
    approval_required = "approval_required"


class SupervisorTerminalState(StrEnum):
    completed = "completed"
    needs_user_approval = "needs_user_approval"
    needs_clarification = "needs_clarification"
    failed_with_recoverable_context = "failed_with_recoverable_context"
    failed_terminal = "failed_terminal"


class LocalCheck(BaseModel):
    name: str
    passed: bool
    severity: Literal["info", "warning", "error"] = "info"
    detail: str = ""


class BusinessFlag(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"] = "info"
    message: str


class ValidationBlock(BaseModel):
    local_checks: list[LocalCheck] = Field(default_factory=list)
    integrity_refs: list[ArtifactRef] = Field(default_factory=list)
    business_flags: list[BusinessFlag] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(check.severity == "error" and not check.passed for check in self.local_checks) or any(
            flag.severity == "error" for flag in self.business_flags
        )

    @property
    def has_warnings(self) -> bool:
        return any(check.severity == "warning" and not check.passed for check in self.local_checks) or any(
            flag.severity == "warning" for flag in self.business_flags
        )


class RetryHint(BaseModel):
    retryable: bool = False
    suggested_action: str = "continue"
    reason_code: str = "none"
    details: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequirement(BaseModel):
    required: bool = False
    reason: str = ""
    approval_type: str = ""


class ContextRef(BaseModel):
    kind: str
    ref_id: str
    summary: str = ""


class AgentEnvelope(BaseModel):
    status: AgentStatus = AgentStatus.success
    agent_name: str
    summary: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    validation: ValidationBlock = Field(default_factory=ValidationBlock)
    retry_hint: RetryHint = Field(default_factory=RetryHint)
    approval: ApprovalRequirement = Field(default_factory=ApprovalRequirement)
    context_refs: list[ContextRef] = Field(default_factory=list)
    next_handoff: str = "validation_agent"

    def artifact_ids(self) -> list[str]:
        return [ref.artifact_id for ref in self.artifact_refs]


class AnalysisPlan(BaseModel):
    goal: str
    datasource_id: str | None = None
    catalog_summary: dict[str, Any] | None = None
    retry_context: dict[str, Any] | None = None
    planner_mode: Literal["llm", "deterministic"] = "deterministic"
    metric: str | None = None
    dimension: str | None = None
    filters: list[str] = Field(default_factory=list)
    requires_mart_review: bool = False
    route_kind: Literal["simple", "eda", "trend", "mart", "comprehensive"] = "simple"
    generated_sql: str = "SELECT 1 AS sample_value"
    source_sql: str = "SELECT 1 AS sample_value"


class OrchestrationState(BaseModel):
    run_id: str
    thread_id: str | None = None
    datasource_id: str | None = None
    catalog_summary: dict[str, Any] | None = None
    user_query: str
    goal: str = ""
    plan: AnalysisPlan | None = None
    retry_context: dict[str, Any] | None = None
    current_step: str = "created"
    artifact_ids: dict[str, list[str]] = Field(default_factory=dict)
    mart_candidate_ids: list[str] = Field(default_factory=list)
    mart_id: str | None = None
    validation_status: str = "not_started"
    approval_ids: list[str] = Field(default_factory=list)
    error_state: dict[str, Any] = Field(default_factory=dict)
    terminal_state: SupervisorTerminalState | None = None
    remaining_agents: list[str] = Field(default_factory=list)
    completed_agents: list[str] = Field(default_factory=list)
    last_agent: str | None = None
    route_kind: str = "simple"
    planner_mode: Literal["llm", "deterministic"] = "deterministic"
    generated_sql: str = ""
    retry_counts: dict[str, int] = Field(default_factory=dict)
    max_retry_per_agent: int = 1

    def add_artifacts(self, key: str, artifact_ids: list[str]) -> None:
        if not artifact_ids:
            return
        existing = self.artifact_ids.setdefault(key, [])
        existing.extend(artifact_id for artifact_id in artifact_ids if artifact_id not in existing)
