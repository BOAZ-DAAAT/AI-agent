from __future__ import annotations

from data_agent_backend.models.common import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


FindingDisposition = Literal["error", "warning", "diagnostic", "semantic_evidence", "limitation"]


class ValidationFinding(BaseModel):
    code: str
    source: str
    severity: Literal["info", "warning", "error"] = "info"
    disposition: FindingDisposition = "diagnostic"
    message: str
    retryable: bool = False
    suggested_action: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_disposition(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        disposition = str(normalized.get("disposition") or "").strip()
        if disposition == "blocking":
            normalized["disposition"] = "error"
        elif disposition == "retry_required":
            normalized["disposition"] = "error"
            normalized["retryable"] = True
        elif disposition == "advisory":
            normalized["disposition"] = "diagnostic"
        elif not disposition:
            normalized["disposition"] = "diagnostic"
        return normalized


class ValidationBlock(BaseModel):
    local_checks: list[LocalCheck] = Field(default_factory=list)
    integrity_refs: list[ArtifactRef] = Field(default_factory=list)
    business_flags: list[BusinessFlag] = Field(default_factory=list)
    findings: list[ValidationFinding] = Field(default_factory=list)

    def normalized_findings(self) -> list[ValidationFinding]:
        normalized = list(self.findings)
        normalized.extend(
            ValidationFinding(
                code=check.name,
                source="local_check",
                severity=check.severity,
                disposition="error" if check.severity == "error" else "limitation",
                message=check.detail or check.name,
                retryable=check.severity == "error",
            )
            for check in self.local_checks
            if not check.passed and check.severity in {"warning", "error"}
        )
        normalized.extend(
            ValidationFinding(
                code=flag.code,
                source="business_flag",
                severity=flag.severity,
                disposition="error" if flag.severity == "error" else "limitation",
                message=flag.message,
            )
            for flag in self.business_flags
            if flag.severity in {"warning", "error"}
        )
        return normalized

    @property
    def has_errors(self) -> bool:
        return any(finding.disposition == "error" for finding in self.normalized_findings())

    @property
    def has_warnings(self) -> bool:
        return any(
            finding.disposition in {"warning", "limitation"}
            for finding in self.normalized_findings()
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
    fallback_used: bool = False
    error: str = ""

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
    query_rules: dict[str, Any] = Field(default_factory=dict)
    required_derivations: list[dict[str, Any]] = Field(default_factory=list)
    analysis_heuristics: list[dict[str, Any]] = Field(default_factory=list)
    route_kind: Literal["simple", "eda", "trend", "mart", "comprehensive"] = "simple"
    generated_sql: str = ""
    source_sql: str = ""
    # comprehensive(마트) 경로에서 SQL 에이전트가 analytics 스키마에 적재한 마트 테이블 참조.
    # 하류(EDA/분석)는 이 이름으로 DB에서 마트를 직접 조회한다. simple 경로면 None.
    target_table: str | None = None
    # SQL 에이전트가 사용한 원천 테이블 목록 — 하류(EDA/분석)가 GE 정합성 결과를 이 테이블들로
    # 스코핑해 읽는 데 쓴다(load_scoped_integrity_text). 비어 있으면 캐비어트 없이 폴백.
    source_tables: list[str] = Field(default_factory=list)
    # SQL 에이전트가 선언한 엔티티 grain(예: "one row per customer_unique_id"). EDA가 자체
    # grain 추정치와 교차검증하는 데 쓴다. analysis의 time_grain(일/주/월)과는 다른 개념. None이면 스킵.
    business_grain: str | None = None
    mart_design: dict[str, Any] = Field(default_factory=dict)
    analysis_data_contract: dict[str, Any] = Field(default_factory=dict)


class OrchestrationState(BaseModel):
    run_id: str
    thread_id: str | None = None
    datasource_id: str | None = None
    catalog_summary: dict[str, Any] | None = None
    user_query: str
    final_answer: str = ""
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
    limitations: list[str] = Field(default_factory=list)
    analysis_review_decisions: list[dict[str, Any]] = Field(default_factory=list)

    def add_artifacts(self, key: str, artifact_ids: list[str]) -> None:
        if not artifact_ids:
            return
        existing = self.artifact_ids.setdefault(key, [])
        existing.extend(artifact_id for artifact_id in artifact_ids if artifact_id not in existing)


class SupervisorInterruptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["clarification", "analysis_review"]
    status: Literal["waiting_input"]
    run_id: str
    thread_id: str
    question: str
    node: str
    expected_resume: dict[str, str] = Field(default_factory=lambda: {"answer": "string"})
    approval_id: str | None = None
    review_request: dict[str, Any] | None = None
    input_mode: Literal["free_text", "choice_with_free_text", "approval"] = "free_text"
    options: list[dict[str, Any]] = Field(default_factory=list)
    allow_free_text: bool = True

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> "SupervisorInterruptPayload":
        if self.type == "clarification":
            if self.approval_id is not None or self.review_request is not None:
                raise ValueError("clarification interrupt에는 analysis review 필드를 포함할 수 없습니다.")
            if self.expected_resume != {"answer": "string"}:
                raise ValueError("clarification interrupt의 expected_resume이 올바르지 않습니다.")
            if self.input_mode == "choice_with_free_text" and not self.options:
                raise ValueError("choice clarification interrupt requires options.")
            return self
        if not (self.approval_id or "").strip() or not isinstance(self.review_request, dict):
            raise ValueError("analysis_review interrupt에는 approval_id와 review_request가 필요합니다.")
        expected = {
            "approval_id": "string",
            "selected_option_id": "string?",
            "free_text": "string?",
        }
        if self.expected_resume != expected:
            raise ValueError("analysis_review interrupt의 expected_resume이 올바르지 않습니다.")
        return self


class SupervisorRunResult(BaseModel):
    kind: Literal["state", "interrupt"]
    state: OrchestrationState | None = None
    interrupt: SupervisorInterruptPayload | None = None

    def __getattr__(self, item: str):
        state = object.__getattribute__(self, "state")
        if state is not None and hasattr(state, item):
            return getattr(state, item)
        raise AttributeError(item)

    @model_validator(mode="after")
    def validate_result_payload(self) -> SupervisorRunResult:
        if self.kind == "state":
            if self.state is None or self.interrupt is not None:
                raise ValueError("state 결과에는 state만 포함해야 합니다.")
        if self.kind == "interrupt":
            if self.interrupt is None or self.state is not None:
                raise ValueError("interrupt 결과에는 interrupt만 포함해야 합니다.")
        return self
