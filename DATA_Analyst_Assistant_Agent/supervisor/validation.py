from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    PostExecutionNextAction,
    SupervisorState,
    artifact_ids_by_agent,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    SupervisorTerminalState,
    ValidationFinding,
)


ValidationDisposition = Literal[
    "accept",
    "accept_with_limitations",
    "recover",
    "retry",
    "await_approval",
    "reject",
]


class ValidationCheckResult(BaseModel):
    name: Literal["contract", "result", "evidence", "semantic"]
    passed: bool
    findings: list[ValidationFinding] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationOutcome(BaseModel):
    disposition: ValidationDisposition
    reason: str = ""
    reason_code: str = "none"
    retry_target: AgentName | None = None
    recovery_action: NextAction | None = None
    terminal_state: str = "running"


class ValidationRecord(BaseModel):
    candidate_id: str
    validation_id: str
    agent: AgentName
    outcome: ValidationOutcome
    checks: list[ValidationCheckResult] = Field(default_factory=list)


class GuardDecision(BaseModel):
    allowed: bool
    next_action: NextAction
    reason: str


class ResultValidationDecision(BaseModel):
    valid: bool
    next_action: PostExecutionNextAction
    reason: str = ""
    decision: Literal["accept", "accept_with_limitations", "retry", "await_approval", "reject"] = "accept"
    terminal_state: str = "running"
    final_answer: str = ""
    reason_code: str = "none"
    failure_reason: str = ""
    repeated_failure: bool = False
    failure_streak: dict[str, Any] | None = None


class CompletionReadinessDecision(BaseModel):
    status: Literal["ready", "insight_required", "invalid"]
    reason: str


def contract_check_from_decision(
    result: AgentCompactResult,
    decision: ResultValidationDecision,
) -> ValidationCheckResult:
    """기존 결정론 검증 결과를 통합 검사 형식으로 변환한다."""
    passed = decision.decision in {"accept", "accept_with_limitations", "await_approval"}
    return ValidationCheckResult(
        name="result",
        passed=passed,
        findings=list(result.findings),
        details={
            "status": result.status,
            "validation_errors": list(result.validation_errors),
            "validation_warnings": list(result.validation_warnings),
            "fallback_used": result.fallback_used,
            "failure_reason": decision.failure_reason,
            "repeated_failure": decision.repeated_failure,
            "failure_streak": decision.failure_streak,
        },
    )


def outcome_from_contract_decision(
    agent: AgentName,
    decision: ResultValidationDecision,
) -> ValidationOutcome:
    """기존 라우팅 결정을 그래프 독립적인 통합 outcome으로 변환한다."""
    return ValidationOutcome(
        disposition=decision.decision,
        reason=decision.reason,
        reason_code=decision.reason_code,
        retry_target=agent if decision.decision == "retry" else None,
        terminal_state=decision.terminal_state,
    )


_AGENT_CALL_ACTIONS: dict[AgentName, NextAction] = {
    "sql_agent": "call_sql_agent",
    "eda_agent": "call_eda_agent",
    "analysis_agent": "call_analysis_agent",
    "insight": "call_insight",
}

_KNOWN_AGENTS: set[AgentName] = {
    "sql_agent", "eda_agent", "analysis_agent", "insight",
}
_EVIDENCE_AGENTS: set[AgentName] = {"sql_agent", "eda_agent", "analysis_agent"}


def _has_artifact(state: SupervisorState, *agents: AgentName) -> bool:
    artifacts = artifact_ids_by_agent(state)
    return any(bool(artifacts.get(agent)) for agent in agents)


def _has_completed_evidence(state: SupervisorState) -> bool:
    return _has_artifact(state, *_EVIDENCE_AGENTS)


def _has_valid_accepted_artifact(state: SupervisorState, agent: AgentName) -> bool:
    evidence = state.get("accepted_evidence") or {}
    return any(
        bool(str(item.get("artifact_id") or "").strip())
        for item in evidence.get(agent, [])
        if isinstance(item, dict)
    )


def _check_completion_readiness(state: SupervisorState) -> CompletionReadinessDecision:
    has_analysis_evidence = any(
        _has_valid_accepted_artifact(state, agent)
        for agent in _EVIDENCE_AGENTS
    )
    if not has_analysis_evidence:
        return CompletionReadinessDecision(
            status="invalid",
            reason="검증·승격된 SQL, EDA, 분석 근거가 없어 완료할 수 없습니다.",
        )

    insight_completed = "insight" in state.get("completed_agents", [])
    has_insight_artifact = _has_valid_accepted_artifact(state, "insight")
    if insight_completed and not has_insight_artifact:
        return CompletionReadinessDecision(
            status="invalid",
            reason="insight 완료 표식은 있지만 승격된 인사이트 아티팩트가 없습니다.",
        )
    if has_insight_artifact and not insight_completed:
        return CompletionReadinessDecision(
            status="invalid",
            reason="승격된 인사이트 아티팩트는 있지만 insight 완료 표식이 없습니다.",
        )
    if not insight_completed:
        return CompletionReadinessDecision(
            status="insight_required",
            reason="검증·승격된 분석 근거가 있어 인사이트 생성이 필요합니다.",
        )

    return CompletionReadinessDecision(
        status="ready",
        reason="분석 근거와 검증·승격된 인사이트가 있어 완료할 수 있습니다.",
    )


def guard_agent_preconditions(agent: AgentName | str, state: SupervisorState) -> GuardDecision:
    if agent not in _KNOWN_AGENTS:
        return GuardDecision(
            allowed=False,
            next_action="fail",
            reason=f"알 수 없는 unknown agent입니다: {agent}",
        )

    if agent == "sql_agent":
        return GuardDecision(
            allowed=True,
            next_action="call_sql_agent",
            reason="SQL 에이전트는 선행 산출물 없이 실행할 수 있습니다.",
        )

    if agent == "eda_agent":
        if _has_artifact(state, "sql_agent"):
            return GuardDecision(
                allowed=True,
                next_action="call_eda_agent",
                reason="SQL 산출물이 있어 EDA 에이전트를 실행할 수 있습니다.",
            )
        return GuardDecision(
            allowed=False,
            next_action="call_sql_agent",
            reason="EDA를 실행하려면 먼저 SQL 산출물이 필요합니다.",
        )

    if agent == "analysis_agent":
        if _has_artifact(state, "sql_agent", "eda_agent"):
            return GuardDecision(
                allowed=True,
                next_action="call_analysis_agent",
                reason="SQL 또는 EDA 산출물이 있어 분석 에이전트를 실행할 수 있습니다.",
            )
        return GuardDecision(
            allowed=False,
            next_action="call_sql_agent",
            reason="분석을 실행하려면 먼저 SQL 또는 EDA 산출물이 필요합니다.",
        )

    if agent == "insight":
        if _has_completed_evidence(state):
            return GuardDecision(
                allowed=True,
                next_action="call_insight",
                reason="SQL, EDA, 분석 중 하나 이상의 근거 산출물이 있어 인사이트를 생성할 수 있습니다.",
            )
        return GuardDecision(
            allowed=False,
            next_action="call_analysis_agent",
            reason="인사이트를 생성하려면 먼저 SQL, EDA, 분석 중 하나 이상의 근거가 필요합니다.",
        )


def validate_subagent_result(
    state: SupervisorState,
    result: AgentCompactResult,
) -> ResultValidationDecision:
    fallback_used = result.fallback_used
    blocking_findings = [finding for finding in result.findings if finding.disposition == "blocking"]
    retry_findings = [finding for finding in result.findings if finding.disposition == "retry_required"]
    limitation_findings = [finding for finding in result.findings if finding.disposition == "limitation"]
    has_validation_errors = bool(result.validation_errors or blocking_findings)

    if result.status == "failed":
        return _route_explicit_failure(state, result)

    if result.status in {"success", "warning"} and (has_validation_errors or fallback_used):
        return _route_invalid_result(
            state,
            result,
            has_validation_errors=has_validation_errors,
            fallback_used=fallback_used,
        )

    if retry_findings:
        retry_count = int(state.get("retry_counts", {}).get(result.agent, 0))
        max_retry = int(state.get("max_retry_per_agent", 0))
        if retry_count < max_retry:
            return ResultValidationDecision(
                valid=False,
                next_action=_AGENT_CALL_ACTIONS[result.agent],
                reason="; ".join(finding.message for finding in retry_findings),
                decision="retry",
            )
        return ResultValidationDecision(
            valid=False,
            next_action="fail",
            reason=f"retry-required finding의 재시도 한도 {max_retry}회에 도달했습니다.",
            decision="reject",
            terminal_state=SupervisorTerminalState.failed_terminal.value,
        )

    if result.approval.required:
        return ResultValidationDecision(
            valid=False,
            next_action="finalize",
            reason=result.approval.reason or f"{result.agent} 실행 결과에 승인이 필요합니다: {result.summary}",
            decision="await_approval",
        )

    if result.status == "approval_required":
        failure_reason = "status=approval_required이지만 approval.required=false입니다."
        return ResultValidationDecision(
            valid=False,
            next_action="fail",
            reason=failure_reason,
            decision="reject",
            terminal_state=SupervisorTerminalState.failed_terminal.value,
            reason_code="approval_contract_mismatch",
            failure_reason=failure_reason,
        )

    if result.status in {"success", "warning"}:
        return ResultValidationDecision(
            valid=True,
            next_action="decide_next_action",
            reason=f"{result.agent} 결과가 유효해 Supervisor의 다음 행동 재판단으로 이동합니다.",
            decision="accept_with_limitations" if limitation_findings else "accept",
        )

    return ResultValidationDecision(
        valid=False,
        next_action="fail",
        reason=f"{result.agent} 결과 상태를 처리할 수 없습니다: {result.status}",
        decision="reject",
        terminal_state=SupervisorTerminalState.failed_terminal.value,
    )


def _route_explicit_failure(
    state: SupervisorState,
    result: AgentCompactResult,
) -> ResultValidationDecision:
    reason_code = _failure_reason_code(result)
    failure_reason = _failure_reason(result)
    signature = _failure_signature(reason_code, failure_reason)
    previous = (state.get("failure_streaks") or {}).get(result.agent) or {}
    repeated_failure = previous.get("signature") == signature
    consecutive_count = int(previous.get("consecutive_count", 0)) + 1 if repeated_failure else 1
    failure_streak = {
        "reason_code": reason_code,
        "failure_reason": failure_reason,
        "signature": signature,
        "consecutive_count": consecutive_count,
    }

    retry_count = int(state.get("retry_counts", {}).get(result.agent, 0))
    max_retry = int(state.get("max_retry_per_agent", 0))
    retryable = bool(result.retryable or result.retry_hint.retryable)
    if not repeated_failure and retryable and retry_count < max_retry:
        return ResultValidationDecision(
            valid=False,
            next_action=_AGENT_CALL_ACTIONS[result.agent],
            reason=(
                f"{result.agent} 실패 [{reason_code}]: {failure_reason} "
                f"(repeated_failure=false) 재시도 {retry_count}/{max_retry}"
            ),
            decision="retry",
            reason_code=reason_code,
            failure_reason=failure_reason,
            repeated_failure=False,
            failure_streak=failure_streak,
        )

    terminal_state = SupervisorTerminalState.failed_terminal.value
    if repeated_failure and result.agent == "analysis_agent" and _has_artifact(
        state, "sql_agent", "eda_agent"
    ):
        terminal_state = SupervisorTerminalState.failed_with_recoverable_context.value
    reason = (
        f"{result.agent} 실패 [{reason_code}]: {failure_reason} "
        f"(repeated_failure={'true' if repeated_failure else 'false'})"
    )
    return ResultValidationDecision(
        valid=False,
        next_action="fail",
        reason=reason,
        decision="reject",
        terminal_state=terminal_state,
        final_answer=reason,
        reason_code=reason_code,
        failure_reason=failure_reason,
        repeated_failure=repeated_failure,
        failure_streak=failure_streak,
    )


def _failure_reason_code(result: AgentCompactResult) -> str:
    reason_code = str(result.retry_hint.reason_code or "")
    return reason_code if reason_code.strip() else "none"


def _failure_reason(result: AgentCompactResult) -> str:
    values = (
        result.retry_hint.details.get("failure_reason"),
        result.error,
        result.summary,
    )
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return "알 수 없는 실패"


def _failure_signature(reason_code: str, failure_reason: str) -> str:
    normalized = [
        " ".join(reason_code.split()).casefold(),
        " ".join(failure_reason.split()).casefold(),
    ]
    return json.dumps(normalized, ensure_ascii=False)


def _route_invalid_result(
    state: SupervisorState,
    result: AgentCompactResult,
    *,
    has_validation_errors: bool,
    fallback_used: bool,
) -> ResultValidationDecision:
    retry_count = int(state.get("retry_counts", {}).get(result.agent, 0))
    max_retry = int(state.get("max_retry_per_agent", 0))
    detail = _invalid_reason_detail(
        result,
        has_validation_errors=has_validation_errors,
        fallback_used=fallback_used,
    )

    if result.retryable and retry_count < max_retry:
        return ResultValidationDecision(
            valid=False,
            next_action=_AGENT_CALL_ACTIONS[result.agent],
            reason=f"{detail} 재시도 가능하며 현재 재시도 횟수는 {retry_count}/{max_retry}입니다.",
            decision="retry",
        )

    if result.retryable:
        reason = f"{detail} 재시도 한도 {max_retry}회에 도달해 실패로 종료합니다."
    else:
        reason = f"{detail} 재시도할 수 없어 실패로 종료합니다."
    return ResultValidationDecision(
        valid=False,
        next_action="fail",
        reason=reason,
        decision="reject",
        terminal_state=SupervisorTerminalState.failed_terminal.value,
    )


def _invalid_reason_detail(
    result: AgentCompactResult,
    *,
    has_validation_errors: bool,
    fallback_used: bool,
) -> str:
    if fallback_used:
        return f"{result.agent} 결과가 fallback으로 생성되어 성공처럼 처리할 수 없습니다: {result.summary}"
    if has_validation_errors:
        errors = "; ".join(result.validation_errors)
        return f"{result.agent} 결과에 validation_errors가 있습니다: {errors}"
    message = result.error or result.summary
    return f"{result.agent} 실행이 실패했습니다: {message}"
