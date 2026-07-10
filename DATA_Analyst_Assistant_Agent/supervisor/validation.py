from __future__ import annotations

from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    SupervisorState,
    artifact_ids_by_agent,
)


class GuardDecision(BaseModel):
    allowed: bool
    next_action: NextAction
    reason: str


class ResultValidationDecision(BaseModel):
    valid: bool
    next_action: NextAction
    reason: str


_AGENT_CALL_ACTIONS: dict[AgentName, NextAction] = {
    "sql_agent": "call_sql_agent",
    "eda_agent": "call_eda_agent",
    "analysis_agent": "call_analysis_agent",
    "report_agent": "call_report_agent",
}

_KNOWN_AGENTS: set[AgentName] = {"sql_agent", "eda_agent", "analysis_agent", "report_agent"}
_EVIDENCE_AGENTS: set[AgentName] = {"sql_agent", "eda_agent", "analysis_agent"}


def _has_artifact(state: SupervisorState, *agents: AgentName) -> bool:
    artifacts = artifact_ids_by_agent(state)
    return any(bool(artifacts.get(agent)) for agent in agents)


def _has_completed_evidence(state: SupervisorState) -> bool:
    return _has_artifact(state, *_EVIDENCE_AGENTS)


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

    if _has_completed_evidence(state):
        return GuardDecision(
            allowed=True,
            next_action="call_report_agent",
            reason="SQL, EDA, 분석 중 하나 이상의 근거 산출물이 있어 리포트를 생성할 수 있습니다.",
        )
    return GuardDecision(
        allowed=False,
        next_action="call_analysis_agent",
        reason="리포트를 생성하려면 먼저 SQL, EDA, 분석 중 하나 이상의 근거가 필요합니다.",
    )


def validate_subagent_result(
    state: SupervisorState,
    result: AgentCompactResult,
) -> ResultValidationDecision:
    if result.status == "approval_required":
        return ResultValidationDecision(
            valid=False,
            next_action="finalize",
            reason=f"{result.agent} 실행 결과에 승인 대기가 필요합니다: {result.summary}",
        )

    fallback_used = result.fallback_used
    has_validation_errors = bool(result.validation_errors)
    has_failure = result.status == "failed" or has_validation_errors or fallback_used
    if has_failure:
        return _route_invalid_result(
            state,
            result,
            has_validation_errors=has_validation_errors,
            fallback_used=fallback_used,
        )

    if result.status in {"success", "warning"}:
        if result.agent == "report_agent":
            if not _result_has_artifact(result):
                return ResultValidationDecision(
                    valid=False,
                    next_action="fail",
                    reason="리포트 에이전트 결과에 리포트 산출물 ID가 없어 완료할 수 없습니다.",
                )
            return ResultValidationDecision(
                valid=True,
                next_action="finalize",
                reason="리포트 에이전트가 유효한 결과를 반환해 종료할 수 있습니다.",
            )
        return ResultValidationDecision(
            valid=True,
            next_action="create_plan",
            reason=f"{result.agent} 결과가 유효해 다음 계획 수립으로 이동합니다.",
        )

    return ResultValidationDecision(
        valid=False,
        next_action="fail",
        reason=f"{result.agent} 결과 상태를 처리할 수 없습니다: {result.status}",
    )


def _result_has_artifact(result: AgentCompactResult) -> bool:
    if any(bool(artifact_id) for artifact_id in result.artifact_ids):
        return True
    return any(bool(artifact.artifact_id) for artifact in result.artifacts)


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
        )

    if result.retryable:
        reason = f"{detail} 재시도 한도 {max_retry}회에 도달해 실패로 종료합니다."
    else:
        reason = f"{detail} 재시도할 수 없어 실패로 종료합니다."
    return ResultValidationDecision(valid=False, next_action="fail", reason=reason)


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
