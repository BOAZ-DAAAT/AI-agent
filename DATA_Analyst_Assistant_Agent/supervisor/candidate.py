from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState, ValidationFinding
from DATA_Analyst_Assistant_Agent.supervisor.decision import (
    SemanticValidationAdvisoryDecision,
    build_result_validation_context,
    invoke_supervisor_decision,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import SEMANTIC_VALIDATION_ADVISORY_PROMPT
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    StepSummary,
    SupervisorState,
    normalize_supervisor_state,
    promote_pending_result,
    reject_pending_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    ValidationCheckResult,
    ValidationOutcome,
    ValidationRecord,
    contract_check_from_decision,
    outcome_from_contract_decision,
    validate_subagent_result,
)


_ACTION_BY_AGENT: dict[AgentName, NextAction] = {
    "sql_agent": "call_sql_agent",
    "eda_agent": "call_eda_agent",
    "analysis_agent": "call_analysis_agent",
    "report_agent": "call_report_agent",
}


def build_step_summary(
    result: AgentCompactResult,
    action: NextAction,
    next_action: NextAction,
) -> StepSummary:
    """승격된 에이전트 결과의 결정론적 단계 요약을 생성한다."""
    artifact_ids = list(
        dict.fromkeys(
            [*result.artifact_ids, *(artifact.artifact_id for artifact in result.artifacts)]
        )
    )
    return StepSummary(
        step="generate_report" if result.agent == "report_agent" else "execute_subagent",
        agent=result.agent,
        action=action,
        summary=result.summary,
        artifact_ids=artifact_ids,
        next_action=next_action,
    )


def validate_candidate(
    state: SupervisorState,
    model: Any | None,
) -> SupervisorState:
    """격리 후보의 계약, 결과, 의미 검증 결과를 하나의 레코드로 만든다."""
    pending = state.get("pending_result")
    payload = pending.get("result") if isinstance(pending, dict) else None
    if not isinstance(payload, dict):
        return _failure(state, "검증할 후보 결과가 없습니다.", node="validate_candidate")
    try:
        result = AgentCompactResult.model_validate(payload)
    except Exception as exc:
        return _failure(
            state,
            f"에이전트 실행 결과 형식이 올바르지 않습니다: {exc}",
            node="validate_candidate",
        )

    approval_contract_valid = not (
        result.status == "approval_required" and not result.approval.required
    )
    findings = [] if approval_contract_valid else [
        ValidationFinding(
            code="approval_contract_mismatch",
            source="supervisor",
            severity="error",
            disposition="blocking",
            message="status=approval_required이지만 approval.required=false입니다.",
        )
    ]
    checks = [
        ValidationCheckResult(
            name="contract",
            passed=approval_contract_valid,
            findings=findings,
            details={
                "schema": "AgentCompactResult",
                "approval_contract_valid": approval_contract_valid,
            },
        )
    ]
    if not approval_contract_valid:
        return _validation_updates(
            pending,
            result,
            checks,
            ValidationOutcome(
                disposition="reject",
                reason=findings[0].message,
                reason_code="approval_contract_mismatch",
                terminal_state=SupervisorTerminalState.failed_terminal.value,
            ),
        )

    contract_decision = validate_subagent_result(state, result)
    checks.append(contract_check_from_decision(result, contract_decision))
    outcome = outcome_from_contract_decision(result.agent, contract_decision)
    if outcome.disposition in {"retry", "reject"}:
        return _validation_updates(pending, result, checks, outcome)

    updated_pending = dict(pending)

    provisional = ValidationRecord(
        candidate_id=str(pending.get("candidate_id") or ""),
        validation_id=str(pending.get("validation_id") or ""),
        agent=result.agent,
        outcome=outcome,
        checks=checks,
    ).model_dump(mode="json")
    semantic_state = {
        **state,
        "pending_result": updated_pending,
        "validation_history": [*state.get("validation_history", []), provisional],
    }
    semantic_decision = None
    semantic_error: Exception | None = None
    semantic_failures = 0
    for _ in range(2):
        try:
            semantic_decision = invoke_supervisor_decision(
                semantic_state,
                model,
                SEMANTIC_VALIDATION_ADVISORY_PROMPT,
                SemanticValidationAdvisoryDecision,
                extra=build_result_validation_context(semantic_state),
            )
            break
        except Exception as exc:
            semantic_error = exc
            semantic_failures += 1

    if semantic_decision is None:
        reason = f"semantic validation 모델 호출에 반복 실패했습니다: {semantic_error}"
        checks.append(
            ValidationCheckResult(
                name="semantic",
                passed=False,
                findings=[
                    ValidationFinding(
                        code="semantic_model_failed",
                        source="supervisor",
                        severity="error",
                        disposition="blocking",
                        message=reason,
                    )
                ],
                details={"attempts": 2},
            )
        )
        updates = {
            **_validation_updates(
                pending,
                result,
                checks,
                ValidationOutcome(
                    disposition="reject",
                    reason=reason,
                    reason_code="semantic_model_failed",
                    terminal_state=SupervisorTerminalState.failed_with_recoverable_context.value,
                ),
            ),
            "pending_result": updated_pending,
        }
        counts = dict(state.get("semantic_retry_counts", {}))
        counts[str(pending.get("candidate_id") or result.agent)] = semantic_failures
        updates["semantic_retry_counts"] = counts
        return updates

    semantic_invalid = (
        not semantic_decision.semantic_valid
        or semantic_decision.severity == "error"
        or bool(semantic_decision.missing_evidence)
        or not _semantic_action_allowed(result.agent, semantic_decision.recommended_next_action)
    )
    semantic_findings = [] if not semantic_invalid else [
        ValidationFinding(
            code="semantic_validation_failed",
            source="supervisor",
            severity="error",
            disposition="blocking",
            message=semantic_decision.reason or "semantic validation을 통과하지 못했습니다.",
            details={"missing_evidence": list(semantic_decision.missing_evidence)},
        )
    ]
    checks.append(
        ValidationCheckResult(
            name="semantic",
            passed=not semantic_invalid,
            findings=semantic_findings,
            details=semantic_decision.model_dump(mode="json"),
        )
    )
    if semantic_invalid:
        outcome = ValidationOutcome(
            disposition="reject",
            reason=semantic_decision.reason or "semantic validation을 통과하지 못했습니다.",
            reason_code="semantic_validation_failed",
            terminal_state=SupervisorTerminalState.failed_terminal.value,
        )
    elif contract_decision.decision == "await_approval":
        outcome = outcome_from_contract_decision(result.agent, contract_decision)
    elif contract_decision.decision == "accept_with_limitations":
        outcome = ValidationOutcome(
            disposition="accept_with_limitations",
            reason=contract_decision.reason,
        )
    else:
        outcome = ValidationOutcome(disposition="accept", reason=contract_decision.reason)
    updates = {
        **_validation_updates(pending, result, checks, outcome),
        "pending_result": updated_pending,
        "llm_decisions": _append_llm_decision(state, semantic_decision),
    }
    if semantic_failures:
        counts = dict(state.get("semantic_retry_counts", {}))
        counts[str(pending.get("candidate_id") or result.agent)] = semantic_failures
        updates["semantic_retry_counts"] = counts
    return updates


def commit_candidate(
    state: SupervisorState,
    backend_adapter: Any | None,
    *,
    approval_granted: bool = False,
) -> SupervisorState:
    """검증 결과를 멱등하게 기록하고 후보를 승격하거나 거절한다."""
    normalized = normalize_supervisor_state(state)
    pending = normalized.get("pending_result")
    validation_payload = normalized.get("pending_validation")

    if not isinstance(pending, dict):
        return {**normalized, "pending_validation": None, "current_step": "commit_candidate"}

    if not isinstance(validation_payload, dict) and approval_granted:
        validation_payload = next(
            (
                item
                for item in reversed(normalized.get("validation_history", []))
                if item.get("candidate_id") == pending.get("candidate_id")
                and item.get("validation_id") == pending.get("validation_id")
            ),
            None,
        )
        if not isinstance(validation_payload, dict):
            result_payload = pending.get("result") or {}
            validation_payload = ValidationRecord(
                candidate_id=str(pending.get("candidate_id") or ""),
                validation_id=str(pending.get("validation_id") or ""),
                agent=AgentCompactResult.model_validate(result_payload).agent,
                outcome=ValidationOutcome(
                    disposition="await_approval",
                    reason="기존 승인 대기 체크포인트를 재개합니다.",
                ),
                checks=[],
            ).model_dump(mode="json")
    if not isinstance(validation_payload, dict):
        return _failure(normalized, "반영할 후보 검증 결과가 없습니다.")

    try:
        record = ValidationRecord.model_validate(validation_payload)
        result = AgentCompactResult.model_validate(pending.get("result") or {})
    except Exception as exc:
        return _failure(normalized, f"후보 검증 결과 형식이 올바르지 않습니다: {exc}")

    if (
        record.candidate_id != pending.get("candidate_id")
        or record.validation_id != pending.get("validation_id")
        or record.agent != result.agent
    ):
        return _failure(normalized, "후보와 검증 결과의 식별자가 일치하지 않습니다.")

    record_payload = record.model_dump(mode="json")
    history = list(normalized.get("validation_history", []))
    already_committed = any(
        item.get("candidate_id") == record.candidate_id
        and item.get("validation_id") == record.validation_id
        for item in history
    )
    if not already_committed:
        history.append(record_payload)

    working: SupervisorState = {
        **normalized,
        "validation_history": history,
        "pending_validation": None,
        "current_step": "commit_candidate",
    }
    disposition = record.outcome.disposition

    result_check = next((check for check in record.checks if check.name == "result"), None)
    failure_streak = result_check.details.get("failure_streak") if result_check else None
    if isinstance(failure_streak, dict):
        streaks = {key: dict(value) for key, value in working.get("failure_streaks", {}).items()}
        streaks[result.agent] = dict(failure_streak)
        working["failure_streaks"] = streaks

    if disposition == "await_approval" and not approval_granted:
        approval = {
            "approval_id": f"{working['current_run_id']}:{result.agent}:approval",
            "agent": result.agent,
            "reason": result.approval.reason or result.summary,
            "approval_type": result.approval.approval_type or "agent_approval",
            "candidate_id": record.candidate_id,
            "validation_id": record.validation_id,
            "content_hashes": dict(pending.get("content_hashes") or {}),
        }
        return {
            **working,
            "pending_approval": approval,
            "terminal_state": SupervisorTerminalState.needs_user_approval.value,
            "next_action": "finalize",
            "final_answer": approval["reason"],
        }

    if disposition in {"retry", "reject"}:
        rejected = reject_pending_result(
            working,
            record.outcome.reason,
            metadata={
                "agent": record.agent,
                "reason_code": record.outcome.reason_code,
                "disposition": disposition,
            },
        )
        rejected["pending_validation"] = None
        rejected["current_step"] = "commit_candidate"
        if disposition == "retry":
            counts = dict(rejected.get("retry_counts", {}))
            if not already_committed:
                counts[result.agent] = int(counts.get(result.agent, 0)) + 1
            rejected["retry_counts"] = counts
            rejected["next_action"] = _ACTION_BY_AGENT[result.agent]
            rejected["terminal_state"] = "running"
        else:
            failed = list(rejected.get("failed_agents", []))
            if result.agent not in failed:
                failed.append(result.agent)
            rejected["failed_agents"] = failed
            rejected["next_action"] = "finalize"
            rejected["terminal_state"] = record.outcome.terminal_state
            rejected["final_answer"] = record.outcome.reason
        if not already_committed:
            _emit_event(
                backend_adapter,
                working,
                "validation.rejected",
                record.outcome.reason,
                metadata={
                    "candidate_id": record.candidate_id,
                    "validation_id": record.validation_id,
                    "agent": record.agent,
                    "disposition": disposition,
                },
            )
        return rejected

    if disposition not in {"accept", "accept_with_limitations", "await_approval"}:
        return _failure(working, f"지원하지 않는 validation disposition입니다: {disposition}")

    promoted = promote_pending_result(working, approval_granted=approval_granted)
    next_action = _next_action(result, record)
    promoted["next_action"] = next_action
    promoted["current_step"] = "commit_candidate"
    summaries = list(promoted.get("step_summaries", []))
    if not any(
        item.get("agent") == result.agent
        and item.get("action") == _ACTION_BY_AGENT[result.agent]
        and item.get("summary") == result.summary
        and item.get("artifact_ids") == build_step_summary(
            result, _ACTION_BY_AGENT[result.agent], next_action
        ).artifact_ids
        for item in summaries
    ):
        summaries.append(
            build_step_summary(
                result,
                _ACTION_BY_AGENT[result.agent],
                next_action,
            ).model_dump(mode="json")
        )
    promoted["step_summaries"] = summaries
    if not already_committed:
        _emit_event(
            backend_adapter,
            working,
            "evidence.promoted",
            f"{result.agent} 후보 근거를 승격했습니다.",
            artifact_ids=build_step_summary(
                result,
                _ACTION_BY_AGENT[result.agent],
                next_action,
            ).artifact_ids,
            metadata={
                "candidate_id": record.candidate_id,
                "validation_id": record.validation_id,
            },
        )
    return promoted


def _next_action(result: AgentCompactResult, record: ValidationRecord) -> NextAction:
    if result.agent == "report_agent":
        return "finalize"
    semantic = next((check.details for check in record.checks if check.name == "semantic"), {})
    recommendation = str(semantic.get("recommended_next_action") or "")
    if _semantic_action_allowed(result.agent, recommendation):
        return recommendation or "decide_next_action"  # type: ignore[return-value]
    return "decide_next_action"


def _semantic_action_allowed(agent: str, action: str) -> bool:
    if not action:
        return True
    if action in {"decide_next_action", "finalize", "fail"}:
        return True
    allowed = {
        "sql_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"},
        "eda_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"},
        "analysis_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"},
        "report_agent": {"call_report_agent"},
    }
    return action in allowed.get(agent, set())


def _validation_updates(
    pending: dict[str, Any],
    result: AgentCompactResult,
    checks: list[ValidationCheckResult],
    outcome: ValidationOutcome,
) -> SupervisorState:
    record = ValidationRecord(
        candidate_id=str(pending.get("candidate_id") or ""),
        validation_id=str(pending.get("validation_id") or ""),
        agent=result.agent,
        outcome=outcome,
        checks=checks,
    )
    return {
        "pending_validation": record.model_dump(mode="json"),
        "current_step": "validate_candidate",
    }


def _append_llm_decision(
    state: SupervisorState,
    decision: SemanticValidationAdvisoryDecision,
) -> list[dict[str, Any]]:
    entries = list(state.get("llm_decisions", []))
    entries.append(
        {
            "node": "validate_candidate",
            "schema": decision.__class__.__name__,
            "decision": decision.model_dump(mode="json"),
        }
    )
    return entries


def _failure(
    state: SupervisorState,
    message: str,
    *,
    node: str = "commit_candidate",
) -> SupervisorState:
    return {
        **state,
        "pending_validation": None,
        "terminal_state": SupervisorTerminalState.failed_terminal.value,
        "next_action": "finalize",
        "final_answer": message,
        "current_step": node,
        "error_state": {
            "node": node,
            "message": message,
            "retryable": False,
        },
    }


def _emit_event(
    backend_adapter: Any | None,
    state: SupervisorState,
    event_type: str,
    message: str,
    *,
    artifact_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    append_event = getattr(backend_adapter, "append_run_event", None)
    run_id = state.get("current_run_id")
    if append_event is None or not run_id:
        return
    append_event(
        run_id,
        event_type,
        message,
        node_name="supervisor",
        artifact_ids=artifact_ids,
        metadata=metadata,
    )
