from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState, ValidationFinding
from DATA_Analyst_Assistant_Agent.supervisor.decision import (
    SemanticValidationAdvisoryDecision,
    build_result_validation_context,
    invoke_supervisor_decision,
)
from DATA_Analyst_Assistant_Agent.supervisor.lifecycle import (
    emit_node_lifecycle_event,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import SEMANTIC_VALIDATION_ADVISORY_PROMPT
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    StepSummary,
    SupervisorState,
    PendingApproval,
    begin_or_retry_agent_node,
    complete_active_node,
    discard_active_node_for_recovery,
    fail_active_node,
    normalize_supervisor_state,
    promote_pending_result,
    reject_pending_result,
    wait_active_node,
)
from DATA_Analyst_Assistant_Agent.supervisor.analysis_review import (
    InvalidAnalysisReviewRequest,
    extract_analysis_review_request,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    ValidationCheckResult,
    ValidationOutcome,
    ValidationRecord,
    contract_check_from_decision,
    outcome_from_contract_decision,
    guard_agent_preconditions,
    validate_subagent_result,
)


_ACTION_BY_AGENT: dict[AgentName, NextAction] = {
    "sql_agent": "call_sql_agent",
    "eda_agent": "call_eda_agent",
    "analysis_agent": "call_analysis_agent",
    "insight": "call_insight",
}
_AGENT_BY_ACTION: dict[str, AgentName] = {
    action: agent for agent, action in _ACTION_BY_AGENT.items()
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
        step="execute_subagent",
        agent=result.agent,
        action=action,
        summary=result.summary,
        artifact_ids=artifact_ids,
        next_action=next_action,
    )


def _enrich_step_summary_with_node_finding(
    step_summary: StepSummary,
    backend_adapter: Any | None,
) -> StepSummary:
    """가능하면 결정론적 summary를, generate_node_summary()가 만든 key_finding으로 교체한다.

    노드가 UI에 완료 상태로 뜰 때 같이 보일 한 줄이라 여기서 채워 넣는다. 실패(어댑터
    없음/LLM 오류/근거 부족 등)해도 원래 결정론적 summary로 안전하게 되돌아간다 —
    이 함수가 노드 완료 자체를 막으면 안 된다.
    """
    if backend_adapter is None or not step_summary.artifact_ids:
        return step_summary

    try:
        import json

        from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
        from DATA_Analyst_Assistant_Agent.supervisor.summary.generator import generate_node_summary

        runtime = AgentRuntime(backend_adapter)
        ref = generate_node_summary(step_summary.artifact_ids, runtime)
        payload = json.loads(backend_adapter.read_artifact_text(ref.artifact_id))
        key_finding = str(payload.get("key_finding") or "").strip()
        updates: dict[str, Any] = {"summary_artifact_id": ref.artifact_id}
        if key_finding:
            updates["summary"] = key_finding
        return step_summary.model_copy(update=updates)
    except Exception:  # noqa: BLE001 - 서머리는 부가 정보, 실패해도 완료 흐름은 계속
        return step_summary


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

    semantic_recover = (
        semantic_decision.severity == "error"
        or bool(semantic_decision.missing_evidence)
        or (
            semantic_decision.severity == "info"
            and not semantic_decision.semantic_valid
        )
    )
    semantic_warning = semantic_decision.severity == "warning" and not semantic_decision.missing_evidence
    if semantic_recover:
        semantic_findings = [
            ValidationFinding(
                code="semantic_validation_failed",
                source="supervisor",
                severity="error",
                disposition="blocking",
                message=semantic_decision.reason or "semantic validation을 통과하지 못했습니다.",
                details={"missing_evidence": list(semantic_decision.missing_evidence)},
            )
        ]
    elif semantic_warning:
        semantic_findings = [
            ValidationFinding(
                code="semantic_validation_warning",
                source="supervisor",
                severity="warning",
                disposition="limitation",
                message=semantic_decision.reason or "semantic validation에 제한사항이 있습니다.",
            )
        ]
    else:
        semantic_findings = []
    checks.append(
        ValidationCheckResult(
            name="semantic",
            passed=not semantic_recover,
            findings=semantic_findings,
            details=semantic_decision.model_dump(mode="json"),
        )
    )
    if semantic_recover:
        outcome = ValidationOutcome(
            disposition="recover",
            reason=semantic_decision.reason or "semantic validation을 통과하지 못했습니다.",
            reason_code="semantic_validation_failed",
            recovery_action=semantic_decision.recommended_next_action or None,
        )
    elif contract_decision.decision == "await_approval":
        outcome = outcome_from_contract_decision(result.agent, contract_decision)
    elif semantic_warning or contract_decision.decision == "accept_with_limitations":
        outcome = ValidationOutcome(
            disposition="accept_with_limitations",
            reason=semantic_decision.reason if semantic_warning else contract_decision.reason,
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
    source_schema_version = int(state.get("state_schema_version", 0) or 0)
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

    limitation_messages = [
        finding.message
        for check in record.checks
        for finding in check.findings
        if finding.disposition == "limitation" and finding.message
    ]
    if limitation_messages:
        working["limitations"] = _append_unique(
            working.get("limitations", []),
            *limitation_messages,
        )

    active_node_payload = working.get("active_node")
    if (
        approval_granted
        and isinstance(active_node_payload, dict)
        and active_node_payload.get("status") == "waiting"
    ):
        working, resumed_node, resumed_event_type = begin_or_retry_agent_node(
            working,
            result.agent,
        )
        pending_approval = working.get("pending_approval")
        approval_id = (
            str(pending_approval.get("approval_id") or "")
            if isinstance(pending_approval, dict)
            else ""
        )
        emit_node_lifecycle_event(
            backend_adapter,
            working,
            resumed_event_type,
            resumed_node,
            f"{result.agent} 작업을 재개했습니다.",
            action=_ACTION_BY_AGENT[result.agent],
            approval_id=approval_id or None,
            metadata={
                "candidate_id": record.candidate_id,
                "validation_id": record.validation_id,
            },
        )

    if disposition == "await_approval" and not approval_granted:
        review_request = None
        if result.agent == "analysis_agent" and result.approval.approval_type == "analysis.review":
            try:
                review_request = extract_analysis_review_request(result)
            except InvalidAnalysisReviewRequest as exc:
                return {
                    **working,
                    "terminal_state": SupervisorTerminalState.failed_terminal.value,
                    "next_action": "finalize",
                    "final_answer": str(exc),
                    "error_state": {
                        "node": "commit_candidate",
                        "message": str(exc),
                        "reason_code": "invalid_analysis_review_request",
                        "retryable": False,
                    },
                }
        is_structured_review = review_request is not None
        approval = PendingApproval(
            approval_id=(
                f"{working['current_run_id']}:analysis_agent:{record.candidate_id}:approval"
                if is_structured_review
                else f"{working['current_run_id']}:{result.agent}:approval"
            ),
            agent=result.agent,
            reason=result.approval.reason or result.summary,
            approval_type=result.approval.approval_type or "agent_approval",
            candidate_id=record.candidate_id,
            validation_id=record.validation_id,
            content_hashes=dict(pending.get("content_hashes") or {}),
            review_request=(
                review_request.model_dump(mode="json") if review_request is not None else None
            ),
            expected_resume=(
                {
                    "approval_id": "string",
                    "selected_option_id": "string?",
                    "free_text": "string?",
                }
                if is_structured_review
                else {"approved": "boolean"}
            ),
        ).model_dump(mode="json")
        approval_state: SupervisorState = {
            **working,
            "pending_approval": approval,
            "terminal_state": (
                "running"
                if is_structured_review
                else SupervisorTerminalState.needs_user_approval.value
            ),
            "next_action": "collect_analysis_review" if is_structured_review else "finalize",
            "final_answer": "" if is_structured_review else approval["reason"],
        }
        if isinstance(approval_state.get("active_node"), dict):
            waiting_state, waiting_node, waiting_event_type = wait_active_node(
                approval_state,
                agent_name=result.agent,
                reason=approval["reason"],
            )
            emit_node_lifecycle_event(
                backend_adapter,
                waiting_state,
                waiting_event_type,
                waiting_node,
                approval["reason"],
                action=_ACTION_BY_AGENT[result.agent],
                approval_id=approval["approval_id"],
                metadata={
                    "candidate_id": record.candidate_id,
                    "validation_id": record.validation_id,
                    "approval_type": approval["approval_type"],
                },
            )
            return waiting_state

        if source_schema_version >= 6:
            return _failure(
                approval_state,
                f"{result.agent} 결과를 대기 처리할 active_node가 없습니다.",
            )

        return approval_state

    if disposition == "recover":
        return _commit_semantic_recovery(
            working,
            pending,
            record,
            result,
            backend_adapter,
            already_committed=already_committed,
        )

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

        if disposition == "reject" and isinstance(rejected.get("active_node"), dict):
            failed_state, failed_node, failure_event_type = fail_active_node(
                rejected,
                agent_name=result.agent,
                reason=record.outcome.reason,
                reason_code=record.outcome.reason_code,
            )
            emit_node_lifecycle_event(
                backend_adapter,
                failed_state,
                failure_event_type,
                failed_node,
                record.outcome.reason,
                action=_ACTION_BY_AGENT[result.agent],
                metadata={
                    "candidate_id": record.candidate_id,
                    "validation_id": record.validation_id,
                    "disposition": disposition,
                },
            )
            return failed_state

        if disposition == "reject" and source_schema_version >= 6:
            return _failure(
                rejected,
                f"{result.agent} 결과를 실패 처리할 active_node가 없습니다.",
            )

        return rejected

    if disposition not in {"accept", "accept_with_limitations", "await_approval"}:
        return _failure(working, f"지원하지 않는 validation disposition입니다: {disposition}")

    if disposition == "accept_with_limitations":
        working["limitations"] = _append_unique(
            working.get("limitations", []),
            record.outcome.reason,
        )

    promoted = promote_pending_result(working, approval_granted=approval_granted)
    if result.agent == "analysis_agent":
        promoted["analysis_selection_response"] = None
        promoted["analysis_selection_review_request"] = None
    next_action = _next_action(result, record)
    promoted["next_action"] = next_action
    promoted["current_step"] = "commit_candidate"
    summaries = list(promoted.get("step_summaries", []))
    step_summary = build_step_summary(
        result,
        _ACTION_BY_AGENT[result.agent],
        next_action,
    )
    step_summary = _enrich_step_summary_with_node_finding(step_summary, backend_adapter)
    summary_payload = step_summary.model_dump(mode="json")
    if summary_payload not in summaries:
        summaries.append(summary_payload)
    promoted["step_summaries"] = summaries
    if not already_committed or approval_granted:
        _emit_event(
            backend_adapter,
            working,
            "evidence.promoted",
            f"{result.agent} 후보 근거를 승격했습니다.",
            artifact_ids=step_summary.artifact_ids,
            metadata={
                "candidate_id": record.candidate_id,
                "validation_id": record.validation_id,
            },
        )

    if isinstance(promoted.get("active_node"), dict):
        completed_state, completed_node, completion_event_type = complete_active_node(
            promoted,
            agent_name=result.agent,
            step_summary=step_summary,
        )
        emit_node_lifecycle_event(
            backend_adapter,
            completed_state,
            completion_event_type,
            completed_node,
            f"{result.agent} 작업을 완료했습니다.",
            action=_ACTION_BY_AGENT[result.agent],
            metadata={
                "candidate_id": record.candidate_id,
                "validation_id": record.validation_id,
            },
        )
        return completed_state

    if source_schema_version >= 6:
        return _failure(
            promoted,
            f"{result.agent} 결과를 완료할 active_node가 없습니다.",
        )

    return promoted


def _prepare_active_node_for_recovery_target(
    state: SupervisorState,
    record: ValidationRecord,
    target_agent: AgentName,
    target_action: NextAction,
    reason: str,
    backend_adapter: Any | None,
) -> SupervisorState:
    active_node_payload = state.get("active_node")
    if not isinstance(active_node_payload, dict):
        return state
    if active_node_payload.get("agent_name") == target_agent:
        return state

    transitioned, discarded_node, discarded_event_type = (
        discard_active_node_for_recovery(
            state,
            next_agent_name=target_agent,
            reason=reason,
        )
    )
    emit_node_lifecycle_event(
        backend_adapter,
        transitioned,
        discarded_event_type,
        discarded_node,
        reason,
        action=target_action,
        metadata={
            "candidate_id": record.candidate_id,
            "validation_id": record.validation_id,
            "from_agent": discarded_node.agent_name,
            "to_agent": target_agent,
            "reason_code": record.outcome.reason_code,
        },
    )
    return transitioned


def _fail_active_node_after_recovery(
    state: SupervisorState,
    record: ValidationRecord,
    reason: str,
    backend_adapter: Any | None,
) -> SupervisorState:
    if not isinstance(state.get("active_node"), dict):
        return state

    failed_state, failed_node, failed_event_type = fail_active_node(
        state,
        agent_name=record.agent,
        reason=reason,
        reason_code=record.outcome.reason_code or "semantic_recovery_exhausted",
    )
    emit_node_lifecycle_event(
        backend_adapter,
        failed_state,
        failed_event_type,
        failed_node,
        reason,
        action=_ACTION_BY_AGENT[record.agent],
        metadata={
            "candidate_id": record.candidate_id,
            "validation_id": record.validation_id,
            "disposition": "recover",
        },
    )
    return failed_state


def _commit_semantic_recovery(
    state: SupervisorState,
    pending: dict[str, Any],
    record: ValidationRecord,
    result: AgentCompactResult,
    backend_adapter: Any | None,
    *,
    already_committed: bool,
) -> SupervisorState:
    """Semantic 실패 후보를 격리하고 실행 가능한 1회 복구를 예약한다."""
    original_action = record.outcome.recovery_action
    semantic_check = next((check for check in record.checks if check.name == "semantic"), None)
    missing_evidence = list((semantic_check.details if semantic_check else {}).get("missing_evidence", []))
    limitations = list(state.get("limitations", []))
    if missing_evidence:
        limitations = _append_unique(
            limitations,
            f"누락 근거: {', '.join(str(item) for item in missing_evidence)}",
        )
    limitations = _append_unique(
        limitations,
        record.outcome.reason,
        f"{result.agent} 후보를 Semantic 검증 실패로 격리했습니다.",
    )

    metadata = {
        "candidate_id": record.candidate_id,
        "validation_id": record.validation_id,
        "agent": result.agent,
        "action": original_action or "",
        "attempt": 0,
    }
    recovered = reject_pending_result(
        {**state, "limitations": limitations},
        record.outcome.reason,
        metadata=metadata,
        event_type="semantic_recovery.quarantined",
    )
    recovered["pending_validation"] = None
    recovered["current_step"] = "commit_candidate"
    recovered["terminal_state"] = "running"
    if not already_committed:
        _emit_event(
            backend_adapter,
            state,
            "semantic_recovery.quarantined",
            record.outcome.reason,
            metadata=metadata,
        )

    attempts = dict(recovered.get("semantic_recovery_attempts", {}))
    if result.agent == "insight" and int(attempts.get("insight", 0)) >= 1:
        reason = "제한적 인사이트 후보가 Semantic 검증을 통과하지 못했습니다."
        recovered["limitations"] = _append_unique(recovered.get("limitations", []), reason)
        recovered = _append_recovery_event(
            recovered,
            backend_adapter,
            "semantic_recovery.budget_exhausted",
            reason,
            {
                **metadata,
                "agent": "insight",
                "action": "call_insight",
                "attempt": int(attempts.get("insight", 0)),
            },
        )
        return _finish_semantic_recovery(
            recovered,
            record,
            backend_adapter,
            original_action=original_action,
            path=[original_action] if original_action else [],
            actual_action=None,
            fallback_reason=reason,
            terminal_state=SupervisorTerminalState.failed_with_recoverable_context.value,
        )

    actual_action, path, correction_reasons, unsafe_reason = _resolve_recovery_action(
        original_action,
        recovered,
    )
    for reason in correction_reasons:
        recovered["limitations"] = _append_unique(recovered.get("limitations", []), reason)
    if len(path) > 1:
        adjustment_metadata = {
            **metadata,
            "action": actual_action or "",
            "path": path,
            "attempt": 0,
        }
        recovered = _append_recovery_event(
            recovered,
            backend_adapter,
            "semantic_recovery.precondition_adjusted",
            "Semantic 복구 대상의 선행 조건을 보정했습니다.",
            adjustment_metadata,
        )

    if actual_action is not None:
        target = _AGENT_BY_ACTION[actual_action]
        current_attempt = int(attempts.get(target, 0))
        if current_attempt == 0:
            attempts[target] = 1
            recovered["semantic_recovery_attempts"] = attempts
            recovered["next_action"] = actual_action
            recovered["terminal_state"] = "running"
            recovered = _prepare_active_node_for_recovery_target(
                recovered,
                record,
                target,
                actual_action,
                record.outcome.reason,
                backend_adapter,
            )
            recovered = _record_recovery_audit(
                recovered,
                record,
                original_action=original_action,
                path=path,
                actual_action=actual_action,
                attempt_before=0,
                attempt_after=1,
                correction_reasons=correction_reasons,
            )
            return _append_recovery_event(
                recovered,
                backend_adapter,
                "semantic_recovery.scheduled",
                f"{target} Semantic 복구를 예약했습니다.",
                {
                    **metadata,
                    "agent": target,
                    "action": actual_action,
                    "attempt": 1,
                    "path": path,
                },
            )

        budget_reason = f"{target} Semantic 복구 예산이 이미 소진되었습니다."
        recovered["limitations"] = _append_unique(recovered.get("limitations", []), budget_reason)
        recovered = _append_recovery_event(
            recovered,
            backend_adapter,
            "semantic_recovery.budget_exhausted",
            budget_reason,
            {
                **metadata,
                "agent": target,
                "action": actual_action,
                "attempt": current_attempt,
                "path": path,
            },
        )
        unsafe_reason = budget_reason

    fallback_reason = unsafe_reason or "실행 가능한 Semantic 복구 권고가 없습니다."
    recovered["limitations"] = _append_unique(recovered.get("limitations", []), fallback_reason)
    return _schedule_limited_insight_or_finish(
        recovered,
        record,
        backend_adapter,
        original_action=original_action,
        path=path,
        fallback_reason=fallback_reason,
        correction_reasons=correction_reasons,
    )


def _resolve_recovery_action(
    action: NextAction | None,
    state: SupervisorState,
) -> tuple[NextAction | None, list[str], list[str], str]:
    if action not in _AGENT_BY_ACTION:
        reason = (
            f"Semantic 복구 권고 {action!r}는 실행 대상 에이전트가 아닙니다."
            if action
            else "Semantic 복구 권고가 없습니다."
        )
        return None, [action] if action else [], [], reason

    current = action
    path: list[str] = []
    reasons: list[str] = []
    seen: set[str] = set()
    while current in _AGENT_BY_ACTION:
        if current in seen:
            return None, path, reasons, "Semantic 복구 선행 조건 보정이 반복되었습니다."
        seen.add(current)
        path.append(current)
        target = _AGENT_BY_ACTION[current]
        decision = guard_agent_preconditions(target, state)
        if decision.allowed:
            return current, path, reasons, ""
        reasons.append(decision.reason)
        if decision.next_action not in _AGENT_BY_ACTION:
            return None, path, reasons, "Semantic 복구 선행 조건을 안전하게 보정할 수 없습니다."
        current = decision.next_action
    return None, path, reasons, "Semantic 복구 권고를 실행 대상으로 변환할 수 없습니다."


def _schedule_limited_insight_or_finish(
    state: SupervisorState,
    record: ValidationRecord,
    backend_adapter: Any | None,
    *,
    original_action: NextAction | None,
    path: list[str],
    fallback_reason: str,
    correction_reasons: list[str],
) -> SupervisorState:
    attempts = dict(state.get("semantic_recovery_attempts", {}))
    has_evidence = any(
        bool(str(item.get("artifact_id") or "").strip())
        for agent in ("sql_agent", "eda_agent", "analysis_agent")
        for item in state.get("accepted_evidence", {}).get(agent, [])
        if isinstance(item, dict)
    )
    if not has_evidence:
        return _finish_semantic_recovery(
            state,
            record,
            backend_adapter,
            original_action=original_action,
            path=path,
            actual_action=None,
            fallback_reason=fallback_reason,
            terminal_state=SupervisorTerminalState.failed_terminal.value,
            correction_reasons=correction_reasons,
        )

    insight_attempt = int(attempts.get("insight", 0))
    if insight_attempt >= 1:
        reason = "제한적 인사이트 Semantic 복구 예산이 이미 소진되었습니다."
        state["limitations"] = _append_unique(state.get("limitations", []), reason)
        state = _append_recovery_event(
            state,
            backend_adapter,
            "semantic_recovery.budget_exhausted",
            reason,
            {
                "candidate_id": record.candidate_id,
                "validation_id": record.validation_id,
                "agent": "insight",
                "action": "call_insight",
                "attempt": insight_attempt,
            },
        )
        return _finish_semantic_recovery(
            state,
            record,
            backend_adapter,
            original_action=original_action,
            path=path,
            actual_action=None,
            fallback_reason=reason,
            terminal_state=SupervisorTerminalState.failed_with_recoverable_context.value,
            correction_reasons=correction_reasons,
        )

    attempts["insight"] = 1
    state["semantic_recovery_attempts"] = attempts
    state["next_action"] = "call_insight"
    state["terminal_state"] = "running"
    state["limitations"] = _append_unique(
        state.get("limitations", []),
        "승인된 기존 근거만 사용해 제한적 인사이트 fallback을 생성합니다.",
    )
    state = _prepare_active_node_for_recovery_target(
        state,
        record,
        "insight_agent",
        "call_insight_agent",
        fallback_reason,
        backend_adapter,
    )
    state = _record_recovery_audit(
        state,
        record,
        original_action=original_action,
        path=path,
        actual_action="call_insight",
        attempt_before=0,
        attempt_after=1,
        fallback_reason=fallback_reason,
        correction_reasons=correction_reasons,
    )
    return _append_recovery_event(
        state,
        backend_adapter,
        "semantic_recovery.limited_insight",
        "승인된 기존 근거로 제한적 인사이트를 예약했습니다.",
        {
            "candidate_id": record.candidate_id,
            "validation_id": record.validation_id,
            "agent": "insight",
            "action": "call_insight",
            "attempt": 1,
            "fallback_reason": fallback_reason,
        },
    )


def _finish_semantic_recovery(
    state: SupervisorState,
    record: ValidationRecord,
    backend_adapter: Any | None,
    *,
    original_action: NextAction | None,
    path: list[str | None],
    actual_action: NextAction | None,
    fallback_reason: str,
    terminal_state: str,
    correction_reasons: list[str] | None = None,
) -> SupervisorState:
    state = _fail_active_node_after_recovery(
        state,
        record,
        fallback_reason,
        backend_adapter,
    )
    finished = _record_recovery_audit(
        state,
        record,
        original_action=original_action,
        path=[item for item in path if item],
        actual_action=actual_action,
        attempt_before=None,
        attempt_after=None,
        fallback_reason=fallback_reason,
        correction_reasons=correction_reasons,
    )
    finished["terminal_state"] = terminal_state
    finished["next_action"] = "finalize"
    finished["final_answer"] = fallback_reason
    finished["error_state"] = {
        "node": "commit_candidate",
        "message": fallback_reason,
        "retryable": False,
    }
    return finished


def _record_recovery_audit(
    state: SupervisorState,
    record: ValidationRecord,
    *,
    original_action: NextAction | None,
    path: list[str],
    actual_action: NextAction | None,
    attempt_before: int | None,
    attempt_after: int | None,
    fallback_reason: str = "",
    correction_reasons: list[str] | None = None,
) -> SupervisorState:
    history = [dict(item) for item in state.get("validation_history", [])]
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        if (
            item.get("candidate_id") != record.candidate_id
            or item.get("validation_id") != record.validation_id
        ):
            continue
        updated = dict(item)
        outcome = dict(updated.get("outcome") or {})
        outcome["recovery_action"] = actual_action
        updated["outcome"] = outcome
        checks = [dict(check) for check in updated.get("checks", [])]
        semantic = next((check for check in checks if check.get("name") == "semantic"), None)
        if semantic is not None:
            details = dict(semantic.get("details") or {})
            details.update(
                {
                    "original_recovery_action": original_action,
                    "precondition_path": path,
                    "recovery_action": actual_action,
                    "budget": {
                        "agent": _AGENT_BY_ACTION.get(actual_action or ""),
                        "attempt_before": attempt_before,
                        "attempt_after": attempt_after,
                    },
                    "precondition_reasons": list(correction_reasons or []),
                    "fallback_reason": fallback_reason,
                }
            )
            semantic["details"] = details
        updated["checks"] = checks
        history[index] = updated
        break
    state["validation_history"] = history
    return state


def _append_unique(values: list[str], *items: str) -> list[str]:
    merged = [str(value) for value in values if value]
    for item in items:
        if item and item not in merged:
            merged.append(item)
    return merged


def _append_recovery_event(
    state: SupervisorState,
    backend_adapter: Any | None,
    event_type: str,
    message: str,
    metadata: dict[str, Any],
) -> SupervisorState:
    events = list(state.get("run_events", []))
    events.append({"type": event_type, **metadata})
    state["run_events"] = events
    _emit_event(backend_adapter, state, event_type, message, metadata=metadata)
    return state


def _next_action(result: AgentCompactResult, record: ValidationRecord) -> NextAction:
    if result.agent == "insight":
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
        "sql_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent"},
        "eda_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent"},
        "analysis_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent"},
        "insight": {"call_insight"},
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
