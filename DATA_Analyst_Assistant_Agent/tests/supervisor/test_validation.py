from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    empty_supervisor_state,
    merge_agent_result,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    ApprovalRequirement,
    RetryHint,
    SupervisorTerminalState,
    ValidationFinding,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    ValidationCheckResult,
    ValidationOutcome,
    ValidationRecord,
    _check_completion_readiness,
    contract_check_from_decision,
    guard_agent_preconditions,
    outcome_from_contract_decision,
    validate_subagent_result,
)


def _state():
    return empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )


def test_validation_record_uses_single_internal_contract() -> None:
    record = ValidationRecord(
        candidate_id="candidate_001",
        validation_id="validation_001",
        agent="sql_agent",
        outcome=ValidationOutcome(disposition="accept", reason="검증 통과"),
        checks=[ValidationCheckResult(name="contract", passed=True)],
    )

    payload = record.model_dump(mode="json")

    assert payload["outcome"]["disposition"] == "accept"
    assert payload["checks"] == [
        {"name": "contract", "passed": True, "findings": [], "details": {}}
    ]
    assert "valid" not in payload
    assert "decision" not in payload


def test_validation_outcome_supports_semantic_recovery_action() -> None:
    outcome = ValidationOutcome(
        disposition="recover",
        reason="필수 근거가 누락되었습니다.",
        recovery_action="call_sql_agent",
    )

    assert outcome.disposition == "recover"
    assert outcome.recovery_action == "call_sql_agent"


def test_contract_decision_is_converted_without_legacy_routing_fields() -> None:
    result = AgentCompactResult(
        agent="eda_agent",
        status="warning",
        summary="일부 제한이 있는 EDA",
        findings=[
            ValidationFinding(
                code="small_sample",
                message="표본이 작습니다.",
                source="eda_agent",
                disposition="limitation",
            )
        ],
    )
    decision = validate_subagent_result(_state(), result)

    check = contract_check_from_decision(result, decision)
    outcome = outcome_from_contract_decision(result.agent, decision)

    assert check.name == "result"
    assert check.passed is True
    assert check.findings[0].code == "small_sample"
    assert outcome.disposition == "accept_with_limitations"
    assert outcome.retry_target is None
    assert outcome.terminal_state == "running"


def test_completion_readiness_rejects_state_without_analysis_evidence() -> None:
    decision = _check_completion_readiness(_state())

    assert decision.status == "invalid"
    assert "근거" in decision.reason


def test_completion_readiness_requires_report_after_analysis_evidence() -> None:
    state = _state()
    state["accepted_evidence"] = {
        "sql_agent": [{"artifact_id": "artifact_sql"}],
    }
    state["completed_agents"] = ["sql_agent"]

    decision = _check_completion_readiness(state)

    assert decision.status == "report_required"
    assert "리포트" in decision.reason


def test_completion_readiness_rejects_report_completion_without_artifact() -> None:
    state = _state()
    state["accepted_evidence"] = {
        "analysis_agent": [{"artifact_id": "artifact_analysis"}],
    }
    state["completed_agents"] = ["analysis_agent", "report_agent"]

    decision = _check_completion_readiness(state)

    assert decision.status == "invalid"
    assert "아티팩트" in decision.reason


def test_completion_readiness_rejects_report_artifact_without_completion() -> None:
    state = _state()
    state["accepted_evidence"] = {
        "analysis_agent": [{"artifact_id": "artifact_analysis"}],
        "report_agent": [{"artifact_id": "artifact_report"}],
    }
    state["completed_agents"] = ["analysis_agent"]

    decision = _check_completion_readiness(state)

    assert decision.status == "invalid"
    assert "완료 표식" in decision.reason


def test_completion_readiness_allows_promoted_report_with_analysis_evidence() -> None:
    state = _state()
    state["accepted_evidence"] = {
        "analysis_agent": [{"artifact_id": "artifact_analysis"}],
        "report_agent": [{"artifact_id": "artifact_report"}],
    }
    state["completed_agents"] = ["analysis_agent", "report_agent"]

    decision = _check_completion_readiness(state)

    assert decision.status == "ready"
    assert "완료" in decision.reason


def test_guard_blocks_eda_without_sql_artifact() -> None:
    state = _state()

    decision = guard_agent_preconditions("eda_agent", state)

    assert decision.allowed is False
    assert decision.next_action == "call_sql_agent"


def test_guard_blocks_unknown_agent_explicitly() -> None:
    state = _state()

    decision = guard_agent_preconditions("not_real_agent", state)

    assert decision.allowed is False
    assert decision.next_action == "fail"
    assert "unknown" in decision.reason.lower() or "알 수 없는" in decision.reason


def test_guard_blocks_report_when_only_completed_agent_exists_without_artifact() -> None:
    state = _state()
    state["completed_agents"] = ["analysis_agent"]

    decision = guard_agent_preconditions("report_agent", state)

    assert decision.allowed is False
    assert decision.next_action == "call_analysis_agent"


def test_guard_allows_report_after_evidence_artifact() -> None:
    state = _state()
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["artifact_analysis"],
        ),
    )

    decision = guard_agent_preconditions("report_agent", state)

    assert decision.allowed is True
    assert decision.next_action == "call_report_agent"


def test_validate_failed_retryable_result_routes_to_same_agent() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
        error="SQL validation failed",
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "call_sql_agent"


def test_validate_approval_required_result_finalizes_for_safe_waiting_state() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="분석 결과 승인 필요",
        approval=ApprovalRequirement(required=True),
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "finalize"
    assert "승인" in decision.reason or "approval" in decision.reason.lower()


def test_validate_success_with_validation_errors_is_invalid_and_retries_same_agent() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="eda_agent",
        status="success",
        summary="EDA 완료",
        validation_errors=["row_count 검증 실패"],
        retryable=True,
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "call_eda_agent"


def test_validate_fallback_success_is_invalid_and_retryable_routes_to_same_agent() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="fallback 분석 결과",
        fallback_used=True,
        retryable=True,
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "call_analysis_agent"
    assert "fallback" in decision.reason.lower()


def test_validate_fallback_success_without_retry_fails() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="fallback 분석 결과",
        fallback_used=True,
        retryable=False,
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "fail"
    assert "fallback" in decision.reason.lower()


def test_validate_success_with_only_validation_warnings_requests_supervisor_redecision() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="eda_agent",
        status="warning",
        summary="EDA 경고 포함 완료",
        validation_warnings=["표본 수가 적습니다"],
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is True
    assert decision.next_action == "decide_next_action"
    assert "다음 행동 재판단" in decision.reason


def test_validate_success_requests_supervisor_redecision() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 실행 완료",
        artifact_ids=["artifact_sql"],
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is True
    assert decision.next_action == "decide_next_action"


def test_validate_retry_boundary_allows_before_limit_and_fails_at_limit() -> None:
    state = _state()
    state["max_retry_per_agent"] = 2
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
        error="SQL validation failed",
    )

    state["retry_counts"] = {"sql_agent": 1}
    retry_decision = validate_subagent_result(state, result)

    state["retry_counts"] = {"sql_agent": 2}
    fail_decision = validate_subagent_result(state, result)

    assert retry_decision.valid is False
    assert retry_decision.next_action == "call_sql_agent"
    assert fail_decision.valid is False
    assert fail_decision.next_action == "fail"


def test_validate_retryable_failure_fails_after_retry_limit() -> None:
    state = _state()
    state["retry_counts"] = {"sql_agent": 1}
    state["max_retry_per_agent"] = 1
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
        error="SQL validation failed",
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "fail"


def test_validate_report_success_finalizes() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="report_agent",
        status="success",
        summary="리포트 생성 완료",
        artifact_ids=["artifact_report"],
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is True
    assert decision.next_action == "finalize"


def test_validate_report_success_without_artifact_is_invalid() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="report_agent",
        status="success",
        summary="리포트 생성 완료",
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "fail"


def test_blocking_finding_takes_priority_over_approval() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="승인과 오류가 함께 존재",
        findings=[
            ValidationFinding(
                code="invalid_schema",
                source="local_check",
                severity="error",
                disposition="blocking",
                message="스키마가 잘못되었습니다.",
            )
        ],
        approval=ApprovalRequirement(required=True, reason="실행 승인 필요"),
    )

    decision = validate_subagent_result(state, result)

    assert decision.decision == "reject"
    assert decision.next_action == "fail"


def test_retry_required_warning_routes_to_retry_instead_of_success() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="조인 재검증 필요",
        findings=[
            ValidationFinding(
                code="invalid_join_plan",
                source="sql_langgraph",
                severity="warning",
                disposition="retry_required",
                message="조인 계획을 다시 생성해야 합니다.",
                retryable=True,
            )
        ],
    )

    decision = validate_subagent_result(state, result)

    assert decision.decision == "retry"
    assert decision.next_action == "call_sql_agent"


def test_explicit_failure_takes_priority_over_blocking_finding_and_approval() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="failed",
        summary="분석 실패",
        findings=[
            ValidationFinding(
                code="blocking_output",
                source="analysis_agent",
                severity="error",
                disposition="blocking",
                message="blocking finding",
            )
        ],
        retry_hint=RetryHint(
            retryable=True,
            reason_code="method_review_failed",
            details={"failure_reason": "wrong method"},
        ),
        approval=ApprovalRequirement(required=True, reason="승인 필요"),
    )

    decision = validate_subagent_result(state, result)

    assert decision.decision == "retry"
    assert decision.next_action == "call_analysis_agent"
    assert decision.reason_code == "method_review_failed"
    assert decision.failure_reason == "wrong method"
    assert decision.repeated_failure is False
    assert decision.failure_streak == {
        "reason_code": "method_review_failed",
        "failure_reason": "wrong method",
        "signature": '["method_review_failed", "wrong method"]',
        "consecutive_count": 1,
    }


def test_failure_reason_uses_error_then_summary_then_unknown_fallback() -> None:
    state = _state()
    results = [
        AgentCompactResult(
            agent="sql_agent",
            status="failed",
            summary="summary reason",
            error="error reason",
        ),
        AgentCompactResult(agent="sql_agent", status="failed", summary="summary reason"),
        AgentCompactResult(
            agent="sql_agent",
            status="failed",
            summary="",
            retry_hint=RetryHint(reason_code=""),
        ),
    ]

    decisions = [validate_subagent_result(state, result) for result in results]

    assert [decision.failure_reason for decision in decisions] == [
        "error reason",
        "summary reason",
        "알 수 없는 실패",
    ]
    assert all(decision.reason_code == "none" for decision in decisions)


def test_same_normalized_failure_signature_terminates_analysis_with_recoverable_context() -> None:
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql"],
        ),
    )
    state["max_retry_per_agent"] = 10
    state["failure_streaks"] = {
        "analysis_agent": {
            "reason_code": "METHOD_REVIEW_FAILED",
            "failure_reason": "Wrong Method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        }
    }
    result = AgentCompactResult(
        agent="analysis_agent",
        status="failed",
        summary="분석 실패",
        retry_hint=RetryHint(
            retryable=True,
            reason_code="METHOD_REVIEW_FAILED",
            details={"failure_reason": "  WRONG   method  "},
        ),
    )

    decision = validate_subagent_result(state, result)

    assert decision.repeated_failure is True
    assert decision.failure_streak is not None
    assert decision.failure_streak["consecutive_count"] == 2
    assert decision.next_action == "fail"
    assert decision.terminal_state == SupervisorTerminalState.failed_with_recoverable_context.value
    assert decision.reason == (
        "analysis_agent 실패 [METHOD_REVIEW_FAILED]: WRONG   method (repeated_failure=true)"
    )


def test_same_failure_with_different_reason_code_starts_new_streak() -> None:
    state = _state()
    state["failure_streaks"] = {
        "analysis_agent": {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        }
    }
    result = AgentCompactResult(
        agent="analysis_agent",
        status="failed",
        summary="분석 실패",
        retry_hint=RetryHint(
            retryable=True,
            reason_code="data_quality_failed",
            details={"failure_reason": "wrong method"},
        ),
    )

    decision = validate_subagent_result(state, result)

    assert decision.repeated_failure is False
    assert decision.next_action == "call_analysis_agent"
    assert decision.failure_streak is not None
    assert decision.failure_streak["consecutive_count"] == 1


def test_repeated_analysis_failure_ignores_analysis_and_pending_artifacts() -> None:
    state = _state()
    state["accepted_evidence"] = {
        "analysis_agent": [{"artifact_id": "prior_analysis"}],
    }
    state["artifacts"] = dict(state["accepted_evidence"])
    state["failure_streaks"] = {
        "analysis_agent": {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        }
    }
    result = AgentCompactResult(
        agent="analysis_agent",
        status="failed",
        summary="분석 실패",
        artifact_ids=["failed_candidate_artifact"],
        retry_hint=RetryHint(
            retryable=True,
            reason_code="method_review_failed",
            details={"failure_reason": "wrong method"},
        ),
    )

    decision = validate_subagent_result(state, result)

    assert decision.terminal_state == SupervisorTerminalState.failed_terminal.value


def test_approval_required_status_without_required_flag_is_contract_failure() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="승인 필요",
        approval=ApprovalRequirement(required=False),
    )

    decision = validate_subagent_result(state, result)

    assert decision.decision == "reject"
    assert decision.next_action == "fail"
    assert decision.reason_code == "approval_contract_mismatch"
    assert decision.failure_reason == "status=approval_required이지만 approval.required=false입니다."
    assert decision.terminal_state == SupervisorTerminalState.failed_terminal.value


def test_legacy_approval_required_status_with_required_flag_waits_for_approval() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="승인 필요",
        approval=ApprovalRequirement(required=True, reason="분석 승인 필요"),
    )

    decision = validate_subagent_result(state, result)

    assert decision.decision == "await_approval"
    assert decision.next_action == "finalize"
