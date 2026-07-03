from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    empty_supervisor_state,
    merge_agent_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    guard_agent_preconditions,
    validate_subagent_result,
)


def _state():
    return empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )


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


def test_validate_success_with_only_validation_warnings_is_valid() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="eda_agent",
        status="warning",
        summary="EDA 경고 포함 완료",
        validation_warnings=["표본 수가 적습니다"],
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is True
    assert decision.next_action == "create_plan"


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
