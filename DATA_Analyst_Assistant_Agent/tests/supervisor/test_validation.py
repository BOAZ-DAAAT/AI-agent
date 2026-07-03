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
