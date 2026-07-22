from __future__ import annotations

import json

import pytest

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AnalysisPlan,
    ApprovalRequirement,
    SupervisorTerminalState,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    ActiveNodeExecution,
    AgentCompactResult,
    ArtifactSummary,
    artifact_ids_by_agent,
    empty_supervisor_state,
    merge_agent_result,
    normalize_supervisor_state,
    promote_pending_result,
    stage_candidate_result,
    to_orchestration_state,
)


def test_empty_supervisor_state_uses_compact_defaults() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )

    assert state["thread_id"] == "thread_sales_001"
    assert state["current_run_id"] == "run_001"
    assert state["run_ids"] == ["run_001"]
    assert state["latest_user_query"] == "월별 매출 추이를 분석해줘"
    assert state["user_turns"] == [{"run_id": "run_001", "query": "월별 매출 추이를 분석해줘"}]
    assert state["agent_results"] == []
    assert state["artifacts"] == {}
    assert state["last_agent_result"] == {}
    assert state["validation_history"] == []
    assert "validation_results" not in state
    assert "evidence_validation_results" not in state
    assert "semantic_validation_results" not in state
    assert state["state_schema_version"] == 7
    assert state["retrieval_query"] == ""
    assert state["retrieval_query_generation"] == {"status": "not_started"}
    assert state["clarification_answers"] == []
    assert state["analysis_rule_context"] is None
    assert state["analysis_rule_retrieval"] == {"status": "not_started"}
    assert state["active_node"] is None
    assert state["last_completed_node_id"] is None
    assert state["node_sequence"] == 0
    assert state["analysis_selection_response"] is None
    assert state["analysis_selection_review_request"] is None
    assert state["analysis_review_decisions"] == []
    assert state["semantic_recovery_attempts"] == {}
    assert state["limitations"] == []
    assert state["failure_streaks"] == {}
    assert state["terminal_state"] == "running"


def test_active_node_execution_tracks_retry_attempt_without_changing_node() -> None:
    node = ActiveNodeExecution(
        node_id="run_001:node:1",
        agent_name="sql_agent",
        parent_node_id=None,
        node_sequence=1,
        attempt=3,
    )

    assert node.node_id == "run_001:node:1"
    assert node.agent_name == "sql_agent"
    assert node.node_sequence == 1
    assert node.attempt == 3


def test_normalize_v2_validation_arrays_into_v4_history() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state.update(
        {
            "state_schema_version": 2,
            "validation_history": [],
            "validation_results": [
                {
                    "agent": "analysis_agent",
                    "valid": True,
                    "decision": "accept",
                    "reason": "결정론 검증 통과",
                }
            ],
            "evidence_validation_results": [
                {"valid": True, "decision": "accept", "content_hashes": {"a1": "hash"}}
            ],
            "semantic_validation_results": [
                {
                    "semantic_valid": True,
                    "severity": "info",
                    "reason": "의미 검증 통과",
                    "recommended_next_action": "finalize",
                }
            ],
        }
    )

    normalized = normalize_supervisor_state(state)

    assert normalized["state_schema_version"] == 7
    assert normalized["analysis_selection_response"] is None
    assert normalized["analysis_selection_review_request"] is None
    assert normalized["analysis_review_decisions"] == []
    assert normalized["semantic_recovery_attempts"] == {}
    assert normalized["limitations"] == []
    assert "validation_results" not in normalized
    assert "evidence_validation_results" not in normalized
    assert "semantic_validation_results" not in normalized
    record = normalized["validation_history"][0]
    assert record["candidate_id"] == "legacy_validation_0"
    assert record["validation_id"] == "legacy_validation_0"
    assert record["agent"] == "analysis_agent"
    assert [check["name"] for check in record["checks"]] == ["result", "evidence", "semantic"]
    assert record["outcome"]["disposition"] == "accept"


def test_normalize_v2_pending_approval_preserves_candidate_hashes() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state.update(
        {
            "state_schema_version": 2,
            "pending_approval": {
                "approval_id": "approval_001",
                "agent": "analysis_agent",
                "reason": "승인 필요",
                "approval_type": "human_review",
                "candidate_id": "candidate_001",
                "validation_id": "validation_001",
                "content_hashes": {"artifact_001": "hash_001"},
            },
            "pending_result": {
                "candidate_id": "candidate_001",
                "validation_id": "validation_001",
                "result": {
                    "agent": "analysis_agent",
                    "status": "approval_required",
                    "summary": "승인 필요",
                    "approval": {"required": True},
                },
                "state_updates": {},
                "content_hashes": {"artifact_001": "hash_001"},
            },
        }
    )

    normalized = normalize_supervisor_state(state)

    assert normalized["pending_approval"] == {
        **state["pending_approval"],
        "review_request": None,
        "expected_resume": {"approved": "boolean"},
    }
    assert normalized["pending_result"] == state["pending_result"]
    assert normalized["state_schema_version"] == 7


def test_normalize_v3_checkpoint_adds_semantic_recovery_fields_without_losing_history() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["state_schema_version"] = 3
    state["semantic_retry_counts"] = {"candidate_001": 1}
    state["validation_history"] = [
        {
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "agent": "analysis_agent",
            "outcome": {"disposition": "accept", "reason": "기존 검증"},
            "checks": [],
        }
    ]

    normalized = normalize_supervisor_state(state)

    assert normalized["state_schema_version"] == 7
    assert normalized["semantic_retry_counts"] == {"candidate_001": 1}
    assert normalized["semantic_recovery_attempts"] == {}
    assert normalized["limitations"] == []
    assert normalized["validation_history"] == state["validation_history"]


def test_normalize_v4_analysis_approval_does_not_retrofit_native_review() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state["state_schema_version"] = 4
    state["analysis_selection_response"] = {"selected_option_id": "legacy"}
    state["analysis_selection_review_request"] = {"question": "legacy"}
    state["analysis_review_decisions"] = [{"approval_id": "legacy"}]
    state["pending_approval"] = {
        "approval_id": "approval_legacy",
        "agent": "analysis_agent",
        "reason": "기존 승인",
        "approval_type": "analysis.review",
        "review_request": {"question": "legacy"},
        "expected_resume": {"approval_id": "string"},
    }

    normalized = normalize_supervisor_state(state)

    assert normalized["state_schema_version"] == 7
    assert normalized["analysis_selection_response"] is None
    assert normalized["analysis_selection_review_request"] is None
    assert normalized["analysis_review_decisions"] == []
    assert normalized["pending_approval"]["review_request"] is None
    assert normalized["pending_approval"]["expected_resume"] == {"approved": "boolean"}


def test_normalize_v6_checkpoint_adds_retrieval_fields() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="매출 분석",
        datasource_id=None,
    )
    state["state_schema_version"] = 6
    state.pop("retrieval_query")
    state.pop("retrieval_query_generation")
    state.pop("clarification_answers")
    state["analysis_rule_retrieval"] = {"status": "success", "hit_count": 3}

    normalized = normalize_supervisor_state(state)

    assert normalized["state_schema_version"] == 7
    assert normalized["retrieval_query"] == ""
    assert normalized["retrieval_query_generation"] == {"status": "not_started"}
    assert normalized["clarification_answers"] == []
    assert normalized["analysis_rule_retrieval"] == {"status": "success", "hit_count": 3}


def test_normalize_v6_checkpoint_preserves_retrieval_and_clarification_fields() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="seller별 리뷰와 배송 관계 분석",
        datasource_id=None,
    )
    state["state_schema_version"] = 6
    state["clarification_answers"] = ["단일 판매자 기준으로 해줘"]
    state["retrieval_query"] = "단일 판매자 기준 seller 리뷰 배송 분석"
    state["retrieval_query_generation"] = {"status": "success", "reason": "user clarified"}

    normalized = normalize_supervisor_state(state)

    assert normalized["state_schema_version"] == 7
    assert normalized["clarification_answers"] == ["단일 판매자 기준으로 해줘"]
    assert normalized["retrieval_query"] == "단일 판매자 기준 seller 리뷰 배송 분석"
    assert normalized["retrieval_query_generation"] == {
        "status": "success",
        "reason": "user clarified",
    }


def test_to_orchestration_state_merges_state_and_result_limitations_without_duplicates() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["limitations"] = ["표본이 작습니다.", "후보를 격리했습니다."]
    state["agent_results"] = [
        {
            "agent": "analysis_agent",
            "status": "warning",
            "summary": "제한적 분석",
            "findings": [
                {
                    "code": "small_sample",
                    "source": "analysis_agent",
                    "severity": "warning",
                    "disposition": "limitation",
                    "message": "표본이 작습니다.",
                }
            ],
        }
    ]

    orchestration = to_orchestration_state(state)

    assert orchestration.limitations == ["표본이 작습니다.", "후보를 격리했습니다."]


def test_merge_agent_result_adds_artifacts_and_completion() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 실행 완료",
        artifact_ids=["artifact_sql_result"],
        artifacts=[
            ArtifactSummary(
                artifact_id="artifact_sql_result",
                type="sql_result",
                kind="sql_result",
                summary="10 rows",
            )
        ],
    )

    merged = merge_agent_result(state, result)

    assert merged["completed_agents"] == ["sql_agent"]
    assert merged["failed_agents"] == []
    assert merged["agent_results"][0]["summary"] == "SQL 실행 완료"
    assert merged["artifacts"]["sql_agent"][0]["artifact_id"] == "artifact_sql_result"


def test_agent_compact_result_preserves_fallback_contract_field() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="fallback 분석 결과",
        fallback_used=True,
    )

    assert result.fallback_used is True
    assert result.model_dump(mode="json")["fallback_used"] is True


def test_to_orchestration_state_preserves_existing_agent_artifacts() -> None:
    catalog_summary = {"tables": [{"name": "sales", "columns": ["month", "amount"]}]}
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
        catalog_summary=catalog_summary,
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 실행 완료",
        artifact_ids=["artifact_sql_result"],
        artifacts=[
            ArtifactSummary(
                artifact_id="artifact_sql_summary_only",
                type="sql_validation",
                kind="validation",
                summary="artifact_ids 목록에는 없고 summary에만 있는 산출물",
            )
        ],
    )
    state = merge_agent_result(state, result)
    state["generated_sql"] = "SELECT 1 AS sample_value"
    state["terminal_state"] = SupervisorTerminalState.completed.value
    state["retry_counts"] = {"sql_agent": 1}
    state["max_retry_per_agent"] = 3

    orchestration = to_orchestration_state(state)

    assert orchestration.run_id == "run_001"
    assert orchestration.thread_id == "thread_sales_001"
    assert orchestration.datasource_id == "ds_001"
    assert orchestration.user_query == "월별 매출 추이를 분석해줘"
    assert artifact_ids_by_agent(state) == {
        "sql_agent": ["artifact_sql_result", "artifact_sql_summary_only"]
    }
    assert orchestration.artifact_ids == {
        "sql_agent": ["artifact_sql_result", "artifact_sql_summary_only"]
    }
    assert orchestration.catalog_summary == catalog_summary
    assert orchestration.completed_agents == ["sql_agent"]
    assert orchestration.retry_context == {"sql_agent": 1}
    assert orchestration.retry_counts == {"sql_agent": 1}
    assert orchestration.max_retry_per_agent == 3
    assert orchestration.generated_sql == "SELECT 1 AS sample_value"
    assert orchestration.terminal_state == SupervisorTerminalState.completed


def test_merge_agent_result_marks_approval_required_as_pending_not_completed() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="approval_required",
        summary="데이터마트 사용 승인이 필요합니다",
        approval=ApprovalRequirement(required=True),
    )

    merged = merge_agent_result(state, result)

    assert merged["completed_agents"] == []
    assert merged["pending_approval"]["approval_id"] == "run_001:sql_agent:approval"
    assert merged["pending_approval"]["agent"] == "sql_agent"
    assert merged["pending_approval"]["reason"] == "데이터마트 사용 승인이 필요합니다"
    assert merged["pending_approval"]["candidate_id"]
    assert merged["pending_approval"]["validation_id"]
    assert merged["terminal_state"] == SupervisorTerminalState.needs_user_approval.value


def test_to_orchestration_state_rejects_invalid_terminal_state() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["terminal_state"] = "not_a_real_terminal_state"

    with pytest.raises(ValueError, match="Invalid supervisor terminal_state"):
        to_orchestration_state(state)


def test_analysis_plan_sql_defaults_are_empty() -> None:
    plan = AnalysisPlan(goal="월별 매출 추이 분석")

    assert plan.generated_sql == ""
    assert plan.source_sql == ""
    assert plan.query_rules == {}


def test_to_orchestration_state_preserves_supervisor_plan_intent_fields() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="고객별 RFM과 리뷰 점수를 결합해줘",
        datasource_id="ds_001",
    )
    state["analysis_plan"] = {
        "goal": "고객별 RFM 리뷰 결합 마트 생성",
        "route_kind": "comprehensive",
        "planner_mode": "llm",
        "metric": "RFM 및 평균 리뷰 점수",
        "dimension": "customer_unique_id",
        "filters": ["Monetary 상위 25%", "평균 리뷰 점수 하위 25%"],
        "requires_mart_review": True,
        "query_rules": {
            "document_id": "customer_value",
            "entity_grain": ["customer_unique_id 기준"],
        },
    }

    orchestration = to_orchestration_state(state)

    assert orchestration.plan is not None
    assert orchestration.plan.route_kind == "comprehensive"
    assert orchestration.plan.metric == "RFM 및 평균 리뷰 점수"
    assert orchestration.plan.dimension == "customer_unique_id"
    assert orchestration.plan.filters == ["Monetary 상위 25%", "평균 리뷰 점수 하위 25%"]
    assert orchestration.plan.requires_mart_review is True
    assert orchestration.plan.query_rules == {
        "document_id": "customer_value",
        "entity_grain": ["customer_unique_id 기준"],
    }


def test_to_orchestration_state_without_plan_preserves_failure_context() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state.update(
        {
            "terminal_state": SupervisorTerminalState.failed_terminal.value,
            "final_answer": "계획 생성에 실패했습니다.",
            "error_state": {
                "message": "플래너 응답을 해석할 수 없습니다.",
                "retryable": False,
            },
            "retry_counts": {"planner": 1},
        }
    )

    orchestration = to_orchestration_state(state)

    assert orchestration.final_answer == "계획 생성에 실패했습니다."
    assert orchestration.generated_sql == ""
    assert orchestration.plan is None
    assert orchestration.goal == "월별 매출 추이를 분석해줘"
    assert orchestration.route_kind == "simple"
    assert orchestration.planner_mode == "deterministic"
    assert orchestration.error_state == {
        "message": "플래너 응답을 해석할 수 없습니다.",
        "retryable": False,
    }
    assert orchestration.retry_context == {
        "planner": 1,
        "last_error": "플래너 응답을 해석할 수 없습니다.",
        "retryable": False,
    }


def test_to_orchestration_state_clears_plan_sql_when_sql_was_not_generated() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["analysis_plan"] = {
        "goal": "월별 매출 추이 분석",
        "route_kind": "trend",
        "planner_mode": "llm",
        "generated_sql": "SELECT sample FROM placeholder",
        "source_sql": "SELECT raw FROM sales",
    }

    orchestration = to_orchestration_state(state)

    assert orchestration.plan is not None
    assert orchestration.generated_sql == ""
    assert orchestration.plan.generated_sql == ""
    assert orchestration.plan.source_sql == ""
    assert orchestration.goal == "월별 매출 추이 분석"
    assert orchestration.route_kind == "trend"
    assert orchestration.planner_mode == "llm"


@pytest.mark.parametrize(
    ("source_sql", "expected_source_sql"),
    [
        ("SELECT raw_amount FROM sales", "SELECT raw_amount FROM sales"),
        ("", "SELECT month, SUM(amount) FROM sales GROUP BY month"),
    ],
)
def test_to_orchestration_state_uses_generated_sql_and_preserves_source_sql(
    source_sql: str,
    expected_source_sql: str,
) -> None:
    generated_sql = "SELECT month, SUM(amount) FROM sales GROUP BY month"
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["analysis_plan"] = {
        "goal": "월별 매출 추이 분석",
        "source_sql": source_sql,
    }
    state["generated_sql"] = generated_sql

    orchestration = to_orchestration_state(state)

    assert orchestration.plan is not None
    assert orchestration.generated_sql == generated_sql
    assert orchestration.plan.generated_sql == generated_sql
    assert orchestration.plan.source_sql == expected_source_sql


def test_to_orchestration_state_exposes_pending_approval_id() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="approval_required",
            summary="SQL 실행 승인 필요",
            approval=ApprovalRequirement(required=True),
        ),
    )

    orchestration = to_orchestration_state(state)

    assert orchestration.approval_ids == ["run_001:sql_agent:approval"]


def test_supervisor_state_defaults_and_merged_results_are_json_serializable() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    json.dumps(state, ensure_ascii=False)

    merged = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 실행 완료",
            artifact_ids=["artifact_sql_result"],
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_sql_result",
                    type="sql_result",
                    kind="sql_result",
                    summary="10 rows",
                )
            ],
        ),
    )

    json.dumps(merged, ensure_ascii=False)


def test_empty_supervisor_state_rejects_non_json_serializable_catalog_summary() -> None:
    with pytest.raises(ValueError, match="SupervisorState must be JSON serializable"):
        empty_supervisor_state(
            thread_id="thread_sales_001",
            run_id="run_001",
            user_query="월별 매출 추이를 분석해줘",
            datasource_id="ds_001",
            catalog_summary={"bad": object()},
        )


def test_success_result_clears_pending_approval_for_same_agent() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="approval_required",
            summary="데이터마트 사용 승인이 필요합니다",
            approval=ApprovalRequirement(required=True),
        ),
    )

    merged = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="승인 후 SQL 실행 완료",
        ),
    )

    assert merged["pending_approval"] is None
    assert merged["terminal_state"] == "running"
    assert merged["completed_agents"] == ["sql_agent"]


def test_merge_agent_result_rejects_non_json_serializable_existing_state() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["catalog_summary"] = {"bad": object()}

    with pytest.raises(ValueError, match="SupervisorState must be JSON serializable"):
        merge_agent_result(
            state,
            AgentCompactResult(
                agent="sql_agent",
                status="success",
                summary="SQL 실행 완료",
            ),
        )


def test_stage_candidate_does_not_pollute_operational_state_until_promotion() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["artifact_sql"],
        artifacts=[ArtifactSummary(artifact_id="artifact_sql", type="sql_result", kind="sql_result")],
    )

    staged = stage_candidate_result(
        state,
        result,
        {"generated_sql": "SELECT 42", "error_state": {"warning": "candidate only"}},
    )

    assert staged["pending_result"]["result"]["summary"] == "SQL 완료"
    assert staged["artifacts"] == {}
    assert staged["completed_agents"] == []
    assert staged["generated_sql"] == ""
    assert staged["error_state"] == {}

    promoted = promote_pending_result(staged)

    assert promoted["pending_result"] is None
    assert promoted["completed_agents"] == ["sql_agent"]
    assert promoted["generated_sql"] == "SELECT 42"
    assert promoted["accepted_evidence"]["sql_agent"][0]["artifact_id"] == "artifact_sql"


def test_normalize_v1_checkpoint_quarantines_legacy_artifacts_without_accepting_them() -> None:
    legacy = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    legacy.pop("state_schema_version")
    legacy["artifacts"] = {
        "sql_agent": [{"artifact_id": "legacy_sql", "type": "sql_result", "kind": "sql_result"}]
    }
    legacy["completed_agents"] = ["sql_agent"]

    normalized = normalize_supervisor_state(legacy)
    normalized_twice = normalize_supervisor_state(normalized)

    assert normalized["state_schema_version"] == 7
    assert normalized["accepted_evidence"] == {}
    assert normalized["artifacts"] == {}
    assert normalized["completed_agents"] == []
    assert normalized["failure_streaks"] == {}
    assert normalized["quarantined_artifacts"][0]["artifact_id"] == "legacy_sql"
    assert normalized_twice == normalized


def test_normalize_v2_rebuilds_compatibility_projections_from_accepted_evidence() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="매출 요약",
        datasource_id=None,
    )
    state["artifacts"] = {"sql_agent": [{"artifact_id": "unvalidated"}]}
    state["completed_agents"] = ["sql_agent"]

    normalized = normalize_supervisor_state(state)

    assert normalized["artifacts"] == {}
    assert normalized["completed_agents"] == []
    assert normalized["failure_streaks"] == {}


def test_to_orchestration_state_exposes_analysis_last_failure_in_plan_and_state() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["failure_streaks"] = {
        "analysis_agent": {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        }
    }
    state["analysis_plan"] = {"goal": "월별 매출 추이 분석"}

    orchestration = to_orchestration_state(state)

    expected = {
        "last_failure": {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
        }
    }
    assert orchestration.retry_context == expected
    assert orchestration.plan is not None
    assert orchestration.plan.retry_context == expected


def test_to_orchestration_state_exposes_agent_feedback_from_validation_history() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="상위 20% 고객군과 일반 고객군의 만족도 분포 차이를 검정해줘",
        datasource_id="ds_001",
    )
    state["analysis_plan"] = {"goal": "고객 세그먼트별 만족도 분포 비교"}
    state["validation_history"] = [
        {
            "candidate_id": "cand_1",
            "validation_id": "val_1",
            "agent": "eda_agent",
            "outcome": {
                "disposition": "recover",
                "reason": "그룹별 분포 비교 없이 컬럼 나열만 반복함",
                "reason_code": "semantic_validation_failed",
            },
            "checks": [
                {
                    "name": "semantic",
                    "passed": False,
                    "findings": [],
                    "details": {"missing_evidence": ["상위20% vs 일반군 기술통계", "유의성 검정 결과"]},
                }
            ],
        },
        {
            "candidate_id": "cand_2",
            "validation_id": "val_2",
            "agent": "sql_agent",
            "outcome": {
                "disposition": "accept",
                "reason": "정합성 통과",
                "reason_code": "none",
            },
            "checks": [],
        },
    ]

    orchestration = to_orchestration_state(state)

    expected_feedback = {
        "eda_agent": {
            "reason": "그룹별 분포 비교 없이 컬럼 나열만 반복함",
            "missing_evidence": ["상위20% vs 일반군 기술통계", "유의성 검정 결과"],
            "source": "semantic",
        }
    }
    assert orchestration.retry_context["agent_feedback"] == expected_feedback
    assert orchestration.plan is not None
    assert orchestration.plan.retry_context["agent_feedback"] == expected_feedback
    # accept 판정은 재시도 사유가 아니므로 피드백에 섞이면 안 된다
    assert "sql_agent" not in orchestration.retry_context["agent_feedback"]


def test_to_orchestration_state_agent_feedback_keeps_latest_record_per_agent() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="고객 단위 RFM 세그먼트를 분석해줘",
        datasource_id="ds_001",
    )
    state["analysis_plan"] = {"goal": "고객 단위 RFM"}
    state["validation_history"] = [
        {
            "candidate_id": "cand_1",
            "validation_id": "val_1",
            "agent": "sql_agent",
            "outcome": {"disposition": "recover", "reason": "1차: 주문 단위로 생성됨", "reason_code": "x"},
            "checks": [{"name": "semantic", "passed": False, "findings": [], "details": {"missing_evidence": []}}],
        },
        {
            "candidate_id": "cand_2",
            "validation_id": "val_2",
            "agent": "sql_agent",
            "outcome": {"disposition": "reject", "reason": "2차: 여전히 주문 단위", "reason_code": "y"},
            "checks": [],
        },
    ]

    orchestration = to_orchestration_state(state)

    feedback = orchestration.retry_context["agent_feedback"]["sql_agent"]
    assert feedback["reason"] == "2차: 여전히 주문 단위"
    assert feedback["source"] == "hard_failure"


def test_success_promotion_clears_only_matching_agent_failure_streak() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["failure_streaks"] = {
        "analysis_agent": {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        },
        "sql_agent": {
            "reason_code": "timeout",
            "failure_reason": "timeout",
            "signature": '["timeout", "timeout"]',
            "consecutive_count": 1,
        },
    }

    promoted = merge_agent_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="warning",
            summary="제한사항 포함 분석 완료",
        ),
    )

    assert "analysis_agent" not in promoted["failure_streaks"]
    assert promoted["failure_streaks"]["sql_agent"]["reason_code"] == "timeout"


def test_approval_pending_keeps_streak_and_legacy_approval_promotes_as_success() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    state["failure_streaks"] = {
        "analysis_agent": {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        }
    }
    pending = merge_agent_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="approval_required",
            summary="레거시 분석 승인 필요",
            approval=ApprovalRequirement(required=True, reason="검토 필요"),
        ),
    )

    assert "analysis_agent" in pending["failure_streaks"]

    promoted = promote_pending_result(pending, approval_granted=True)

    assert promoted["agent_results"][-1]["status"] == "success"
    assert promoted["completed_agents"] == ["analysis_agent"]
    assert "analysis_agent" not in promoted["failure_streaks"]
    assert promoted["pending_approval"] is None


def test_promote_legacy_approval_requires_explicit_grant() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    staged = stage_candidate_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="approval_required",
            summary="레거시 분석 승인 필요",
            approval=ApprovalRequirement(required=True),
        ),
    )

    with pytest.raises(ValueError, match="승인"):
        promote_pending_result(staged)


def test_merge_agent_result_rejects_approval_contract_mismatch() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )

    with pytest.raises(ValueError, match="approval.required=false"):
        merge_agent_result(
            state,
            AgentCompactResult(
                agent="analysis_agent",
                status="approval_required",
                summary="잘못된 승인 계약",
                approval=ApprovalRequirement(required=False),
            ),
        )


def test_merge_success_with_required_approval_waits_without_promoting() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )

    pending = merge_agent_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료, 승인 필요",
            approval=ApprovalRequirement(required=True, reason="인과 해석 검토"),
        ),
    )

    assert pending["completed_agents"] == []
    assert pending["pending_approval"]["reason"] == "인과 해석 검토"
    assert pending["terminal_state"] == SupervisorTerminalState.needs_user_approval.value
