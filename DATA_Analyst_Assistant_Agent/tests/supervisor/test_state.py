from __future__ import annotations

import json

import pytest

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.state import (
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
    assert state["semantic_validation_results"] == []
    assert state["terminal_state"] == "running"


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


def test_to_orchestration_state_uses_same_generated_sql_fallback_in_plan_and_state() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )

    orchestration = to_orchestration_state(state)

    assert orchestration.generated_sql == "SELECT 1 AS sample_value"
    assert orchestration.plan is not None
    assert orchestration.plan.generated_sql == orchestration.generated_sql
    assert orchestration.route_kind == "simple"
    assert orchestration.plan.route_kind == "simple"


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

    assert normalized["state_schema_version"] == 2
    assert normalized["accepted_evidence"] == {}
    assert normalized["artifacts"] == {}
    assert normalized["completed_agents"] == []
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
