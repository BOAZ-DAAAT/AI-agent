from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    artifact_ids_by_agent,
    empty_supervisor_state,
    merge_agent_result,
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
