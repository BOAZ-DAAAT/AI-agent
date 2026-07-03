from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentToolResult


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class SequencedDecisionModel:
    def __init__(self, actions: list[str] | None = None) -> None:
        self.actions = iter(
            actions
            or [
                "call_sql_agent",
                "call_eda_agent",
                "call_analysis_agent",
                "call_report_agent",
                "finalize",
            ]
        )

    def invoke(self, messages: list[dict[str, str]]) -> FakeMessage:
        action = next(self.actions)
        return FakeMessage(f'{{"next_action":"{action}","reason":"테스트 모델 결정"}}')


class FakeSubAgentAdapter:
    def __init__(self, results: dict[str, AgentToolResult] | None = None) -> None:
        self.results = results or {}
        self.calls: list[str] = []

    def call(self, agent_name: str, state: dict[str, Any]) -> AgentToolResult:
        self.calls.append(agent_name)
        if agent_name in self.results:
            return self.results[agent_name]
        return AgentToolResult(
            agent_result=AgentCompactResult(
                agent=agent_name,
                status="success",
                summary=f"{agent_name} 완료",
                artifact_ids=[f"artifact_{agent_name}"],
                artifacts=[
                    ArtifactSummary(
                        artifact_id=f"artifact_{agent_name}",
                        type="test_artifact",
                        kind=agent_name,
                        summary=f"{agent_name} 산출물 요약",
                    )
                ],
            ),
            state_updates={
                "generated_sql": "SELECT 1 AS sample_value" if agent_name == "sql_agent" else "",
            },
        )


def _state(user_query: str = "월별 매출 추이를 분석해줘") -> dict[str, Any]:
    return empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query=user_query,
        datasource_id=None,
    )


def test_supervisor_graph_runs_all_subagents_and_finalizes() -> None:
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=SequencedDecisionModel())

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "eda_agent", "analysis_agent", "report_agent"]
    assert result["final_answer"] == "최종 리포트 생성이 완료되었습니다."
    assert result["last_agent_result"] == {
        "agent": "report_agent",
        "status": "success",
        "summary": "report_agent 완료",
        "artifact_ids": ["artifact_report_agent"],
        "artifacts": [
            {
                "artifact_id": "artifact_report_agent",
                "type": "test_artifact",
                "kind": "report_agent",
                "summary": "report_agent 산출물 요약",
                "uri": None,
            }
        ],
        "validation_errors": [],
        "validation_warnings": [],
        "fallback_used": False,
        "retryable": False,
        "error": "",
    }


def test_guard_blocked_action_executes_guard_next_action_without_model_redecision() -> None:
    adapter = FakeSubAgentAdapter()
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                "call_eda_agent",
                "call_eda_agent",
                "call_analysis_agent",
                "call_report_agent",
                "finalize",
            ]
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls[:2] == ["sql_agent", "eda_agent"]
    assert result["completed_agents"] == ["sql_agent", "eda_agent", "analysis_agent", "report_agent"]


def test_execute_subagent_merges_allowed_state_updates_without_erasing_sql() -> None:
    adapter = FakeSubAgentAdapter(
        {
            "sql_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="sql_agent",
                    status="success",
                    summary="SQL 완료",
                    artifact_ids=["artifact_sql"],
                ),
                state_updates={
                    "generated_sql": "SELECT 42 AS answer",
                    "analysis_plan": {"goal": "매출 분석", "route_kind": "comprehensive"},
                    "planner_mode": "llm",
                    "error_state": {"warning": "테스트 경고"},
                },
            ),
            "eda_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="eda_agent",
                    status="success",
                    summary="EDA 완료",
                    artifact_ids=["artifact_eda"],
                ),
                state_updates={"generated_sql": ""},
            ),
        }
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(["call_sql_agent", "call_eda_agent", "finalize"]),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["generated_sql"] == "SELECT 42 AS answer"
    assert result["analysis_plan"] == {
        "goal": "매출 분석",
        "route_kind": "comprehensive",
        "planner_mode": "llm",
    }
    assert result["error_state"] == {"warning": "테스트 경고"}


def test_approval_required_result_finalizes_as_user_waiting_state() -> None:
    adapter = FakeSubAgentAdapter(
        {
            "sql_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="sql_agent",
                    status="approval_required",
                    summary="SQL 실행 승인 필요",
                )
            )
        }
    )
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(["call_sql_agent"]))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "needs_user_approval"
    assert result["pending_approval"] == {
        "approval_id": "run_001:sql_agent:approval",
        "agent": "sql_agent",
        "reason": "SQL 실행 승인 필요",
        "approval_type": "agent_approval",
    }
    assert result["completed_agents"] == []
    assert result["final_answer"] == "사용자 승인이 필요합니다."


def test_fallback_used_non_retryable_result_fails_terminally() -> None:
    adapter = FakeSubAgentAdapter(
        {
            "analysis_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="analysis_agent",
                    status="success",
                    summary="fallback 분석 결과",
                    artifact_ids=["artifact_fallback"],
                    fallback_used=True,
                    retryable=False,
                )
            )
        }
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(["call_analysis_agent"]),
    )
    state = _state()
    state["agent_results"] = [
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql"],
        ).model_dump(mode="json")
    ]

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert result["final_answer"] == "에이전트 실행 결과 검증에 실패했습니다."


def test_short_query_finalizes_with_clarification_question() -> None:
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=SequencedDecisionModel())

    result = graph.invoke(_state("매출"), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "needs_clarification"
    assert result["needs_clarification"] is True
    assert result["clarification_question"]
    assert result["final_answer"] == result["clarification_question"]
