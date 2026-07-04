from __future__ import annotations

import json

from typing import Any

from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    build_graph,
    make_create_analysis_plan_node,
    make_execute_subagent_node,
    make_validate_subagent_result_node,
)
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
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> FakeMessage:
        self.messages.append(messages)
        decision = self.decisions.pop(0)
        return FakeMessage(json.dumps(decision, ensure_ascii=False))


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


class SequencedSubAgentAdapter:
    def __init__(self, results: dict[str, list[AgentToolResult]]) -> None:
        self.results = {agent: list(items) for agent, items in results.items()}
        self.calls: list[str] = []

    def call(self, agent_name: str, state: dict[str, Any]) -> AgentToolResult:
        self.calls.append(agent_name)
        return self.results[agent_name].pop(0)


def _state(user_query: str = "월별 매출 추이를 분석해줘") -> dict[str, Any]:
    return empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query=user_query,
        datasource_id=None,
    )


def _clarify_decision(
    *,
    needs_clarification: bool = False,
    clarified_query: str = "월별 매출 추이를 분석해줘",
    clarification_question: str = "",
) -> dict[str, Any]:
    return {
        "needs_clarification": needs_clarification,
        "clarified_query": clarified_query,
        "clarification_question": clarification_question,
        "reason": "clarify 결정",
    }


def _plan_decision(route_kind: str = "trend") -> dict[str, Any]:
    return {
        "goal": "월별 매출 추이 분석",
        "route_kind": route_kind,
        "steps": ["SQL 집계", "EDA 탐색", "리포트 생성"],
        "metric": "매출",
        "dimension": "월",
        "filters": [],
        "requires_mart_review": False,
        "reason": "plan 결정",
    }


def _next_action_decision(action: str) -> dict[str, Any]:
    return {"next_action": action, "reason": f"{action} 결정"}


def _guard_decision(action: str, *, allowed: bool = True) -> dict[str, Any]:
    return {"allowed": allowed, "next_action": action, "reason": f"{action} guard"}


def _validation_decision(
    *,
    valid: bool = True,
    next_action: str = "create_plan",
    terminal_state: str = "running",
    final_answer: str = "",
) -> dict[str, Any]:
    return {
        "valid": valid,
        "next_action": next_action,
        "reason": "검증 decision",
        "terminal_state": terminal_state,
        "final_answer": final_answer,
    }


def _summary_decision(agent: str, next_action: str = "create_plan") -> dict[str, Any]:
    return {
        "step": "validate_subagent_result",
        "agent": agent,
        "action": f"call_{agent}",
        "summary": f"{agent} 단계 요약",
        "artifact_ids": [f"artifact_{agent}"],
        "next_action": next_action,
        "reason": "summary decision",
    }


def _final_decision(
    terminal_state: str = "completed",
    final_answer: str = "LLM 최종 답변",
) -> dict[str, Any]:
    return {
        "terminal_state": terminal_state,
        "final_answer": final_answer,
        "next_action": "finalize",
        "reason": "finalize decision",
    }


def _agent_flow_decisions(
    actions: list[str],
    *,
    validation_next_actions: list[str] | None = None,
    summary_next_actions: list[str] | None = None,
    final_terminal_state: str = "completed",
    final_answer: str = "LLM 최종 답변",
) -> list[dict[str, Any]]:
    validation_next_actions = validation_next_actions or ["create_plan"] * len(actions)
    summary_next_actions = summary_next_actions or validation_next_actions
    decisions = [_clarify_decision(), _plan_decision()]
    for action, validation_next_action, summary_next_action in zip(
        actions,
        validation_next_actions,
        summary_next_actions,
        strict=True,
    ):
        agent = action.replace("call_", "")
        decisions.extend(
            [
                _next_action_decision(action),
                _guard_decision(action),
                _validation_decision(next_action=validation_next_action),
                _summary_decision(agent, next_action=summary_next_action),
            ]
        )
    decisions.append(_final_decision(final_terminal_state, final_answer))
    return decisions


def test_supervisor_graph_runs_all_llm_nodes_and_finalizes() -> None:
    actions = ["call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"]
    model = SequencedDecisionModel(
        _agent_flow_decisions(
            actions,
            validation_next_actions=["create_plan", "create_plan", "create_plan", "finalize"],
            summary_next_actions=["create_plan", "create_plan", "create_plan", "finalize"],
            final_answer="리포트가 완료되었습니다.",
        )
    )
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=model)

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "eda_agent", "analysis_agent", "report_agent"]
    assert result["final_answer"] == "리포트가 완료되었습니다."
    assert [entry["node"] for entry in result["llm_decisions"]] == [
        "clarify_query",
        "create_analysis_plan",
        "decide_next_action",
        "execute_subagent",
        "validate_subagent_result",
        "summarize_step",
        "decide_next_action",
        "execute_subagent",
        "validate_subagent_result",
        "summarize_step",
        "decide_next_action",
        "execute_subagent",
        "validate_subagent_result",
        "summarize_step",
        "decide_next_action",
        "execute_subagent",
        "validate_subagent_result",
        "summarize_step",
        "finalize",
    ]
    assert len(model.messages) == len(result["llm_decisions"])


def test_build_graph_accepts_positional_subagent_adapter() -> None:
    graph = build_graph(
        FakeSubAgentAdapter(),
        model=SequencedDecisionModel(
            _agent_flow_decisions(
                ["call_sql_agent", "call_report_agent"],
                validation_next_actions=["create_plan", "finalize"],
                summary_next_actions=["create_plan", "finalize"],
            )
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "report_agent"]


def test_finalize_llm_can_fail_without_report_evidence() -> None:
    graph = build_graph(
        FakeSubAgentAdapter(),
        model=SequencedDecisionModel(
            _agent_flow_decisions(
                ["call_sql_agent"],
                validation_next_actions=["finalize"],
                summary_next_actions=["finalize"],
                final_terminal_state="failed_terminal",
                final_answer="리포트 근거가 부족합니다.",
            )
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert result["final_answer"] == "리포트 근거가 부족합니다."


def test_plan_node_records_llm_planner_mode() -> None:
    node = make_create_analysis_plan_node(SequencedDecisionModel([_plan_decision(route_kind="comprehensive")]))

    result = node(_state())

    assert result["analysis_plan"]["route_kind"] == "comprehensive"
    assert result["analysis_plan"]["planner_mode"] == "llm"
    assert result["llm_decisions"][0]["node"] == "create_analysis_plan"


def test_execute_guard_blocks_eda_and_redirects_to_sql() -> None:
    adapter = FakeSubAgentAdapter()
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_eda_agent"),
        _guard_decision("call_sql_agent", allowed=False),
        _guard_decision("call_sql_agent"),
        _validation_decision(next_action="create_plan"),
        _summary_decision("sql_agent", next_action="create_plan"),
        _next_action_decision("call_eda_agent"),
        _guard_decision("call_eda_agent"),
        _validation_decision(next_action="finalize"),
        _summary_decision("eda_agent", next_action="finalize"),
        _final_decision(),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["sql_agent", "eda_agent"]
    assert result["completed_agents"] == ["sql_agent", "eda_agent"]


def test_execute_guard_unsupported_redirect_fails_terminally() -> None:
    state = _state()
    state["next_action"] = "call_eda_agent"
    node = make_execute_subagent_node(
        FakeSubAgentAdapter(),
        SequencedDecisionModel([_guard_decision("create_plan", allowed=False)]),
    )

    result = node(state)

    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert "지원하지 않는 대체 action" in result["final_answer"]


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
        model=SequencedDecisionModel(
            _agent_flow_decisions(
                ["call_sql_agent", "call_eda_agent"],
                validation_next_actions=["create_plan", "finalize"],
                summary_next_actions=["create_plan", "finalize"],
            )
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["generated_sql"] == "SELECT 42 AS answer"
    assert result["analysis_plan"]["goal"] == "매출 분석"
    assert result["analysis_plan"]["route_kind"] == "comprehensive"
    assert result["analysis_plan"]["planner_mode"] == "llm"
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
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_sql_agent"),
        _guard_decision("call_sql_agent"),
        _validation_decision(
            valid=False,
            next_action="finalize",
            terminal_state="needs_user_approval",
            final_answer="사용자 승인이 필요합니다.",
        ),
        _summary_decision("sql_agent", next_action="finalize"),
        _final_decision("needs_user_approval", "사용자 승인이 필요합니다."),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

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


def test_validate_llm_failure_decision_finalizes_terminally() -> None:
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
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_analysis_agent"),
        _guard_decision("call_analysis_agent"),
        _validation_decision(
            valid=False,
            next_action="fail",
            terminal_state="failed_terminal",
            final_answer="검증 실패로 종료합니다.",
        ),
        _summary_decision("analysis_agent", next_action="finalize"),
        _final_decision("failed_terminal", "검증 실패로 종료합니다."),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))
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
    assert result["final_answer"] == "검증 실패로 종료합니다."
    assert "analysis_agent" not in result["completed_agents"]


def test_retryable_failed_agent_is_retried_and_removed_from_failed_agents_after_success() -> None:
    adapter = SequencedSubAgentAdapter(
        {
            "sql_agent": [
                AgentToolResult(
                    agent_result=AgentCompactResult(
                        agent="sql_agent",
                        status="failed",
                        summary="SQL 일시 실패",
                        retryable=True,
                        error="SQL validation failed",
                    )
                ),
                AgentToolResult(
                    agent_result=AgentCompactResult(
                        agent="sql_agent",
                        status="success",
                        summary="SQL 재시도 성공",
                        artifact_ids=["artifact_sql_retry"],
                    ),
                    state_updates={"generated_sql": "SELECT 1 AS sample_value"},
                ),
            ]
        }
    )
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_sql_agent"),
        _guard_decision("call_sql_agent"),
        _validation_decision(valid=False, next_action="call_sql_agent"),
        _summary_decision("sql_agent", next_action="call_sql_agent"),
        _guard_decision("call_sql_agent"),
        _validation_decision(valid=True, next_action="finalize"),
        _summary_decision("sql_agent", next_action="finalize"),
        _final_decision(),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["sql_agent", "sql_agent"]
    assert result["retry_counts"]["sql_agent"] == 1
    assert result["completed_agents"] == ["sql_agent"]
    assert result["failed_agents"] == []


def test_analysis_plan_sql_fields_are_not_overwritten_by_empty_state_updates() -> None:
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
                    "analysis_plan": {
                        "generated_sql": "",
                        "source_sql": None,
                        "route_kind": "updated",
                    }
                },
            )
        }
    )
    state = _state()
    state["next_action"] = "call_sql_agent"
    state["analysis_plan"] = {
        "generated_sql": "SELECT * FROM sales",
        "source_sql": "SELECT * FROM sales",
        "route_kind": "initial",
    }

    result = make_execute_subagent_node(
        adapter,
        SequencedDecisionModel([_guard_decision("call_sql_agent")]),
    )(state)

    assert result["analysis_plan"]["generated_sql"] == "SELECT * FROM sales"
    assert result["analysis_plan"]["source_sql"] == "SELECT * FROM sales"
    assert result["analysis_plan"]["route_kind"] == "updated"


def test_validate_subagent_result_node_handles_corrupt_last_agent_result_as_terminal_failure() -> None:
    state = _state()
    state["last_agent_result"] = {"agent": "sql_agent"}

    result = make_validate_subagent_result_node(SequencedDecisionModel([]))(state)

    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert result["final_answer"] == "에이전트 실행 결과 형식이 올바르지 않습니다."


def test_clarify_llm_decision_finalizes_with_question() -> None:
    question = "분석할 기간을 알려주세요."
    decisions = [
        _clarify_decision(
            needs_clarification=True,
            clarified_query="매출",
            clarification_question=question,
        ),
        _final_decision("needs_clarification", question),
    ]
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state("매출"), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "needs_clarification"
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == question
    assert result["final_answer"] == question


def test_invalid_llm_json_becomes_terminal_failure_without_fallback() -> None:
    class InvalidJsonModel:
        def invoke(self, messages):
            return FakeMessage("SQL부터 실행합니다.")

    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=InvalidJsonModel())

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert result["decision_errors"][0]["node"] == "clarify_query"
    assert result["completed_agents"] == []
