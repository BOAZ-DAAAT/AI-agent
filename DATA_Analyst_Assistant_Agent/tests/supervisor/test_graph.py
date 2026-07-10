from __future__ import annotations

import json

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    _route_after_summarize,
    build_graph,
    make_create_analysis_plan_node,
    make_execute_subagent_node,
    make_resolve_candidate_node,
    make_semantic_validate_subagent_result_node,
    make_summarize_step_node,
    make_validate_subagent_result_node,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
    merge_agent_result,
    stage_candidate_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentContractError, AgentToolResult
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    ApprovalRequirement,
    RetryHint,
)


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
        self.report_calls = 0

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

    def generate(self, state: dict[str, Any]) -> AgentCompactResult:
        self.report_calls += 1
        if "report_agent" in self.results:
            return self.results["report_agent"].agent_result
        return AgentCompactResult(
            agent="report_agent",
            status="success",
            summary="report_agent 완료",
            artifact_ids=["artifact_report_agent"],
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_report_agent",
                    type="report",
                    kind="final_report",
                    summary="report_agent 산출물 요약",
                )
            ],
        )


class SequencedSubAgentAdapter:
    def __init__(self, results: dict[str, list[AgentToolResult]]) -> None:
        self.results = {agent: list(items) for agent, items in results.items()}
        self.calls: list[str] = []

    def call(self, agent_name: str, state: dict[str, Any]) -> AgentToolResult:
        self.calls.append(agent_name)
        return self.results[agent_name].pop(0)


class ContractViolatingSubAgentAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, agent_name: str, state: dict[str, Any]) -> AgentToolResult:
        self.calls.append(agent_name)
        raise AgentContractError(
            f"requested={agent_name}, returned=eda_agent"
        )


def _state(user_query: str = "매출") -> dict[str, Any]:
    return empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query=user_query,
        datasource_id=None,
    )


class RecordingBackendAdapter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append_run_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                **kwargs,
            }
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


def _summary_decision(agent: str, next_action: str = "decide_next_action") -> dict[str, Any]:
    return {
        "step": "validate_subagent_result",
        "agent": agent,
        "action": f"call_{agent}",
        "summary": f"{agent} 단계 요약",
        "artifact_ids": [f"artifact_{agent}"],
        "next_action": next_action,
        "reason": "summary decision",
    }


def _semantic_decision(
    *,
    semantic_valid: bool = True,
    severity: str = "info",
    recommended_next_action: str = "",
) -> dict[str, Any]:
    return {
        "semantic_valid": semantic_valid,
        "severity": severity,
        "recommended_next_action": recommended_next_action,
        "reason": "의미 검증 advisory",
        "missing_evidence": [],
        "alignment_notes": ["계획과 결과가 정렬되어 있습니다."],
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
    summary_next_actions: list[str] | None = None,
    final_terminal_state: str = "completed",
    final_answer: str = "LLM 최종 답변",
) -> list[dict[str, Any]]:
    summary_next_actions = summary_next_actions or ["decide_next_action"] * len(actions)
    decisions = [_clarify_decision(), _plan_decision()]
    for action, summary_next_action in zip(
        actions,
        summary_next_actions,
        strict=True,
    ):
        agent = action.replace("call_", "")
        if action == "call_report_agent":
            decisions.append(_next_action_decision(action))
            decisions.append(_semantic_decision())
            continue
        decisions.extend(
            [
                _next_action_decision(action),
                _guard_decision(action),
                _semantic_decision(),
                _summary_decision(agent, next_action=summary_next_action),
            ]
        )
    decisions.append(_final_decision(final_terminal_state, final_answer))
    return decisions


@pytest.mark.parametrize(
    ("next_action", "expected_node"),
    [
        ("decide_next_action", "decide_next_action"),
        ("call_sql_agent", "execute_subagent"),
        ("call_eda_agent", "execute_subagent"),
        ("call_analysis_agent", "execute_subagent"),
        ("call_report_agent", "generate_report"),
        ("finalize", "finalize"),
        ("fail", "finalize"),
    ],
)
def test_route_after_summarize_maps_supported_action_explicitly(
    next_action: str,
    expected_node: str,
) -> None:
    state = _state()
    state["next_action"] = next_action

    assert _route_after_summarize(state) == expected_node


@pytest.mark.parametrize("next_action", ["create_plan", "unknown_action", None])
def test_route_after_summarize_rejects_unsupported_action(next_action: str | None) -> None:
    state = _state()
    state["next_action"] = next_action

    with pytest.raises(ValueError, match="지원하지 않는 next_action"):
        _route_after_summarize(state)


def test_route_after_summarize_rejects_missing_action() -> None:
    state = _state()
    state.pop("next_action")

    with pytest.raises(ValueError, match="지원하지 않는 next_action"):
        _route_after_summarize(state)


def test_route_after_summarize_prioritizes_terminal_state() -> None:
    state = _state()
    state["terminal_state"] = "failed_terminal"
    state["next_action"] = "unknown_action"

    assert _route_after_summarize(state) == "finalize"


def test_supervisor_graph_runs_all_llm_nodes_and_finalizes() -> None:
    actions = ["call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"]
    model = SequencedDecisionModel(
        _agent_flow_decisions(
            actions,
            summary_next_actions=[
                "decide_next_action",
                "decide_next_action",
                "decide_next_action",
                "finalize",
            ],
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
        "semantic_validate_subagent_result",
        "summarize_step",
        "decide_next_action",
        "execute_subagent",
        "semantic_validate_subagent_result",
        "summarize_step",
        "decide_next_action",
        "execute_subagent",
        "semantic_validate_subagent_result",
            "summarize_step",
            "decide_next_action",
            "semantic_validate_subagent_result",
            "finalize",
    ]
    assert len(model.messages) == len(result["llm_decisions"])


def test_build_graph_accepts_positional_subagent_adapter() -> None:
    graph = build_graph(
        FakeSubAgentAdapter(),
        model=SequencedDecisionModel(
            _agent_flow_decisions(
                ["call_sql_agent", "call_report_agent"],
                summary_next_actions=["decide_next_action", "finalize"],
            )
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "report_agent"]


def test_clarification_interrupt_returns_payload_and_skips_subagents() -> None:
    adapter = FakeSubAgentAdapter()
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                _clarify_decision(
                    needs_clarification=True,
                    clarified_query="매출",
                    clarification_question="어떤 기간과 단위로 매출을 분석할까요?",
                )
            ]
        ),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(_state("매출"), {"configurable": {"thread_id": "thread_clarify_001"}})

    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload == {
        "type": "clarification",
        "status": "waiting_input",
        "run_id": "run_001",
        "thread_id": "thread_sales_001",
        "question": "어떤 기간과 단위로 매출을 분석할까요?",
        "node": "collect_clarification",
        "expected_resume": {"answer": "string"},
    }
    assert adapter.calls == []


def test_clarification_resume_continues_from_create_analysis_plan() -> None:
    adapter = FakeSubAgentAdapter()
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                _clarify_decision(
                    needs_clarification=True,
                    clarified_query="매출",
                    clarification_question="어떤 기간과 단위로 매출을 분석할까요?",
                ),
                _plan_decision(),
                _next_action_decision("finalize"),
                _final_decision(),
            ]
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread_clarify_001"}}

    paused = graph.invoke(_state("매출"), config)
    resumed = graph.invoke(Command(resume={"answer": "최근 6개월 월별 매출"}), config)

    assert "__interrupt__" in paused
    assert resumed["terminal_state"] == "completed"
    assert resumed["analysis_plan"]["goal"] == "월별 매출 추이 분석"
    assert resumed["clarified_query"] == "최근 6개월 월별 매출"
    assert "추가 답변:" not in resumed["clarified_query"]
    assert [entry["node"] for entry in resumed["llm_decisions"]] == [
        "clarify_query",
        "create_analysis_plan",
        "decide_next_action",
        "finalize",
    ]
    assert adapter.calls == []


def test_finalize_llm_can_fail_without_report_evidence() -> None:
    graph = build_graph(
        FakeSubAgentAdapter(),
        model=SequencedDecisionModel(
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("call_sql_agent"),
                _guard_decision("call_sql_agent"),
                _semantic_decision(),
                _summary_decision("sql_agent", next_action="decide_next_action"),
                _next_action_decision("finalize"),
                _final_decision("failed_terminal", "리포트 근거가 부족합니다."),
            ]
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
        _semantic_decision(),
        _summary_decision("sql_agent", next_action="decide_next_action"),
        _next_action_decision("call_eda_agent"),
        _guard_decision("call_eda_agent"),
        _semantic_decision(),
        _summary_decision("eda_agent", next_action="decide_next_action"),
        _next_action_decision("finalize"),
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


def test_execute_subagent_contract_mismatch_becomes_non_retryable_terminal_failure() -> None:
    adapter = ContractViolatingSubAgentAdapter()
    accepted_evidence = {
        "analysis_agent": [{"artifact_id": "artifact_analysis"}],
    }
    state = _state()
    state.update(
        {
            "next_action": "call_sql_agent",
            "accepted_evidence": accepted_evidence,
            "completed_agents": ["analysis_agent"],
            "failed_agents": ["eda_agent"],
            "pending_result": {"candidate_id": "candidate_stale"},
            "last_agent_result": {"agent": "eda_agent", "status": "success"},
        }
    )
    node = make_execute_subagent_node(
        adapter,
        SequencedDecisionModel([_guard_decision("call_sql_agent")]),
    )

    result = node(state)

    assert adapter.calls == ["sql_agent"]
    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert result["current_step"] == "execute_subagent"
    assert result["pending_result"] is None
    assert result["last_agent_result"] == {}
    assert result["failed_agents"] == ["eda_agent", "sql_agent"]
    assert result["completed_agents"] == ["analysis_agent"]
    assert result["accepted_evidence"] == accepted_evidence
    assert result["error_state"]["reason_code"] == "agent_contract_mismatch"
    assert result["error_state"]["retryable"] is False


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
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("call_sql_agent"),
                _guard_decision("call_sql_agent"),
                _semantic_decision(),
                _summary_decision("sql_agent", next_action="decide_next_action"),
                _next_action_decision("call_eda_agent"),
                _guard_decision("call_eda_agent"),
                _semantic_decision(),
                _summary_decision("eda_agent", next_action="decide_next_action"),
                _next_action_decision("finalize"),
                _final_decision(),
            ]
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
                    artifact_ids=["artifact_sql_approval"],
                    approval=ApprovalRequirement(required=True),
                )
            )
        }
    )
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_sql_agent"),
        _guard_decision("call_sql_agent"),
        _semantic_decision(),
        _final_decision("needs_user_approval", "사용자 승인이 필요합니다."),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "needs_user_approval"
    assert result["pending_approval"]["approval_id"] == "run_001:sql_agent:approval"
    assert result["pending_approval"]["candidate_id"]
    assert result["pending_approval"]["validation_id"]
    assert result["completed_agents"] == []
    assert result["final_answer"] == "SQL 실행 승인 필요"
    assert "summarize_step" not in [entry["node"] for entry in result["llm_decisions"]]


def test_approval_required_result_preserves_terminal_state_when_finalize_llm_fails() -> None:
    adapter = FakeSubAgentAdapter(
        {
            "sql_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="sql_agent",
                    status="approval_required",
                    summary="SQL 실행 승인 필요",
                    artifact_ids=["artifact_sql_approval"],
                    approval=ApprovalRequirement(required=True),
                )
            )
        }
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("call_sql_agent"),
                _guard_decision("call_sql_agent"),
                _semantic_decision(),
            ]
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "needs_user_approval"
    assert result["pending_approval"]["approval_id"] == "run_001:sql_agent:approval"
    assert result["pending_approval"]["candidate_id"]
    assert result["decision_errors"][0]["node"] == "finalize"
    assert result["error_state"]["node"] == "finalize"
    assert result["final_answer"] == "SQL 실행 승인 필요"
    assert "summarize_step" not in [entry["node"] for entry in result["llm_decisions"]]


def test_deterministic_fallback_result_finalizes_terminally() -> None:
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
    assert "fallback" in result["final_answer"].lower()
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
        _summary_decision("sql_agent", next_action="call_sql_agent"),
        _guard_decision("call_sql_agent"),
        _semantic_decision(),
        _summary_decision("sql_agent", next_action="decide_next_action"),
        _next_action_decision("finalize"),
        _final_decision(),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["sql_agent", "sql_agent"]
    assert result["retry_counts"]["sql_agent"] == 1
    assert result["completed_agents"] == ["sql_agent"]
    assert result["failed_agents"] == []
    assert result["failure_streaks"] == {}


def test_repeated_analysis_failure_stops_after_second_call_with_recoverable_context() -> None:
    failed_result = AgentToolResult(
        agent_result=AgentCompactResult(
            agent="analysis_agent",
            status="failed",
            summary="분석 실패",
            retry_hint=RetryHint(
                retryable=True,
                reason_code="method_review_failed",
                details={"failure_reason": "wrong method"},
            ),
        )
    )
    adapter = SequencedSubAgentAdapter(
        {"analysis_agent": [failed_result, failed_result.model_copy(deep=True)]}
    )
    model = SequencedDecisionModel(
        [
            _clarify_decision(),
            _plan_decision(),
            _next_action_decision("call_analysis_agent"),
            _guard_decision("call_analysis_agent"),
            _summary_decision("analysis_agent", next_action="call_analysis_agent"),
            _guard_decision("call_analysis_agent"),
            _final_decision("completed", "완료로 덮어쓰면 안 됩니다."),
        ]
    )
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql"],
        ),
    )
    graph = build_graph(adapter, model=model)

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["analysis_agent", "analysis_agent"]
    assert result["retry_counts"] == {"analysis_agent": 1}
    assert result["failure_streaks"]["analysis_agent"]["consecutive_count"] == 2
    assert result["terminal_state"] == "failed_with_recoverable_context"
    assert result["final_answer"] == (
        "analysis_agent 실패 [method_review_failed]: wrong method (repeated_failure=true)"
    )
    assert "analysis_agent" not in result["completed_agents"]


def test_validate_node_records_failure_streak_and_shared_rejection_metadata() -> None:
    backend = RecordingBackendAdapter()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="failed",
        summary="분석 실패",
        retry_hint=RetryHint(
            retryable=True,
            reason_code="method_review_failed",
            details={"failure_reason": "wrong method"},
        ),
    )
    state = stage_candidate_result(_state(), result)
    state["last_agent_result"] = result.model_dump(mode="json")

    updates = make_validate_subagent_result_node(None, backend)(state)

    expected_metadata = {
        "agent": "analysis_agent",
        "reason_code": "method_review_failed",
        "failure_reason": "wrong method",
        "repeated_failure": False,
    }
    assert updates["failure_streaks"]["analysis_agent"]["consecutive_count"] == 1
    assert updates["validation_results"][-1] | expected_metadata == updates["validation_results"][-1]
    internal_event = updates["run_events"][-1]
    assert internal_event["type"] == "validation.rejected"
    assert internal_event | expected_metadata == internal_event
    assert backend.events[-1]["event_type"] == "validation.rejected"
    assert backend.events[-1]["metadata"] | expected_metadata == backend.events[-1]["metadata"]


@pytest.mark.parametrize(
    ("agent", "has_sql_context", "expected_terminal"),
    [
        ("analysis_agent", True, "failed_with_recoverable_context"),
        ("analysis_agent", False, "failed_terminal"),
        ("sql_agent", True, "failed_terminal"),
        ("eda_agent", True, "failed_terminal"),
        ("report_agent", True, "failed_terminal"),
    ],
)
def test_validate_node_honors_repeated_failure_terminal_policy(
    agent: str,
    has_sql_context: bool,
    expected_terminal: str,
) -> None:
    state = _state()
    if has_sql_context:
        state = merge_agent_result(
            state,
            AgentCompactResult(
                agent="sql_agent",
                status="success",
                summary="SQL 완료",
                artifact_ids=["artifact_sql"],
            ),
        )
    state["failure_streaks"] = {
        agent: {
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
            "signature": '["method_review_failed", "wrong method"]',
            "consecutive_count": 1,
        }
    }
    result = AgentCompactResult(
        agent=agent,
        status="failed",
        summary="실패",
        retry_hint=RetryHint(
            retryable=True,
            reason_code="method_review_failed",
            details={"failure_reason": "wrong method"},
        ),
    )
    state = stage_candidate_result(state, result)
    state["last_agent_result"] = result.model_dump(mode="json")

    updates = make_validate_subagent_result_node(None)(state)

    assert updates["terminal_state"] == expected_terminal
    assert updates["next_action"] == "finalize"
    assert updates["validation_results"][-1]["repeated_failure"] is True
    assert updates["final_answer"] == (
        f"{agent} 실패 [method_review_failed]: wrong method (repeated_failure=true)"
    )


def test_failed_result_with_required_approval_is_rejected_not_awaited() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="failed",
        summary="분석 실패",
        retryable=False,
        approval=ApprovalRequirement(required=True, reason="승인 필요"),
    )
    state = stage_candidate_result(_state(), result)
    state["last_agent_result"] = result.model_dump(mode="json")

    updates = make_validate_subagent_result_node(None)(state)

    assert updates["validation_results"][-1]["decision"] == "reject"
    assert updates["terminal_state"] == "failed_terminal"
    assert updates.get("pending_approval") is None


def test_resolve_candidate_uses_only_approval_required_flag() -> None:
    state = stage_candidate_result(
        _state(),
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_analysis",
                    preview={"human_review": {"required": True}},
                )
            ],
            approval=ApprovalRequirement(required=False),
        ),
    )

    promoted = make_resolve_candidate_node()(state)

    assert promoted["pending_approval"] is None
    assert promoted["completed_agents"] == ["analysis_agent"]


def test_resolve_success_with_required_approval_waits_without_promotion() -> None:
    state = stage_candidate_result(
        _state(),
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["artifact_analysis"],
            approval=ApprovalRequirement(required=True, reason="분석 검토 필요"),
        ),
    )

    pending = make_resolve_candidate_node()(state)

    assert pending["terminal_state"] == "needs_user_approval"
    assert pending["pending_approval"]["reason"] == "분석 검토 필요"
    assert pending.get("completed_agents", []) == []


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
    assert result["analysis_plan"]["route_kind"] == "initial"
    assert result["pending_result"]["state_updates"]["analysis_plan"]["route_kind"] == "updated"


def test_validate_subagent_result_node_handles_corrupt_last_agent_result_as_terminal_failure() -> None:
    state = _state()
    state["last_agent_result"] = {"agent": "sql_agent"}

    result = make_validate_subagent_result_node(SequencedDecisionModel([]))(state)

    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert result["final_answer"] == "에이전트 실행 결과 형식이 올바르지 않습니다."


def test_report_success_without_artifact_fails_terminally_without_validation_llm() -> None:
    adapter = FakeSubAgentAdapter(
        {
            "report_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="report_agent",
                    status="success",
                    summary="리포트 생성 완료",
                )
            )
        }
    )
    model = SequencedDecisionModel(
        [
            _clarify_decision(),
            _plan_decision(),
            _next_action_decision("call_report_agent"),
            _final_decision("failed_terminal", "리포트 산출물이 없습니다."),
        ]
    )
    graph = build_graph(subagent_adapter=adapter, model=model)
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql"],
        ),
    )

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert "리포트 산출물 ID" in result["final_answer"]
    assert "report_agent" not in result["completed_agents"]
    assert [entry["node"] for entry in result["llm_decisions"]] == [
        "clarify_query",
        "create_analysis_plan",
        "decide_next_action",
        "finalize",
    ]
    assert len(model.messages) == len(result["llm_decisions"])


def test_semantic_validation_invalid_result_blocks_candidate() -> None:
    state = _state()
    state["last_agent_result"] = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["artifact_sql"],
    ).model_dump(mode="json")
    state["validation_results"] = [
        {
            "agent": "sql_agent",
            "valid": True,
            "next_action": "decide_next_action",
            "terminal_state": "running",
            "reason": "hard validation 통과",
        }
    ]
    state["next_action"] = "decide_next_action"
    model = SequencedDecisionModel(
        [_semantic_decision(semantic_valid=False, severity="warning", recommended_next_action="call_sql_agent")]
    )
    node = make_semantic_validate_subagent_result_node(model)

    result = node(state)

    assert result["next_action"] == "finalize"
    assert result["terminal_state"] == "failed_terminal"
    assert result["semantic_validation_results"][0]["recommended_next_action"] == "call_sql_agent"
    assert result["semantic_validation_results"][0]["semantic_valid"] is False
    assert result["llm_decisions"][0]["node"] == "semantic_validate_subagent_result"


def test_semantic_validation_skips_when_hard_validation_failed() -> None:
    state = _state()
    state["last_agent_result"] = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
    ).model_dump(mode="json")
    state["validation_results"] = [
        {
            "agent": "sql_agent",
            "valid": False,
            "next_action": "call_sql_agent",
            "terminal_state": "running",
            "reason": "hard validation 실패",
        }
    ]
    state["next_action"] = "call_sql_agent"
    model = SequencedDecisionModel([_semantic_decision()])
    node = make_semantic_validate_subagent_result_node(model)

    result = node(state)

    assert result["semantic_validation_results"] == []
    assert result["llm_decisions"] == []
    assert model.messages == []


def test_semantic_validation_failure_retries_once() -> None:
    class FailingModel:
        def invoke(self, messages):
            raise RuntimeError("semantic model unavailable")

    state = _state()
    state["last_agent_result"] = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["artifact_sql"],
    ).model_dump(mode="json")
    state["validation_results"] = [
        {
            "agent": "sql_agent",
            "valid": True,
            "next_action": "decide_next_action",
            "terminal_state": "running",
            "reason": "hard validation 통과",
        }
    ]
    state.pop("next_action")

    result = make_semantic_validate_subagent_result_node(FailingModel())(state)

    assert result["current_step"] == "semantic_validate_retry"
    assert list(result["semantic_retry_counts"].values()) == [1]


def test_resolve_candidate_uses_validated_semantic_recommendation() -> None:
    state = stage_candidate_result(
        _state(),
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql"],
        ),
        {},
    )
    state["semantic_validation_results"] = [
        {
            "agent": "sql_agent",
            "semantic_valid": True,
            "severity": "info",
            "recommended_next_action": "call_eda_agent",
            "missing_evidence": [],
        }
    ]

    result = make_resolve_candidate_node()(state)

    assert result["next_action"] == "call_eda_agent"
    assert result["completed_agents"] == ["sql_agent"]


def test_step_summary_records_llm_action_without_overwriting_state_action() -> None:
    state = _state()
    state["next_action"] = "decide_next_action"
    state["last_agent_result"] = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["artifact_sql"],
    ).model_dump(mode="json")
    node = make_summarize_step_node(
        SequencedDecisionModel(
            [_summary_decision("sql_agent", next_action="call_sql_agent")]
        )
    )

    result = node(state)

    assert "next_action" not in result
    assert result["step_summaries"][0]["next_action"] == "call_sql_agent"


def test_finalize_preserves_validation_terminal_state_when_llm_returns_completed() -> None:
    adapter = FakeSubAgentAdapter(
        {
            "report_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="report_agent",
                    status="success",
                    summary="리포트 생성 완료",
                )
            )
        }
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("call_report_agent"),
                _final_decision("completed", "LLM은 완료로 판단했습니다."),
            ]
        ),
    )
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql"],
        ),
    )

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert "리포트 산출물 ID" in result["final_answer"]


def test_clarify_llm_decision_interrupts_with_question() -> None:
    question = "분석할 기간을 알려주세요."
    decisions = [
        _clarify_decision(
            needs_clarification=True,
            clarified_query="매출",
            clarification_question=question,
        ),
    ]
    graph = build_graph(
        subagent_adapter=FakeSubAgentAdapter(),
        model=SequencedDecisionModel(decisions),
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(_state("매출"), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["__interrupt__"][0].value["question"] == question


def test_invalid_llm_json_becomes_terminal_failure_without_fallback() -> None:
    class InvalidJsonModel:
        def invoke(self, messages):
            return FakeMessage("SQL부터 실행합니다.")

    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=InvalidJsonModel())

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert result["decision_errors"][0]["node"] == "clarify_query"
    assert result["completed_agents"] == []
