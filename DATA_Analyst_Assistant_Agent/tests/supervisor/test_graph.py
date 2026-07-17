from __future__ import annotations

import json

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    _route_after_decide,
    _route_after_summarize,
    build_graph,
    make_create_analysis_plan_node,
    make_decide_next_action_node,
    make_execute_subagent_node,
    make_finalize_node,
    make_resolve_candidate_node,
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
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import ReviewRequest


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

    def generate_insight(self, state: dict[str, Any]) -> AgentCompactResult:
        self.insight_calls = getattr(self, "insight_calls", 0) + 1
        if "insight_agent" in self.results:
            return self.results["insight_agent"].agent_result
        return AgentCompactResult(
            agent="insight_agent",
            status="success",
            summary="insight_agent 완료",
            artifact_ids=["artifact_insight_agent"],
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_insight_agent",
                    type="file",
                    kind="insight_payload",
                    summary="insight_agent 산출물 요약",
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

    def generate_insight(self, state: dict[str, Any]) -> AgentCompactResult:
        self.insight_calls = getattr(self, "insight_calls", 0) + 1
        return AgentCompactResult(
            agent="insight_agent",
            status="success",
            summary="insight_agent 완료",
            artifact_ids=["artifact_insight_agent"],
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_insight_agent",
                    type="file",
                    kind="insight_payload",
                    summary="insight_agent 산출물 요약",
                )
            ],
        )


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


class ForbiddenArtifactBackend(RecordingBackendAdapter):
    def get_artifact(self, _artifact_id: str) -> dict[str, Any]:
        raise AssertionError("후보 검증에서 backend 아티팩트를 조회하면 안 됩니다.")


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
        decisions.extend(
            [
                _next_action_decision(action),
                _semantic_decision(),
            ]
        )
    decisions.append(_final_decision(final_terminal_state, final_answer))
    return decisions


@pytest.mark.parametrize(
    ("next_action", "expected_node"),
    [
        ("call_sql_agent", "execute_subagent"),
        ("call_eda_agent", "execute_subagent"),
        ("call_analysis_agent", "execute_subagent"),
        ("finalize", "completion_guard"),
        ("fail", "completion_guard"),
    ],
)
def test_route_after_decide_maps_supported_action_explicitly(
    next_action: str,
    expected_node: str,
) -> None:
    state = _state()
    state["next_action"] = next_action

    assert _route_after_decide(state) == expected_node


@pytest.mark.parametrize(
    "next_action",
    ["clarify", "create_plan", "decide_next_action", "unknown_action", None],
)
def test_route_after_decide_rejects_unsupported_action(next_action: str | None) -> None:
    state = _state()
    state["next_action"] = next_action

    with pytest.raises(ValueError, match="지원하지 않는 next_action"):
        _route_after_decide(state)


def test_route_after_decide_rejects_missing_action() -> None:
    state = _state()
    state.pop("next_action")

    with pytest.raises(ValueError, match="지원하지 않는 next_action"):
        _route_after_decide(state)


def test_route_after_decide_prioritizes_terminal_state() -> None:
    state = _state()
    state["terminal_state"] = "failed_terminal"
    state["next_action"] = "unknown_action"

    assert _route_after_decide(state) == "finalize"


@pytest.mark.parametrize(
    ("next_action", "expected_node"),
    [
        ("decide_next_action", "decide_next_action"),
        ("call_sql_agent", "execute_subagent"),
        ("call_eda_agent", "execute_subagent"),
        ("call_analysis_agent", "execute_subagent"),
        ("call_report_agent", "generate_report"),
        ("finalize", "completion_guard"),
        ("fail", "completion_guard"),
    ],
)
def test_route_after_summarize_maps_supported_action_explicitly(
    next_action: str,
    expected_node: str,
) -> None:
    state = _state()
    state["next_action"] = next_action

    assert _route_after_summarize(state) == expected_node


@pytest.mark.parametrize("next_action", ["clarify", "create_plan", "unknown_action", None])
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
    actions = ["call_sql_agent", "call_eda_agent", "call_analysis_agent", "finalize"]
    decisions = _agent_flow_decisions(actions, final_answer="분석이 완료되었습니다.")
    model = SequencedDecisionModel(decisions)
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=model)

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == [
        "sql_agent", "eda_agent", "analysis_agent", "insight_agent",
    ]
    assert result["final_answer"] == "분석이 완료되었습니다."
    assert [entry["node"] for entry in result["llm_decisions"]] == [
        "clarify_query",
        "create_analysis_plan",
        "decide_next_action",
        "validate_candidate",
        "decide_next_action",
        "validate_candidate",
        "decide_next_action",
        "validate_candidate",
        "decide_next_action",
        "validate_candidate",
        "finalize",
    ]
    assert len(model.messages) == len(result["llm_decisions"])


def test_semantic_recovery_routes_analysis_candidate_to_sql_then_finalizes() -> None:
    adapter = FakeSubAgentAdapter()
    model = SequencedDecisionModel(
        [
            _clarify_decision(),
            _plan_decision(),
            _next_action_decision("call_analysis_agent"),
            _semantic_decision(
                semantic_valid=False,
                severity="error",
                recommended_next_action="call_sql_agent",
            ),
            _semantic_decision(recommended_next_action="finalize"),
            # completion_guard가 insight_agent를 결정론적으로 강제하므로, insight의
            # validate_candidate 몫 semantic decision.
            _semantic_decision(),
            _final_decision("completed", "복구된 근거로 인사이트를 완료했습니다."),
        ]
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=model,
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_semantic_recovery"}})

    assert adapter.calls == ["analysis_agent", "sql_agent"]
    assert adapter.insight_calls == 1
    assert result["semantic_recovery_attempts"] == {"sql_agent": 1}
    assert result["completed_agents"] == ["sql_agent", "insight_agent"]
    assert result["failed_agents"] == []
    assert result["accepted_evidence"].keys() == {"sql_agent", "insight_agent"}
    assert len(result["rejected_results"]) == 1
    assert result["rejected_results"][0]["result"]["agent"] == "analysis_agent"
    assert result["terminal_state"] == "completed"


def test_semantic_recovery_without_recommendation_generates_limited_insight_from_accepted_evidence() -> None:
    adapter = FakeSubAgentAdapter()
    model = SequencedDecisionModel(
        [
            _clarify_decision(),
            _plan_decision(),
            _next_action_decision("call_sql_agent"),
            _semantic_decision(),
            _next_action_decision("call_analysis_agent"),
            _semantic_decision(semantic_valid=False, severity="error"),
            # analysis 실패 → 복구 권고가 없고 근거는 있어 제한적 인사이트 폴백이
            # insight_agent를 직접 스케줄한다(completion_guard가 아니라 semantic recovery).
            _semantic_decision(),
            _final_decision("completed", "제한적 인사이트를 완료했습니다."),
        ]
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=model,
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_limited_report"}})

    assert adapter.calls == ["sql_agent", "analysis_agent"]
    assert adapter.insight_calls == 1
    assert result["semantic_recovery_attempts"] == {"insight_agent": 1}
    assert result["completed_agents"] == ["sql_agent", "insight_agent"]
    assert "analysis_agent" not in result["accepted_evidence"]
    assert any("제한적 인사이트" in item for item in result["limitations"])
    assert any(
        event["type"] == "semantic_recovery.limited_insight"
        for event in result["run_events"]
    )
    assert result["terminal_state"] == "completed"


def test_supervisor_graph_does_not_query_artifacts_during_candidate_validation() -> None:
    backend = ForbiddenArtifactBackend()
    adapter = FakeSubAgentAdapter()
    adapter.backend_adapter = backend
    decisions = _agent_flow_decisions(["call_analysis_agent", "finalize"])
    model = SequencedDecisionModel(decisions)
    graph = build_graph(subagent_adapter=adapter, model=model)

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_no_artifact_read"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["analysis_agent", "insight_agent"]
    assert any(event["event_type"] == "evidence.promoted" for event in backend.events)


def test_supervisor_graph_has_expected_nodes() -> None:
    graph = build_graph(FakeSubAgentAdapter(), model=SequencedDecisionModel([]))

    assert set(graph.nodes) == {
        "__start__",
        "clarify_query",
        "collect_clarification",
        "collect_analysis_review",
        "resolve_analysis_review",
        "create_analysis_plan",
        "decide_next_action",
        "execute_subagent",
        "generate_insight",
        "validate_candidate",
        "commit_candidate",
        "completion_guard",
        "finalize",
    }


def test_finalize_request_after_sql_forces_insight_generation_before_completion() -> None:
    adapter = FakeSubAgentAdapter()
    model = SequencedDecisionModel(
        [
            _clarify_decision(),
            _plan_decision(),
            _next_action_decision("call_sql_agent"),
            _semantic_decision(),
            _next_action_decision("finalize"),
            # completion_guard가 insight_agent를 결정론적으로 강제한다.
            _semantic_decision(),
            _final_decision("completed", "최종 인사이트가 완료되었습니다."),
        ]
    )
    graph = build_graph(subagent_adapter=adapter, model=model)

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_force_report"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "insight_agent"]
    assert result["accepted_evidence"]["insight_agent"][0]["artifact_id"] == "artifact_insight_agent"
    assert adapter.insight_calls == 1


def test_finalize_request_without_evidence_fails_without_calling_agents() -> None:
    adapter = FakeSubAgentAdapter()
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("finalize"),
                _final_decision("completed", "LLM은 완료로 판단했습니다."),
            ]
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_no_evidence"}})

    assert result["terminal_state"] == "failed_terminal"
    assert "근거" in result["final_answer"]
    assert adapter.calls == []
    assert getattr(adapter, "insight_calls", 0) == 0


def test_finalize_request_with_promoted_insight_does_not_generate_duplicate() -> None:
    adapter = FakeSubAgentAdapter()
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["artifact_analysis"],
        ),
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="insight_agent",
            status="success",
            summary="인사이트 완료",
            artifact_ids=["artifact_insight"],
        ),
    )
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("finalize"),
                # insight_agent가 이미 완료돼있어 completion_guard가 바로 ready를 반환한다.
                _final_decision(),
            ]
        ),
    )

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_existing_insight"}})

    assert result["terminal_state"] == "completed"
    assert getattr(adapter, "insight_calls", 0) == 0
    assert result["completed_agents"].count("insight_agent") == 1


def test_finalize_node_rejects_completed_without_required_report() -> None:
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision("completed", "LLM은 완료로 판단했습니다.")])
    )

    result = node(_state())

    assert result["terminal_state"] == "failed_terminal"
    assert "근거" in result["final_answer"]


def test_finalize_node_rejects_completed_checkpoint_without_insight() -> None:
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["artifact_analysis"],
        ),
    )
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision("completed", "LLM은 완료로 판단했습니다.")])
    )

    result = node(state)

    assert result["terminal_state"] == "failed_terminal"
    assert "인사이트" in result["final_answer"]


@pytest.mark.parametrize(
    "terminal_state",
    [
        "failed_terminal",
        "needs_user_approval",
        "needs_clarification",
        "failed_with_recoverable_context",
    ],
)
def test_finalize_node_preserves_protected_terminal_state_without_report(
    terminal_state: str,
) -> None:
    state = _state()
    state["terminal_state"] = terminal_state
    state["final_answer"] = "기존 terminal 상태를 보존합니다."
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision("completed", "LLM은 완료로 판단했습니다.")])
    )

    result = node(state)

    assert result["terminal_state"] == terminal_state
    assert result["final_answer"] == "기존 terminal 상태를 보존합니다."


def test_decide_next_action_fail_sets_terminal_failure_immediately() -> None:
    node = make_decide_next_action_node(
        SequencedDecisionModel([_next_action_decision("fail")])
    )

    result = node(_state())

    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert "fail" in result["final_answer"]


@pytest.mark.parametrize("next_action", ["clarify", "create_plan"])
def test_decide_next_action_rejects_initial_only_llm_response_as_decision_error(
    next_action: str,
) -> None:
    node = make_decide_next_action_node(
        SequencedDecisionModel([_next_action_decision(next_action)])
    )

    result = node(_state())

    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert result["decision_errors"][0]["node"] == "decide_next_action"
    assert "Supervisor LLM decision에 실패했습니다" in result["final_answer"]


def test_build_graph_accepts_positional_subagent_adapter() -> None:
    decisions = _agent_flow_decisions(["call_sql_agent", "finalize"])
    graph = build_graph(
        FakeSubAgentAdapter(),
        model=SequencedDecisionModel(decisions),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "insight_agent"]


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
    assert resumed["terminal_state"] == "failed_terminal"
    assert "근거" in resumed["final_answer"]
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


def test_finalize_llm_can_fail_without_insight_evidence() -> None:
    graph = build_graph(
        FakeSubAgentAdapter(),
        model=SequencedDecisionModel(
            [
                _clarify_decision(),
                _plan_decision(),
                _next_action_decision("call_sql_agent"),
                _semantic_decision(),
                _next_action_decision("finalize"),
                # completion_guard가 insight_agent를 결정론적으로 강제한다.
                _semantic_decision(),
                _final_decision("failed_terminal", "인사이트 근거가 부족합니다."),
            ]
        ),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "failed_terminal"
    assert result["final_answer"] == "인사이트 근거가 부족합니다."


def test_plan_node_records_llm_planner_mode() -> None:
    node = make_create_analysis_plan_node(SequencedDecisionModel([_plan_decision(route_kind="comprehensive")]))

    result = node(_state())

    assert result["analysis_plan"]["route_kind"] == "comprehensive"
    assert result["analysis_plan"]["planner_mode"] == "llm"
    assert result["llm_decisions"][0]["node"] == "create_analysis_plan"


def test_execute_subagent_runs_only_the_registered_action() -> None:
    adapter = FakeSubAgentAdapter()
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_eda_agent"),
        _semantic_decision(),
        _next_action_decision("finalize"),
        # completion_guard가 insight_agent를 결정론적으로 강제한다.
        _semantic_decision(),
        _final_decision(),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["eda_agent"]
    assert result["completed_agents"] == ["eda_agent", "insight_agent"]
    assert result["terminal_state"] == "completed"


def test_execute_subagent_emits_agent_lifecycle_events() -> None:
    adapter = FakeSubAgentAdapter()
    adapter.backend_adapter = RecordingBackendAdapter()
    state = _state()
    state["next_action"] = "call_sql_agent"
    node = make_execute_subagent_node(adapter, None)

    result = node(state)

    assert adapter.calls == ["sql_agent"]
    assert result["current_step"] == "execute_subagent"
    event_pairs = [
        (event["event_type"], event["node_name"])
        for event in adapter.backend_adapter.events
    ]
    assert ("node.started", "sql_agent") in event_pairs
    assert ("result.staged", "sql_agent") in event_pairs
    assert ("node.completed", "sql_agent") in event_pairs


def test_execute_subagent_contract_mismatch_emits_failed_lifecycle_event() -> None:
    adapter = ContractViolatingSubAgentAdapter()
    adapter.backend_adapter = RecordingBackendAdapter()
    state = _state()
    state["next_action"] = "call_sql_agent"
    node = make_execute_subagent_node(adapter, None)

    result = node(state)

    assert result["terminal_state"] == "failed_terminal"
    event_pairs = [
        (event["event_type"], event["node_name"])
        for event in adapter.backend_adapter.events
    ]
    assert ("node.started", "sql_agent") in event_pairs
    assert ("node.failed", "sql_agent") in event_pairs


def test_execute_subagent_unsupported_action_fails_terminally() -> None:
    state = _state()
    state["next_action"] = "create_plan"
    node = make_execute_subagent_node(
        FakeSubAgentAdapter(),
        None,
    )

    result = node(state)

    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert "지원하지 않는 subagent action" in result["final_answer"]


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

    assert adapter.calls == []
    assert result["terminal_state"] == "failed_terminal"
    assert result["next_action"] == "finalize"
    assert result["current_step"] == "execute_subagent"
    assert "pending_result" not in result
    assert "last_agent_result" not in result
    assert "failed_agents" not in result
    assert "completed_agents" not in result
    assert "accepted_evidence" not in result
    assert "pending_result" in result["error_state"]["message"]


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
                _semantic_decision(),
                _next_action_decision("call_eda_agent"),
                _semantic_decision(),
                _next_action_decision("finalize"),
                # completion_guard가 insight_agent를 결정론적으로 강제한다.
                _semantic_decision(),
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
    assert result["terminal_state"] == "completed"
    assert "insight_agent" in result["completed_agents"]


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


def test_structured_analysis_review_returns_native_interrupt_with_full_request() -> None:
    review_request = {
        "question": "대표값은?",
        "proposal": "대표값 선택",
        "options": [
            {"id": "mean", "label": "평균", "method": "산술 평균", "impact": "평균", "recommended": False},
            {"id": "median", "label": "중앙값", "method": "50% 분위수", "impact": "중앙값", "recommended": True},
        ],
        "recommended_option_id": "median",
        "requires_followup_analysis": True,
    }
    adapter = FakeSubAgentAdapter(
        {
            "analysis_agent": AgentToolResult(
                agent_result=AgentCompactResult(
                    agent="analysis_agent",
                    status="approval_required",
                    summary="분석 선택 필요",
                    artifact_ids=["analysis_001"],
                    artifacts=[
                        ArtifactSummary(
                            artifact_id="analysis_001",
                            kind="analysis_result",
                            metadata={"kind": "analysis_result"},
                            preview={"review_request": review_request},
                            content_hash="hash_001",
                        )
                    ],
                    approval=ApprovalRequirement(
                        required=True,
                        reason="대표값을 선택해 주세요.",
                        approval_type="analysis.review",
                    ),
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
                _next_action_decision("call_analysis_agent"),
                _semantic_decision(),
            ]
        ),
        checkpointer=InMemorySaver(),
    )
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["sql_001"],
            artifacts=[ArtifactSummary(artifact_id="sql_001", content_hash="sql_hash")],
        ),
    )

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_analysis_review"}})

    payload = result["__interrupt__"][0].value
    assert payload["type"] == "analysis_review"
    assert payload["approval_id"].startswith("run_001:analysis_agent:candidate_")
    assert payload["approval_id"].endswith(":approval")
    assert payload["review_request"] == ReviewRequest.model_validate(review_request).model_dump(mode="json")
    assert result["pending_result"] is not None
    assert result["terminal_state"] == "running"


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
        _semantic_decision(),
        _next_action_decision("finalize"),
        # completion_guard가 insight_agent를 결정론적으로 강제한다.
        _semantic_decision(),
        _final_decision(),
    ]
    graph = build_graph(
        subagent_adapter=adapter,
        model=SequencedDecisionModel(decisions),
    )

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["sql_agent", "sql_agent"]
    assert result["retry_counts"]["sql_agent"] == 1
    assert result["completed_agents"] == ["sql_agent", "insight_agent"]
    assert result["failed_agents"] == []
    assert result["failure_streaks"] == {}
    assert result["terminal_state"] == "completed"


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
    state["validation_history"] = [
        {
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "agent": "sql_agent",
            "outcome": {"disposition": "accept"},
            "checks": [
                {
                    "name": "semantic",
                    "passed": True,
                    "findings": [],
                    "details": {
                        "semantic_valid": True,
                        "severity": "info",
                        "recommended_next_action": "call_eda_agent",
                        "missing_evidence": [],
                    },
                }
            ],
        }
    ]

    result = make_resolve_candidate_node()(state)

    assert result["next_action"] == "call_eda_agent"
    assert result["completed_agents"] == ["sql_agent"]


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
