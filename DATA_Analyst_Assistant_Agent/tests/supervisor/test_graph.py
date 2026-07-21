from __future__ import annotations

import json

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    _route_after_decide,
    build_graph,
    make_create_analysis_plan_node,
    make_decide_next_action_node,
    make_execute_subagent_node,
    make_finalize_node,
    make_retrieve_analysis_rules_node,
)
from DATA_Analyst_Assistant_Agent.supervisor.candidate import commit_candidate
from DATA_Analyst_Assistant_Agent.shared.pinecone import CompanyContextHit
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    begin_or_retry_agent_node,
    empty_supervisor_state,
    merge_agent_result,
    stage_candidate_result,
    wait_active_node,
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
        if "insight" in self.results:
            return self.results["insight"].agent_result
        return AgentCompactResult(
            agent="insight",
            status="success",
            summary="insight 완료",
            artifact_ids=["artifact_insight"],
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_insight",
                    type="file",
                    kind="insight_payload",
                    summary="insight 산출물 요약",
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
            agent="insight",
            status="success",
            summary="insight 완료",
            artifact_ids=["artifact_insight"],
            artifacts=[
                ArtifactSummary(
                    artifact_id="artifact_insight",
                    type="file",
                    kind="insight_payload",
                    summary="insight 산출물 요약",
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


def _validation_record(
    state: dict[str, Any],
    agent: str,
    disposition: str,
    *,
    recommended_next_action: str = "",
) -> dict[str, Any]:
    pending = state["pending_result"]
    semantic_details = {
        "semantic_valid": True,
        "severity": "info",
        "recommended_next_action": recommended_next_action,
        "missing_evidence": [],
    }
    return {
        "candidate_id": pending["candidate_id"],
        "validation_id": pending["validation_id"],
        "agent": agent,
        "outcome": {
            "disposition": disposition,
            "reason": "ok",
            "reason_code": "",
            "terminal_state": "running",
        },
        "checks": [
            {
                "name": "semantic",
                "passed": True,
                "findings": [],
                "details": semantic_details,
            }
        ],
    }


def test_retrieve_analysis_rules_uses_llm_to_select_only_query_relevant_rules() -> None:
    document_text = """# 구매 빈도 분석 규칙

## definition
- 자주 구매한 고객은 구매 완료 주문 횟수로 집계한다.

## default_metrics
- customer_unique_id별 distinct order_id 수를 사용한다.

## entity_grain
- customer_unique_id당 한 행이다.

## time_basis
- order_purchase_timestamp를 사용한다.

## required_tables
- customers
- orders

## join_constraints
- customers.customer_id = orders.customer_id로 조인한다.

## status_and_null_rules
- 취소 주문을 제외한다.

## constraints
- 결제 합계로 구매 빈도를 대체하지 않는다.

## clarify_when
- 기간이 없으면 전체 관측 기간인지 확인한다.

## ambiguous_examples
- 이 예시는 state에 저장하면 안 된다.

## positive_examples
- 이 원문도 state에 저장하면 안 된다.
"""

    def fake_search(query: str, **kwargs: Any) -> list[CompanyContextHit]:
        assert query == "자주 구매하는 고객 특징을 분석해줘"
        assert kwargs["top_k"] == 1
        assert kwargs["metadata_filter"] == {
            "doc_type": {"$eq": "analysis_query_rule"}
        }
        return [
            CompanyContextHit(
                record_id="purchase_frequency",
                score=0.94,
                text=document_text,
                document_id="purchase_frequency",
                title="Olist 구매 빈도 분석 규칙",
                metadata={"query_type": "purchase_frequency", "version": "1.0"},
            )
        ]

    model = SequencedDecisionModel(
        [
            {
                "applicable": True,
                "rules": [
                    "구매 빈도는 결제 합계가 아니라 customer_unique_id별 distinct order_id 수로 계산한다.",
                    "취소 주문은 구매 횟수에서 제외한다.",
                ],
                "clarification_needed": False,
                "clarification_question": "",
                "reason": "구매 빈도 집계에 직접 필요한 규칙만 선택했다.",
            }
        ]
    )
    updates = make_retrieve_analysis_rules_node(fake_search, model)(
        _state("자주 구매하는 고객 특징을 분석해줘")
    )

    assert updates["analysis_rule_retrieval"]["status"] == "success"
    assert updates["analysis_rule_context"]["document_id"] == "purchase_frequency"
    assert updates["analysis_rule_context"]["query_type"] == "purchase_frequency"
    assert updates["analysis_rule_context"]["rules"] == [
        "구매 빈도는 결제 합계가 아니라 customer_unique_id별 distinct order_id 수로 계산한다.",
        "취소 주문은 구매 횟수에서 제외한다.",
    ]
    extraction_payload = json.loads(model.messages[0][1]["content"])
    assert extraction_payload["document"]["content"] == document_text
    serialized = json.dumps(updates["analysis_rule_context"], ensure_ascii=False)
    assert "이 예시는 state에 저장하면 안 된다" not in serialized
    assert "이 원문도 state에 저장하면 안 된다" not in serialized
    assert document_text not in serialized


def test_retrieve_analysis_rules_fails_open_when_search_errors() -> None:
    state = _state("배송 지연 분석")

    def failing_search(query: str, **kwargs: Any) -> list[Any]:
        raise RuntimeError("pinecone unavailable")

    updates = make_retrieve_analysis_rules_node(failing_search)(state)

    assert updates["analysis_rule_context"] is None
    assert updates["analysis_rule_retrieval"]["status"] == "failed"
    assert updates["current_step"] == "retrieve_analysis_rules"
    assert updates["terminal_state"] == "running"
    assert "pinecone unavailable" in updates["limitations"][-1]


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
    reason: str = "의미 검증 advisory",
    missing_evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "semantic_valid": semantic_valid,
        "severity": severity,
        "recommended_next_action": recommended_next_action,
        "reason": reason,
        "missing_evidence": missing_evidence or [],
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


def test_supervisor_graph_runs_all_llm_nodes_and_finalizes() -> None:
    actions = ["call_sql_agent", "call_eda_agent", "call_analysis_agent", "finalize"]
    decisions = _agent_flow_decisions(actions, final_answer="분석이 완료되었습니다.")
    model = SequencedDecisionModel(decisions)
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=model)

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == [
        "sql_agent", "eda_agent", "analysis_agent", "insight",
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
            # completion_guard가 insight를 결정론적으로 강제하므로, insight의
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
    assert result["completed_agents"] == ["sql_agent", "insight"]
    assert result["failed_agents"] == []
    assert result["accepted_evidence"].keys() == {"sql_agent", "insight"}
    assert len(result["rejected_results"]) == 1
    assert result["rejected_results"][0]["result"]["agent"] == "analysis_agent"
    assert result["terminal_state"] == "completed"


def test_missing_evidence_recommendation_promotes_candidate_then_routes_to_sql() -> None:
    adapter = FakeSubAgentAdapter()
    model = SequencedDecisionModel(
        [
            _clarify_decision(),
            _plan_decision(),
            _next_action_decision("call_analysis_agent"),
            _semantic_decision(
                recommended_next_action="call_sql_agent",
                reason="",
                missing_evidence=["monthly_sales"],
            ),
            _semantic_decision(recommended_next_action="finalize"),
            _semantic_decision(),
            _final_decision("completed", "후속 SQL 근거로 인사이트를 완료했습니다."),
        ]
    )
    graph = build_graph(subagent_adapter=adapter, model=model)

    result = graph.invoke(
        _state(),
        {"configurable": {"thread_id": "thread_missing_evidence_advisory"}},
    )

    assert adapter.calls == ["analysis_agent", "sql_agent"]
    assert result["semantic_recovery_attempts"] == {}
    assert result["rejected_results"] == []
    assert result["completed_agents"] == ["analysis_agent", "sql_agent", "insight"]
    assert result["accepted_evidence"].keys() == {
        "analysis_agent", "sql_agent", "insight",
    }
    assert "누락 근거: monthly_sales" in result["limitations"]
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
            # insight를 직접 스케줄한다(completion_guard가 아니라 semantic recovery).
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
    assert result["semantic_recovery_attempts"] == {"insight": 1}
    assert result["completed_agents"] == ["sql_agent", "insight"]
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
    assert result["completed_agents"] == ["analysis_agent", "insight"]
    assert any(event["event_type"] == "evidence.promoted" for event in backend.events)


def test_supervisor_graph_has_expected_nodes() -> None:
    graph = build_graph(FakeSubAgentAdapter(), model=SequencedDecisionModel([]))

    assert set(graph.nodes) == {
        "__start__",
        "retrieve_analysis_rules",
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
            # completion_guard가 insight를 결정론적으로 강제한다.
            _semantic_decision(),
            _final_decision("completed", "최종 인사이트가 완료되었습니다."),
        ]
    )
    graph = build_graph(subagent_adapter=adapter, model=model)

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_force_report"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "insight"]
    assert result["accepted_evidence"]["insight"][0]["artifact_id"] == "artifact_insight"
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
            agent="insight",
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
                # insight가 이미 완료돼있어 completion_guard가 바로 ready를 반환한다.
                _final_decision(),
            ]
        ),
    )

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_existing_insight"}})

    assert result["terminal_state"] == "completed"
    assert getattr(adapter, "insight_calls", 0) == 0
    assert result["completed_agents"].count("insight") == 1


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


@pytest.mark.parametrize(
    "terminal_state",
    ["failed_terminal", "failed_with_recoverable_context"],
)
def test_finalize_terminal_failure_closes_active_node(terminal_state: str) -> None:
    backend = RecordingBackendAdapter()
    state = _state()
    state["last_completed_node_id"] = "run_001:node:1"
    state["node_sequence"] = 1
    state, active_node, _ = begin_or_retry_agent_node(state, "sql_agent")
    state["terminal_state"] = terminal_state
    state["final_answer"] = "복구할 수 없는 오류로 종료합니다."
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision()]),
        backend,
    )

    result = node(state)
    merged = {**state, **result}

    assert merged["active_node"] is None
    assert merged["last_completed_node_id"] == "run_001:node:1"
    assert merged["node_sequence"] == 2
    assert merged["failed_agents"] == ["sql_agent"]
    assert backend.events[-1]["event_type"] == "agent.failed"
    assert backend.events[-1]["metadata"]["node_id"] == active_node.node_id
    assert backend.events[-1]["metadata"]["terminal_state"] == terminal_state


def test_finalize_preserves_waiting_node_for_user_approval() -> None:
    backend = RecordingBackendAdapter()
    state, _, _ = begin_or_retry_agent_node(_state(), "sql_agent")
    state, waiting_node, _ = wait_active_node(
        state,
        agent_name="sql_agent",
        reason="SQL 실행 승인이 필요합니다.",
    )
    state["terminal_state"] = "needs_user_approval"
    state["final_answer"] = "SQL 실행 승인이 필요합니다."
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision()]),
        backend,
    )

    result = node(state)
    merged = {**state, **result}

    assert merged["terminal_state"] == "needs_user_approval"
    assert merged["active_node"] == waiting_node.model_dump(mode="json")
    assert backend.events == []


def test_finalize_rejects_completed_state_with_active_node() -> None:
    backend = RecordingBackendAdapter()
    state = merge_agent_result(
        _state(),
        AgentCompactResult(
            agent="insight",
            status="success",
            summary="인사이트 완료",
            artifact_ids=["artifact_insight"],
        ),
    )
    state, _, _ = begin_or_retry_agent_node(state, "sql_agent")
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision("completed", "완료")]),
        backend,
    )

    result = node(state)

    assert result["terminal_state"] == "failed_terminal"
    assert result["active_node"] is None
    assert result["error_state"]["reason_code"] == "active_node_in_completed_state"
    assert backend.events[-1]["event_type"] == "agent.failed"


def test_repeated_finalize_does_not_emit_duplicate_agent_failed() -> None:
    backend = RecordingBackendAdapter()
    state, _, _ = begin_or_retry_agent_node(_state(), "sql_agent")
    state["terminal_state"] = "failed_terminal"
    state["final_answer"] = "최종 실패"
    node = make_finalize_node(
        SequencedDecisionModel([_final_decision(), _final_decision()]),
        backend,
    )

    first = {**state, **node(state)}
    second = {**first, **node(first)}

    assert second["active_node"] is None
    assert [event["event_type"] for event in backend.events] == ["agent.failed"]


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
    assert result["completed_agents"] == ["sql_agent", "insight"]


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
        "input_mode": "free_text",
        "options": [],
        "allow_free_text": True,
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
                # completion_guard가 insight를 결정론적으로 강제한다.
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
    state = _state()
    state["analysis_rule_context"] = {
        "document_id": "sales-orders",
        "default_metrics": ["SUM(order_payments.payment_value)"],
    }

    result = node(state)

    assert result["analysis_plan"]["route_kind"] == "comprehensive"
    assert result["analysis_plan"]["planner_mode"] == "llm"
    assert result["analysis_plan"]["query_rules"] == state["analysis_rule_context"]
    assert result["llm_decisions"][0]["node"] == "create_analysis_plan"


def test_execute_subagent_runs_only_the_registered_action() -> None:
    adapter = FakeSubAgentAdapter()
    decisions = [
        _clarify_decision(),
        _plan_decision(),
        _next_action_decision("call_eda_agent"),
        _semantic_decision(),
        _next_action_decision("finalize"),
        # completion_guard가 insight를 결정론적으로 강제한다.
        _semantic_decision(),
        _final_decision(),
    ]
    graph = build_graph(subagent_adapter=adapter, model=SequencedDecisionModel(decisions))

    result = graph.invoke(_state(), {"configurable": {"thread_id": "thread_sales_001"}})

    assert adapter.calls == ["eda_agent"]
    assert result["completed_agents"] == ["eda_agent", "insight"]
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
    assert ("agent.started", "sql_agent") in event_pairs
    assert ("result.staged", "sql_agent") in event_pairs
    assert ("agent.completed", "sql_agent") not in event_pairs
    assert result["active_node"]["status"] == "running"


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
    assert ("agent.started", "sql_agent") in event_pairs
    assert ("agent.failed", "sql_agent") in event_pairs
    assert result["active_node"] is None


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
                # completion_guard가 insight를 결정론적으로 강제한다.
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
    assert "insight" in result["completed_agents"]


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
        # completion_guard가 insight를 결정론적으로 강제한다.
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
    assert result["completed_agents"] == ["sql_agent", "insight"]
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

    state["pending_validation"] = _validation_record(state, "analysis_agent", "accept")

    promoted = commit_candidate(state, None)

    assert promoted["pending_approval"] is None
    assert promoted["completed_agents"] == ["analysis_agent"]


def test_resolve_success_with_required_approval_waits_without_promotion() -> None:
    state, _, _ = begin_or_retry_agent_node(_state(), "analysis_agent")
    state = stage_candidate_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["artifact_analysis"],
            approval=ApprovalRequirement(required=True, reason="분석 검토 필요"),
        ),
    )

    state["pending_validation"] = _validation_record(state, "analysis_agent", "await_approval")

    pending = commit_candidate(state, None)

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
    state, _, _ = begin_or_retry_agent_node(_state(), "sql_agent")
    state = stage_candidate_result(
        state,
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

    state["pending_validation"] = _validation_record(
        state,
        "sql_agent",
        "accept",
        recommended_next_action="call_eda_agent",
    )

    result = commit_candidate(state, None)

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
