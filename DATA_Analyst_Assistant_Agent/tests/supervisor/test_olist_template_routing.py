"""Supervisor의 Olist 템플릿 우선 라우팅 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.contracts import RetryHint
from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    _route_after_olist_template_match,
    make_match_olist_template_node,
    build_graph,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    empty_supervisor_state,
    to_orchestration_state,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import validate_subagent_result
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentToolResult


@pytest.fixture(autouse=True)
def _enable_olist_template_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLIST_TEMPLATE_ROUTING_ENABLED", "true")


def _catalog() -> dict:
    path = Path(__file__).parents[2] / "agents" / "sql" / "data" / "db_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _state(question: str) -> dict:
    return empty_supervisor_state(
        thread_id="thread_olist",
        run_id="run_olist",
        user_query=question,
        datasource_id="olist",
        catalog_summary=_catalog(),
    )


def test_template_match_builds_deterministic_plan_before_semantic_search() -> None:
    state = _state("월별 매출과 주문 수를 보여줘")
    updates = make_match_olist_template_node()(state)

    assert updates["olist_template_match"]["status"] == "matched"
    assert updates["analysis_plan"]["planner_mode"] == "deterministic"
    assert updates["analysis_plan"]["sql_generation_source"] == "olist_template"
    assert updates["analysis_plan"]["sql_template_id"] == "monthly_sales_orders"
    assert updates["next_action"] == "call_sql_agent"
    assert _route_after_olist_template_match(updates) == "execute_subagent"

    orchestration = to_orchestration_state({**state, **updates})
    assert orchestration.plan is not None
    assert orchestration.plan.sql_generation_source == "olist_template"
    assert orchestration.plan.sql_template_id.value == "monthly_sales_orders"


def test_disabled_template_routing_uses_semantic_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLIST_TEMPLATE_ROUTING_ENABLED", "false")

    updates = make_match_olist_template_node()(_state("월별 매출과 주문 수를 보여줘"))

    assert updates["olist_template_match"]["status"] == "disabled"
    assert updates["next_action"] == "create_plan"
    assert "analysis_plan" not in updates
    assert _route_after_olist_template_match(updates) == "semantic_fallback"


def test_unsupported_template_routes_to_existing_semantic_flow() -> None:
    updates = make_match_olist_template_node()(_state("최근 6개월 월별 매출과 주문 수"))

    assert updates["olist_template_match"]["status"] == "not_matched"
    assert "analysis_plan" not in updates
    assert _route_after_olist_template_match(updates) == "semantic_fallback"


def test_mart_template_builds_comprehensive_plan_and_parameters() -> None:
    updates = make_match_olist_template_node()(_state("2024년 RFM 데이터마트"))

    assert updates["olist_template_match"]["status"] == "matched"
    assert updates["analysis_plan"]["route_kind"] == "comprehensive"
    assert updates["analysis_plan"]["sql_template_kind"] == "mart"
    assert updates["analysis_plan"]["sql_template_id"] == "customer_rfm"
    assert updates["analysis_plan"]["sql_template_parameters"]["start_date"] == "2024-01-01"
    assert updates["analysis_plan"]["requires_mart_review"] is True
    assert _route_after_olist_template_match(updates) == "execute_subagent"


def test_existing_required_derivation_skips_deterministic_template() -> None:
    state = _state("월별 매출과 주문 수를 보여줘")
    state["analysis_plan"] = {
        "goal": "월별 매출",
        "required_derivations": [
            {"name": "순매출", "preferred_name": "net_revenue"}
        ],
    }

    updates = make_match_olist_template_node()(state)

    assert updates["olist_template_match"]["status"] == "skipped_structured_derivations"
    assert "analysis_plan" not in updates
    assert _route_after_olist_template_match(updates) == "semantic_fallback"


def test_existing_heuristic_only_plan_keeps_deterministic_template() -> None:
    state = _state("월별 매출과 주문 수를 보여줘")
    state["analysis_plan"] = {
        "goal": "월별 매출",
        "analysis_heuristics": [{"name": "이상치 민감도 기록"}],
    }

    updates = make_match_olist_template_node()(state)

    assert updates["olist_template_match"]["status"] == "matched"
    assert updates["analysis_plan"]["sql_template_id"] == "monthly_sales_orders"
    assert updates["analysis_plan"]["analysis_heuristics"][0]["name"] == "이상치 민감도 기록"
    assert updates["analysis_plan"]["analysis_heuristics"][0]["must_record"] is True
    assert _route_after_olist_template_match(updates) == "execute_subagent"


def test_semantic_sql_failure_requests_clarification_only_once() -> None:
    state = _state("복합 매출 분석")
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 검증 실패",
        retry_hint=RetryHint(
            retryable=False,
            suggested_action="clarify",
            reason_code="result_shape_mismatch",
            details={"clarification_question": "분석 기간과 집계 단위를 알려주세요."},
        ),
        error="SQL 검증 실패",
    )

    first = validate_subagent_result(state, result)
    state["sql_clarification_count"] = 1
    second = validate_subagent_result(state, result)

    assert first.decision == "clarify"
    assert first.next_action == "fail"
    assert second.decision == "reject"
    assert second.terminal_state == "failed_terminal"


def test_database_failure_does_not_request_user_clarification() -> None:
    state = _state("복합 매출 분석")
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="DB 연결 오류",
        retry_hint=RetryHint(
            retryable=False,
            suggested_action="stop_and_surface_error",
            reason_code="execution_error",
        ),
        error="DB 연결 오류",
    )

    decision = validate_subagent_result(state, result)

    assert decision.decision == "reject"
    assert decision.next_action == "fail"


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _DecisionModel:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)

    def invoke(self, messages: list[dict[str, str]]) -> _Message:
        if "semantic search 검색문 생성기" in messages[0]["content"]:
            payload = json.loads(messages[1]["content"])
            return _Message(json.dumps({"retrieval_query": payload.get("clarified_query") or payload.get("original_query"), "reason": "검색문"}))
        return _Message(json.dumps(self.decisions.pop(0), ensure_ascii=False))


class _ClarifyingSQLAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, agent_name: str, _state: dict[str, Any]) -> AgentToolResult:
        assert agent_name == "sql_agent"
        self.calls += 1
        result = AgentCompactResult(
            agent="sql_agent",
            status="failed",
            summary="semantic SQL 최종 검증 실패",
            retry_hint=RetryHint(
                retryable=False,
                suggested_action="clarify",
                reason_code="result_shape_mismatch",
                details={"clarification_question": "분석 기간과 집계 단위를 알려주세요."},
            ),
            error="semantic SQL 최종 검증 실패",
        )
        return AgentToolResult(agent_result=result)


def test_final_semantic_failure_interrupts_once_and_restarts_from_template_match(monkeypatch) -> None:
    import DATA_Analyst_Assistant_Agent.supervisor.graph as graph_module

    original_match = graph_module.match_olist_template
    match_calls: list[str] = []

    def recording_match(question: str, catalog: Any, **kwargs: Any):
        match_calls.append(question)
        return original_match(question, catalog, **kwargs)

    monkeypatch.setattr(graph_module, "match_olist_template", recording_match)
    decisions = [
        {"needs_clarification": False, "clarified_query": "복합 매출 분석", "clarification_question": "", "reason": "충분"},
        {"goal": "복합 매출 분석", "route_kind": "simple", "steps": ["SQL"], "metric": "매출", "dimension": None, "filters": [], "requires_mart_review": False, "reason": "계획"},
        {"next_action": "call_sql_agent", "reason": "SQL 실행"},
        {"goal": "최근 6개월 월별 매출", "route_kind": "simple", "steps": ["SQL"], "metric": "매출", "dimension": "월", "filters": ["최근 6개월"], "requires_mart_review": False, "reason": "재계획"},
        {"next_action": "call_sql_agent", "reason": "SQL 재실행"},
        {"terminal_state": "failed_terminal", "final_answer": "SQL 생성에 실패했습니다.", "next_action": "finalize", "reason": "재질문 한도"},
    ]
    adapter = _ClarifyingSQLAdapter()
    graph = build_graph(
        adapter,
        model=_DecisionModel(decisions),
        checkpointer=InMemorySaver(),
        analysis_rule_search=lambda *_args, **_kwargs: [],
    )
    state = empty_supervisor_state(
        thread_id="thread_sql_clarify",
        run_id="run_sql_clarify",
        user_query="복합 매출 분석",
        datasource_id=None,
    )
    config = {"configurable": {"thread_id": "thread_sql_clarify"}}

    interrupted = graph.invoke(state, config)
    interrupt_payload = interrupted["__interrupt__"][0].value
    resumed = graph.invoke(Command(resume={"answer": "최근 6개월 월별 매출"}), config)

    assert interrupt_payload["node"] == "collect_clarification"
    assert interrupt_payload["question"] == "분석 기간과 집계 단위를 알려주세요."
    assert len(match_calls) == 2
    assert adapter.calls == 2
    assert resumed["sql_clarification_count"] == 1
    assert resumed["terminal_state"] == "failed_terminal"
