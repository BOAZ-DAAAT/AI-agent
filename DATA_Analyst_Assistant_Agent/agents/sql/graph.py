"""SQL 에이전트 LangGraph 배선."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.agents.sql import nodes
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState

SQL_MAX_RETRIES = 1


def route_after_plan(state: AgentState):
    validation = state.get("validation") or {}
    if validation.get("result") != "invalid":
        return "refresh"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= SQL_MAX_RETRIES:
        return "finalize"
    return "retry"


def route_after_mart_design(state: AgentState):
    validation = state.get("validation") or {}
    if validation.get("result") != "invalid":
        return "generate"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= SQL_MAX_RETRIES:
        return "finalize"
    return "retry"


def route_after_refresh_context(state: AgentState):
    route_kind = str((state.get("plan") or {}).get("route_kind") or "").strip().lower()
    if route_kind == "simple":
        return "generate"
    if route_kind == "comprehensive":
        return "design"
    raise ValueError(f"unsupported route_kind: {route_kind or 'empty'}")


def route_after_schema_refresh(state: AgentState):
    validation = state.get("validation") or {}
    if validation.get("result") != "invalid":
        return "finalize_plan"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= SQL_MAX_RETRIES:
        return "finalize"
    return "retry"


def route_after_finalize_table_plan(state: AgentState):
    validation = state.get("validation") or {}
    if validation.get("result") == "invalid":
        if not (state.get("retry_hint") or {}).get("retryable", True):
            return "finalize"
        if state["retry_count"] >= SQL_MAX_RETRIES:
            return "finalize"
        return "retry"
    return route_after_refresh_context(state)


# 기존 테스트와 외부 호출 호환성을 유지한다.
route_after_refresh_integrity_context = route_after_refresh_context


def route_after_validation(state: AgentState):
    if state["validation"].get("result") == "valid":
        return "finalize"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= SQL_MAX_RETRIES:
        return "finalize"
    return "retry"


def route_after_retry(state: AgentState):
    """오류가 발생한 단계와 repair 가능성에 맞는 최소 노드로 돌아간다."""
    reason_code = (state.get("retry_hint") or {}).get("reason_code", "")
    retry_details = (state.get("retry_hint") or {}).get("details") or {}
    if reason_code == "sql_plan_failed":
        return "replan"
    if reason_code in {"sql_mart_design_failed", "mart_grain_missing", "mart_column_contract_invalid"}:
        return "redesign"
    if reason_code in {
        "mysql_dialect_error",
        "route_kind_mismatch",
        "intent_mismatch",
        "result_shape_mismatch",
        "mart_policy_mismatch",
        "mart_summary_bias",
        "execution_error",
    }:
        return "repair"
    if reason_code == "sql_generation_failed" and retry_details.get("generation_stage") == "repair":
        return "repair"
    return "regenerate"


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", nodes.load_context)
    graph.add_node("preplan_integrity_gate", nodes.preplan_integrity_gate)
    graph.add_node("plan_question", nodes.plan_question)
    graph.add_node("refresh_integrity_context", nodes.refresh_integrity_context)
    graph.add_node("refresh_schema_context", nodes.refresh_schema_context)
    graph.add_node("finalize_table_plan", nodes.finalize_table_plan)
    graph.add_node("design_mart", nodes.design_mart)
    graph.add_node("generate_sql", nodes.generate_sql)
    graph.add_node("repair_sql", nodes.repair_sql)
    graph.add_node("prevalidate_sql", nodes.prevalidate_sql)
    graph.add_node("execute_sql", nodes.execute_sql)
    graph.add_node("validate_sql_and_result", nodes.validate_sql_and_result)
    graph.add_node("increase_retry", nodes.increase_retry)
    graph.add_node("finalize_answer", nodes.finalize_answer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "preplan_integrity_gate")
    graph.add_edge("preplan_integrity_gate", "plan_question")
    graph.add_conditional_edges(
        "plan_question",
        route_after_plan,
        {"refresh": "refresh_integrity_context", "retry": "increase_retry", "finalize": "finalize_answer"},
    )
    graph.add_edge("refresh_integrity_context", "refresh_schema_context")
    graph.add_conditional_edges(
        "refresh_schema_context",
        route_after_schema_refresh,
        {"finalize_plan": "finalize_table_plan", "retry": "increase_retry", "finalize": "finalize_answer"},
    )
    graph.add_conditional_edges(
        "finalize_table_plan",
        route_after_finalize_table_plan,
        {"generate": "generate_sql", "design": "design_mart", "retry": "increase_retry", "finalize": "finalize_answer"},
    )
    graph.add_conditional_edges(
        "design_mart",
        route_after_mart_design,
        {"generate": "generate_sql", "retry": "increase_retry", "finalize": "finalize_answer"},
    )
    graph.add_edge("generate_sql", "prevalidate_sql")
    graph.add_edge("repair_sql", "prevalidate_sql")
    graph.add_conditional_edges(
        "prevalidate_sql",
        nodes.route_after_prevalidation,
        {"validate": "validate_sql_and_result", "execute": "execute_sql"},
    )
    graph.add_edge("execute_sql", "validate_sql_and_result")

    graph.add_conditional_edges(
        "validate_sql_and_result",
        route_after_validation,
        {"retry": "increase_retry", "finalize": "finalize_answer"},
    )

    graph.add_conditional_edges(
        "increase_retry",
        route_after_retry,
        {"replan": "plan_question", "redesign": "design_mart", "regenerate": "generate_sql", "repair": "repair_sql"},
    )
    graph.add_edge("finalize_answer", END)

    return graph.compile()
