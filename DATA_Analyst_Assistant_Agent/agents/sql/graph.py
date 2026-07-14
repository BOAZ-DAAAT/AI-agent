"""SQL 에이전트 LangGraph 배선."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.agents.sql import nodes
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def route_after_plan(state: AgentState):
    validation = state.get("validation") or {}
    if validation.get("result") != "invalid":
        return "refresh"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"


def route_after_mart_design(state: AgentState):
    validation = state.get("validation") or {}
    if validation.get("result") != "invalid":
        return "generate"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"


def route_after_refresh_integrity_context(state: AgentState):
    route_kind = str((state.get("plan") or {}).get("route_kind") or "").strip().lower()
    if route_kind == "simple":
        return "generate"
    if route_kind == "comprehensive":
        return "design"
    raise ValueError(f"unsupported route_kind: {route_kind or 'empty'}")


def route_after_validation(state: AgentState):
    if state["validation"].get("result") == "valid":
        return "finalize"
    if not (state.get("retry_hint") or {}).get("retryable", True):
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"


# reason_code가 이 집합에 해당하면 plan 단계 재수립이 필요
_REPLAN_CODES = {"sql_plan_failed", "sql_mart_design_failed", "invalid_join_plan", "result_shape_mismatch", "intent_mismatch"}


def route_after_retry(state: AgentState):
    """retry 후 plan 단계 재수립이 필요한지, SQL만 재생성할지 판단."""
    reason_code = (state.get("retry_hint") or {}).get("reason_code", "")
    return "replan" if reason_code in _REPLAN_CODES else "regenerate"


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", nodes.load_context)
    graph.add_node("preplan_integrity_gate", nodes.preplan_integrity_gate)
    graph.add_node("plan_question", nodes.plan_question)
    graph.add_node("refresh_integrity_context", nodes.refresh_integrity_context)
    graph.add_node("design_mart", nodes.design_mart)
    graph.add_node("generate_sql", nodes.generate_sql)
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
    graph.add_conditional_edges(
        "refresh_integrity_context",
        route_after_refresh_integrity_context,
        {"generate": "generate_sql", "design": "design_mart"},
    )
    graph.add_conditional_edges(
        "design_mart",
        route_after_mart_design,
        {"generate": "generate_sql", "retry": "increase_retry", "finalize": "finalize_answer"},
    )
    graph.add_edge("generate_sql", "prevalidate_sql")
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
        {"replan": "plan_question", "regenerate": "generate_sql"},
    )
    graph.add_edge("finalize_answer", END)

    return graph.compile()
