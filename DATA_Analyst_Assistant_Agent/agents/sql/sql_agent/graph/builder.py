from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.node.context import load_context
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.node.planner import design_mart, plan_question
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.node.sql_steps import (
    execute_sql,
    finalize_answer,
    generate_sql,
    increase_retry,
    route_after_validation,
    validate_sql_and_result,
)
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.node.validation import prevalidate_sql, route_after_prevalidation
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import AgentState


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("plan_question", plan_question)
    graph.add_node("design_mart", design_mart)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("prevalidate_sql", prevalidate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("validate_sql_and_result", validate_sql_and_result)
    graph.add_node("increase_retry", increase_retry)
    graph.add_node("finalize_answer", finalize_answer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan_question")
    graph.add_edge("plan_question", "design_mart")
    graph.add_edge("design_mart", "generate_sql")
    graph.add_edge("generate_sql", "prevalidate_sql")
    graph.add_conditional_edges(
        "prevalidate_sql",
        route_after_prevalidation,
        {"validate": "validate_sql_and_result", "execute": "execute_sql"},
    )
    graph.add_edge("execute_sql", "validate_sql_and_result")
    graph.add_conditional_edges(
        "validate_sql_and_result",
        route_after_validation,
        {"retry": "increase_retry", "finalize": "finalize_answer"},
    )
    graph.add_edge("increase_retry", "generate_sql")
    graph.add_edge("finalize_answer", END)
    return graph.compile()
