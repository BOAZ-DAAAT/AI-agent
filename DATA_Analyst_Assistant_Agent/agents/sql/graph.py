"""SQL 에이전트 LangGraph 배선.

기존 `sql_agent/sql_agent.py` 의 build_app()/route_after_validation 을 분리한 것.
노드 이름과 엣지 구성은 원본과 동일하게 유지한다(컴파일된 그래프 위상 동일).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.agents.sql import nodes
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def route_after_validation(state: AgentState):
    if state["validation"].get("result") == "valid":
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", nodes.load_context)
    graph.add_node("plan_question", nodes.plan_question)
    graph.add_node("design_mart", nodes.design_mart)
    graph.add_node("generate_sql", nodes.generate_sql)
    graph.add_node("execute_sql", nodes.execute_sql)
    graph.add_node("validate_sql_and_result", nodes.validate_sql_and_result)
    graph.add_node("increase_retry", nodes.increase_retry)
    graph.add_node("finalize_answer", nodes.finalize_answer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan_question")
    graph.add_edge("plan_question", "design_mart")
    graph.add_edge("design_mart", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "validate_sql_and_result")

    graph.add_conditional_edges(
        "validate_sql_and_result",
        route_after_validation,
        {
            "retry": "increase_retry",
            "finalize": "finalize_answer"
        }
    )

    graph.add_edge("increase_retry", "generate_sql")
    graph.add_edge("finalize_answer", END)

    return graph.compile()
