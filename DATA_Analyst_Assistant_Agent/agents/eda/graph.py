"""EDA 에이전트 LangGraph 배선 (적응형 플래너 루프).

load_mart → inspect → planner ⇄ (분석 노드들) → ... → insight → hypothesis → chart_selector
플래너가 매 라운드 다음 분석을 고르고, 분석 노드는 끝나면 플래너로 복귀한다.
분석 노드 로직은 원본 그대로이며, '무엇을 언제 돌릴지'만 플래너가 결정한다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.agents.eda import nodes
from DATA_Analyst_Assistant_Agent.agents.eda.planner import ANALYSIS_NAMES, planner_node, route_after_planner
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.validator import route_after_validator, validator_node

# 분석 이름 → 노드 함수 매핑 (플래너 레지스트리와 1:1)
_ANALYSIS_NODE_FN = {
    "quality":      nodes.quality_node,
    "distribution": nodes.distribution_node,
    "comparison":   nodes.comparison_node,
    "relationship": nodes.relationship_node,
    "time":         nodes.time_node,
    "clustering":   nodes.clustering_node,
}


def build_app():
    graph = StateGraph(EDAState)

    graph.add_node("load_mart",       nodes.load_mart_node)
    graph.add_node("inspect",         nodes.inspect_node)
    graph.add_node("planner",         planner_node)
    for name, fn in _ANALYSIS_NODE_FN.items():
        graph.add_node(name, fn)
    graph.add_node("insight",         nodes.insight_node)
    graph.add_node("hypothesis",      nodes.hypothesis_node)
    graph.add_node("validator",       validator_node)
    graph.add_node("chart_selector",  nodes.chart_selector_node)

    graph.add_edge(START,       "load_mart")
    graph.add_edge("load_mart", "inspect")
    graph.add_edge("inspect",   "planner")

    # 플래너 → 고른 분석 노드 (또는 종료 시 insight)
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {**{name: name for name in ANALYSIS_NAMES}, "insight": "insight"},
    )
    # 각 분석 노드 → 플래너로 복귀 (루프)
    for name in _ANALYSIS_NODE_FN:
        graph.add_edge(name, "planner")

    graph.add_edge("insight",        "hypothesis")
    graph.add_edge("hypothesis",     "validator")

    # validator → 통과면 chart_selector, 부족하면 타겟 노드로 되돌림 (타겟 재시도)
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "chart_selector": "chart_selector",
            "planner":        "planner",
            "insight":        "insight",
            "hypothesis":     "hypothesis",
        },
    )
    graph.add_edge("chart_selector", END)

    return graph.compile()
