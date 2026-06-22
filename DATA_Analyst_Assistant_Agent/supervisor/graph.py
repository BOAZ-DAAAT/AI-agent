from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypedDict

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, OrchestrationState

if TYPE_CHECKING:
    from DATA_Analyst_Assistant_Agent.supervisor import SQLAgentSupervisor

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - import-safe fallback
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class GraphExecutionState(TypedDict):
    state: OrchestrationState
    last_agent_result: Optional[AgentEnvelope]
    last_validation_result: Optional[AgentEnvelope]
    gate_result: Optional[str]


AGENT_NODE_NAMES = (
    "run_sql_agent",
    "run_eda_agent",
    "run_analysis_agent",
    "run_visualization_agent",
    "run_report_agent",
)


class FallbackCompiledGraph:
    """Mirrors the explicit-node LangGraph flow when langgraph is absent."""

    def __init__(self, supervisor: SQLAgentSupervisor) -> None:
        self.supervisor = supervisor

    def invoke(self, graph_state: GraphExecutionState) -> GraphExecutionState:
        sup = self.supervisor
        graph_state = sup.parse_plan(graph_state)
        # SQL is always the first explicit agent node after parse_plan.
        graph_state = sup.run_sql_agent(graph_state)
        while True:
            graph_state = sup.central_validate(graph_state)
            graph_state = sup.supervisor_gate(graph_state)
            next_node = sup.route_after_gate(graph_state)
            if next_node == "approval_gate":
                return sup.approval_gate(graph_state)
            if next_node == "finalize":
                return sup.finalize(graph_state)
            if next_node == "end":
                return graph_state
            # Otherwise next_node is an explicit agent node name.
            graph_state = getattr(sup, next_node)(graph_state)


def build_graph(adapter: BackendAdapter, *, supervisor: SQLAgentSupervisor | None = None):
    if supervisor is None:
        from DATA_Analyst_Assistant_Agent.supervisor import SQLAgentSupervisor

        supervisor = SQLAgentSupervisor(adapter, build_runtime_graph=False)

    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        return FallbackCompiledGraph(supervisor)

    builder = StateGraph(GraphExecutionState)
    builder.add_node("parse_plan", supervisor.parse_plan)
    builder.add_node("run_sql_agent", supervisor.run_sql_agent)
    builder.add_node("run_eda_agent", supervisor.run_eda_agent)
    builder.add_node("run_analysis_agent", supervisor.run_analysis_agent)
    builder.add_node("run_visualization_agent", supervisor.run_visualization_agent)
    builder.add_node("run_report_agent", supervisor.run_report_agent)
    builder.add_node("central_validate", supervisor.central_validate)
    builder.add_node("supervisor_gate", supervisor.supervisor_gate)
    builder.add_node("approval_gate", supervisor.approval_gate)
    builder.add_node("finalize", supervisor.finalize)

    builder.add_edge(START, "parse_plan")
    # SQL is always the first explicit agent node after parse_plan.
    builder.add_edge("parse_plan", "run_sql_agent")
    # central_validate runs after every agent node.
    for agent_node in AGENT_NODE_NAMES:
        builder.add_edge(agent_node, "central_validate")
    builder.add_edge("central_validate", "supervisor_gate")
    builder.add_conditional_edges(
        "supervisor_gate",
        supervisor.route_after_gate,
        {
            "run_sql_agent": "run_sql_agent",
            "run_eda_agent": "run_eda_agent",
            "run_analysis_agent": "run_analysis_agent",
            "run_visualization_agent": "run_visualization_agent",
            "run_report_agent": "run_report_agent",
            "approval_gate": "approval_gate",
            "finalize": "finalize",
            "end": END,
        },
    )
    builder.add_edge("approval_gate", END)
    builder.add_edge("finalize", END)
    return builder.compile()
