from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.methods import build_analysis_result
from DATA_Analyst_Assistant_Agent.agents.analysis.planner import build_analysis_plan
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisExecutionPlan
from DATA_Analyst_Assistant_Agent.agents.analysis.self_check import run_analysis_self_check
from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck, OrchestrationState

from langgraph.graph import END, START, StateGraph


class AnalysisWorkflowState(TypedDict, total=False):
    orchestration_state: OrchestrationState
    dataframe: pd.DataFrame
    eda_profiles: list[dict[str, Any]]
    question_type: str | None
    planner_model: Any | None
    execution_plan: AnalysisExecutionPlan
    result: dict[str, Any]
    local_checks: list[LocalCheck]
    terminal_reason: str


def plan_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    orchestration = state["orchestration_state"]
    context = build_analysis_context(
        orchestration,
        state["dataframe"],
        state.get("eda_profiles", []),
        question_type=state.get("question_type"),
    )
    return {
        "execution_plan": build_analysis_plan(context, model=state.get("planner_model"))
    }


def execute_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    result = build_analysis_result(
        state["orchestration_state"],
        dataframe=state["dataframe"],
        eda_profiles=state.get("eda_profiles", []),
        question_type=state.get("question_type"),
        execution_plan=state["execution_plan"],
    )
    return {"result": result}


def validate_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    checks = run_analysis_self_check(state["result"])
    passed = all(check.passed or check.severity != "error" for check in checks)
    return {
        "local_checks": checks,
        "terminal_reason": "validated_result" if passed else "validation_failed",
    }


def build_analysis_graph():
    builder = StateGraph(AnalysisWorkflowState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("validate", validate_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "validate")
    builder.add_edge("validate", END)
    return builder.compile()


def run_analysis_workflow(
    state: OrchestrationState,
    dataframe: pd.DataFrame,
    eda_profiles: list[dict[str, Any]],
    *,
    question_type: str | None = None,
    planner_model: Any | None = None,
) -> tuple[dict[str, Any], list[LocalCheck], str]:
    output = build_analysis_graph().invoke({
        "orchestration_state": state,
        "dataframe": dataframe,
        "eda_profiles": eda_profiles,
        "question_type": question_type,
        "planner_model": planner_model,
    })
    return output["result"], output["local_checks"], output["terminal_reason"]
