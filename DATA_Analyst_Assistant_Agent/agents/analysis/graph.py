from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.execute import build_analysis_result
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.finalize import finalize_analysis
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.plan import build_analysis_plan
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.retry import increase_retry
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.validate import run_analysis_self_check
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisExecutionPlan
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
    error: str
    retry_count: int
    max_retries: int
    terminal_reason: str


def plan_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    try:
        orchestration = state["orchestration_state"]
        context = build_analysis_context(
            orchestration,
            state["dataframe"],
            state.get("eda_profiles", []),
            question_type=state.get("question_type"),
        )
        return {
            "execution_plan": build_analysis_plan(context, model=state.get("planner_model")),
            "error": "",
            "terminal_reason": "",
        }
    except Exception as exc:
        return {"error": str(exc), "terminal_reason": "plan_failed"}


def execute_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    try:
        result = build_analysis_result(
            state["orchestration_state"],
            dataframe=state["dataframe"],
            eda_profiles=state.get("eda_profiles", []),
            question_type=state.get("question_type"),
            execution_plan=state["execution_plan"],
        )
        return {"result": result, "error": "", "terminal_reason": ""}
    except Exception as exc:
        return {"error": str(exc), "terminal_reason": "execute_failed"}


def validate_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    checks = run_analysis_self_check(state["result"])
    passed = all(check.passed or check.severity != "error" for check in checks)
    return {
        "local_checks": checks,
        "terminal_reason": "validated_result" if passed else "validation_failed",
    }


def finalize_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    return finalize_analysis(state)


def route_after_plan(state: AnalysisWorkflowState) -> str:
    if state.get("error"):
        return "retry" if state.get("retry_count", 0) < state.get("max_retries", 1) else "finalize"
    return "execute"


def route_after_execute(state: AnalysisWorkflowState) -> str:
    if state.get("error"):
        return "retry" if state.get("retry_count", 0) < state.get("max_retries", 1) else "finalize"
    return "validate"


def route_after_validate(state: AnalysisWorkflowState) -> str:
    failed = state.get("terminal_reason") == "validation_failed"
    if failed and state.get("retry_count", 0) < state.get("max_retries", 1):
        return "retry"
    return "finalize"


def build_analysis_graph():
    builder = StateGraph(AnalysisWorkflowState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("validate", validate_node)
    builder.add_node("retry", increase_retry)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan, {"execute": "execute", "retry": "retry", "finalize": "finalize"})
    builder.add_conditional_edges("execute", route_after_execute, {"validate": "validate", "retry": "retry", "finalize": "finalize"})
    builder.add_conditional_edges("validate", route_after_validate, {"retry": "retry", "finalize": "finalize"})
    builder.add_edge("retry", "plan")
    builder.add_edge("finalize", END)
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
        "retry_count": 0,
        "max_retries": state.max_retry_per_agent,
        "error": "",
    })
    return output["result"], output["local_checks"], output["terminal_reason"]
