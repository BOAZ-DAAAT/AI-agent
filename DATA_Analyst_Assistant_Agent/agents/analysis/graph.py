"""Analysis graph: classify -> analyze (generate/execute/critic loop) -> assemble -> chart -> finalize.

Codegen-first analysis. `classify` extracts intent + a deterministic time grain;
`analyze` runs the generate/execute/critic reflect loop; `assemble` packs the
outcome into the stable AnalysisResult; the chart branch (ported from the legacy
graph) attaches multimodal visual evidence when EDA charts are available.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.analyze import (
    AnalysisOutcome,
    run_analysis,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.assemble import build_result_from_outcome
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.classify import classify_intent
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.finalize import finalize_analysis
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.chart import (
    attach_visual_evidence,
    decide_chart_inspection,
    fetch_chart_artifacts,
    read_chart_artifacts,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisContext, AnalysisIntent
from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck, OrchestrationState


class AnalysisWorkflowState(TypedDict, total=False):
    orchestration_state: OrchestrationState
    dataframe: pd.DataFrame
    eda_profiles: list[dict[str, Any]]
    question_type: str | None
    classify_model: Any | None
    code_generator_model: Any | None
    critic_model: Any | None
    chart_artifact_loader: Any | None
    chart_reader: Any | None
    max_attempts: int
    analysis_context: AnalysisContext
    intent: AnalysisIntent
    outcome: AnalysisOutcome
    result: dict[str, Any]
    chart_requests: list[dict[str, Any]]
    selected_charts: list[dict[str, Any]]
    chart_images: list[dict[str, Any]]
    visual_evidence: list[dict[str, Any]]
    chart_status: str
    local_checks: list[LocalCheck]
    terminal_reason: str
    error: str


def classify_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    try:
        context = build_analysis_context(
            state["orchestration_state"],
            state["dataframe"],
            state.get("eda_profiles", []),
            question_type=state.get("question_type"),
        )
        intent = classify_intent(context, state["dataframe"], model=state.get("classify_model"))
        return {"analysis_context": context, "intent": intent, "error": "", "terminal_reason": ""}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "terminal_reason": "classify_failed"}


def analyze_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    try:
        outcome = run_analysis(
            state["intent"],
            state["analysis_context"],
            state["dataframe"],
            code_generator_model=state.get("code_generator_model"),
            critic_model=state.get("critic_model"),
            max_attempts=state.get("max_attempts", 3),
        )
        return {"outcome": outcome, "error": "", "terminal_reason": ""}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "terminal_reason": "analyze_failed"}


def assemble_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    outcome = state["outcome"]
    result = build_result_from_outcome(
        state["orchestration_state"],
        state["analysis_context"],
        state["intent"],
        outcome,
        state["dataframe"],
        state.get("eda_profiles", []),
    )
    passed = outcome.status in {"passed", "review_required"}
    return {
        "result": result,
        "local_checks": _local_checks(outcome),
        "terminal_reason": "validated_result" if passed else "method_review_failed",
    }


def decide_chart_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    return decide_chart_inspection(state)


def fetch_chart_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    return fetch_chart_artifacts(state)


def read_chart_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    return read_chart_artifacts(state)


def attach_visual_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    return attach_visual_evidence(state)


def finalize_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    return finalize_analysis(state)


def _local_checks(outcome: AnalysisOutcome) -> list[LocalCheck]:
    issues = outcome.critique.method_issues if outcome.critique else []
    method_passed = outcome.status in {"passed", "review_required"}
    method_detail = "; ".join(issues) or (
        "Generated analysis requires review before operational interpretation."
        if outcome.status == "review_required"
        else "Generated analysis must pass adversarial method review."
    )
    return [
        LocalCheck(
            name="analysis_code_executed",
            passed=outcome.result is not None,
            severity="error",
            detail="Generated analysis code must run and produce a result dict.",
        ),
        LocalCheck(
            name="method_review_passed",
            passed=method_passed,
            severity="warning" if outcome.status == "review_required" else "error",
            detail=method_detail,
        ),
    ]


def route_after_classify(state: AnalysisWorkflowState) -> str:
    return "finalize" if state.get("error") else "analyze"


def route_after_analyze(state: AnalysisWorkflowState) -> str:
    return "finalize" if state.get("error") else "assemble"


def route_after_chart_decision(state: AnalysisWorkflowState) -> str:
    return "fetch_chart" if state.get("chart_status") == "needed_available" else "attach_visual"


def route_after_chart_fetch(state: AnalysisWorkflowState) -> str:
    return "read_chart" if state.get("chart_images") else "attach_visual"


def build_analysis_graph():
    builder = StateGraph(AnalysisWorkflowState)
    builder.add_node("classify", classify_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("assemble", assemble_node)
    builder.add_node("decide_chart", decide_chart_node)
    builder.add_node("fetch_chart", fetch_chart_node)
    builder.add_node("read_chart", read_chart_node)
    builder.add_node("attach_visual", attach_visual_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify, {"analyze": "analyze", "finalize": "finalize"})
    builder.add_conditional_edges("analyze", route_after_analyze, {"assemble": "assemble", "finalize": "finalize"})
    builder.add_edge("assemble", "decide_chart")
    builder.add_conditional_edges("decide_chart", route_after_chart_decision, {"fetch_chart": "fetch_chart", "attach_visual": "attach_visual"})
    builder.add_conditional_edges("fetch_chart", route_after_chart_fetch, {"read_chart": "read_chart", "attach_visual": "attach_visual"})
    builder.add_edge("read_chart", "attach_visual")
    builder.add_edge("attach_visual", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=None)


def run_analysis_workflow(
    state: OrchestrationState,
    dataframe: pd.DataFrame,
    eda_profiles: list[dict[str, Any]],
    *,
    question_type: str | None = None,
    planner_model: Any | None = None,
    chart_artifact_loader: Any | None = None,
    chart_reader: Any | None = None,
    code_generator_model: Any | None = None,
    critic_model: Any | None = None,
) -> tuple[dict[str, Any], list[LocalCheck], str]:
    """Run the analysis graph. `planner_model` maps to the classify step."""

    output = build_analysis_graph().invoke({
        "orchestration_state": state,
        "dataframe": dataframe,
        "eda_profiles": eda_profiles,
        "question_type": question_type,
        "classify_model": planner_model,
        "code_generator_model": code_generator_model,
        "critic_model": critic_model,
        "chart_artifact_loader": chart_artifact_loader,
        "chart_reader": chart_reader,
        "max_attempts": state.max_retry_per_agent + 2,
        "error": "",
    }, config={"configurable": {"thread_id": None, "checkpoint_id": None, "checkpoint_ns": ""}})
    return output["result"], output["local_checks"], output["terminal_reason"]
