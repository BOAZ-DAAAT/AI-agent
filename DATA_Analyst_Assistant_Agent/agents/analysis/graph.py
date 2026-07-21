"""Analysis graph: classify -> analyze (generate/execute/critic loop) -> assemble -> chart -> finalize.

Codegen-first analysis. `classify` extracts intent + a deterministic time grain;
`analyze` runs the generate/execute/critic reflect loop; `assemble` packs the
outcome into the stable AnalysisResult; the chart branch (ported from the legacy
graph) attaches multimodal visual evidence when EDA charts are available.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

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
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    AnalysisSelectionResponse,
    ReviewRequest,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck, OrchestrationState


class AnalysisWorkflowState(TypedDict, total=False):
    orchestration_state: OrchestrationState
    dataframe: pd.DataFrame
    eda_profiles: list[dict[str, Any]]
    question_type: str | None
    selection_response: AnalysisSelectionResponse | None
    review_request: ReviewRequest | None
    classify_model: Any | None
    code_generator_model: Any | None
    critic_model: Any | None
    progress_callback: Callable[[str, str, int], None] | None
    chart_artifact_loader: Any | None
    chart_reader: Any | None
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
            selection_response=state.get("selection_response"),
            review_request=state.get("review_request"),
        )
        contract_blockers = _analysis_contract_blockers(context)
        if contract_blockers:
            return {
                "analysis_context": context,
                "error": "분석 입력 계약이 불충분합니다: " + "; ".join(contract_blockers),
                "terminal_reason": "analysis_contract_invalid",
            }
        intent = classify_intent(context, state["dataframe"], model=state.get("classify_model"))
        return {"analysis_context": context, "intent": intent, "error": "", "terminal_reason": ""}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "terminal_reason": "classify_failed"}


def _analysis_contract_blockers(context: AnalysisContext) -> list[str]:
    contract = context.analysis_data_contract or {}
    if not contract:
        return []
    has_sql_datamart = bool(contract.get("target_table") or contract.get("generated_sql"))
    if not has_sql_datamart:
        return []
    blockers: list[str] = []
    if not str(contract.get("row_grain") or "").strip():
        blockers.append("SQL 데이터마트의 row_grain이 명확하지 않습니다")
    if not contract.get("derived_columns") and not context.mart_columns:
        blockers.append("SQL 데이터마트 컬럼의 원본/파생 관계가 명확하지 않습니다")
    return blockers


def analyze_node(state: AnalysisWorkflowState) -> dict[str, Any]:
    try:
        outcome = run_analysis(
            state["intent"],
            state["analysis_context"],
            state["dataframe"],
            code_generator_model=state.get("code_generator_model"),
            critic_model=state.get("critic_model"),
            max_attempts=state["orchestration_state"].max_retry_per_agent + 1,
            progress_callback=state.get("progress_callback"),
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
        "terminal_reason": "validated_result" if passed else _failure_terminal_reason(outcome),
    }


# Maps the stage of the *last recorded* attempt failure to a terminal_reason.
# A repeated generate/execute/result_contract failure never reached the
# critic, so it must not be reported as "method_review_failed" (run
# 019f7970: a sandboxed `time` import block made every attempt fail at
# execute; run 019f7a58: unparseable structured JSON output made every
# attempt fail at generate -- both used to surface as the same generic
# "method review failed" label, sending debugging on a wild goose chase).
_STAGE_TERMINAL_REASONS = {
    "generate": "generation_failed",
    "execute": "execution_failed",
    "result_contract": "result_contract_failed",
    "critic": "method_review_failed",
}


def _failure_terminal_reason(outcome: AnalysisOutcome) -> str:
    if outcome.error_history:
        stage = str(outcome.error_history[-1].get("stage") or "")
        reason = _STAGE_TERMINAL_REASONS.get(stage)
        if reason:
            return reason
    return "method_review_failed"


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
    execution_detail = "Generated analysis code must run and produce a result dict."
    if outcome.error_history and outcome.error_history[-1].get("stage") in {"generate", "execute", "result_contract"}:
        last_failure = outcome.error_history[-1]
        execution_detail = f"[{last_failure.get('stage')}] {last_failure.get('error')}"
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
            detail=execution_detail,
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
    selection_response: AnalysisSelectionResponse | None = None,
    review_request: ReviewRequest | None = None,
    planner_model: Any | None = None,
    chart_artifact_loader: Any | None = None,
    chart_reader: Any | None = None,
    code_generator_model: Any | None = None,
    critic_model: Any | None = None,
    progress_callback: Callable[[str, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[LocalCheck], str]:
    """Run the analysis graph. `planner_model` maps to the classify step."""

    output = build_analysis_graph().invoke({
        "orchestration_state": state,
        "dataframe": dataframe,
        "eda_profiles": eda_profiles,
        "question_type": question_type,
        "selection_response": selection_response,
        "review_request": review_request,
        "classify_model": planner_model,
        "code_generator_model": code_generator_model,
        "critic_model": critic_model,
        "progress_callback": progress_callback,
        "chart_artifact_loader": chart_artifact_loader,
        "chart_reader": chart_reader,
        "error": "",
    }, config={"configurable": {"thread_id": None, "checkpoint_id": None, "checkpoint_ns": ""}})
    return output["result"], output["local_checks"], output["terminal_reason"]
