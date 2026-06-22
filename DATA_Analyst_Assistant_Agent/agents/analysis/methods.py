from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.insight import build_hypotheses, evidence_from_payload
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisExecutionPlan,
    AnalysisKind,
    AnalysisResult,
    HumanReview,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.tools import ANALYSIS_TOOLS
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def build_analysis_result(
    state: OrchestrationState,
    *,
    dataframe: pd.DataFrame | None = None,
    eda_profiles: list[dict[str, Any]] | None = None,
    question_type: str | None = None,
    execution_plan: AnalysisExecutionPlan | None = None,
) -> dict[str, Any]:
    df = dataframe if dataframe is not None else pd.DataFrame()
    profiles = eda_profiles or []
    context = build_analysis_context(state, df, profiles, question_type=question_type)
    if execution_plan is None:
        raise ValueError("execution_plan is required; the workflow must run the LLM plan node first.")
    plan = execution_plan
    records = df.where(pd.notna(df), None).to_dict(orient="records") if not df.empty else []

    evidence: list[AnalysisEvidence] = []
    limitations = [
        "Results describe the registered SQL result artifacts for this run only.",
        "Observed patterns do not by themselves establish causality.",
    ]
    for tool_name in plan.tool_names:
        try:
            payload = _invoke_tool(tool_name, records, plan)
            evidence.append(evidence_from_payload(tool_name, payload, plan))
        except (TypeError, ValueError) as exc:
            limitations.append(f"{tool_name} was not executed: {exc}")

    findings = []
    if not df.empty:
        findings.append(f"SQL result contains {len(df)} rows and {len(df.columns)} columns.")
    findings.extend(item.finding for item in evidence)
    if profiles:
        findings.append("EDA profile evidence was considered before interpreting the analysis results.")
        limitations.append("EDA profile artifacts summarize data quality signals and do not replace full validation.")
    if df.empty:
        findings.append("No usable rows were available in the SQL result artifacts.")
        limitations.append("Statistical tools could not run because the SQL result was empty or unreadable.")
    elif not findings:
        findings.append(f"SQL result contains {len(df)} rows and {len(df.columns)} columns.")

    quality_notes = _quality_notes(df, profiles)
    result = AnalysisResult(
        run_id=state.run_id,
        goal=context.goal,
        plan=plan,
        method_summary=_method_summary(plan.analysis_kind, [item.tool_name for item in evidence]),
        key_findings=findings,
        evidence=evidence,
        hypotheses=build_hypotheses(plan.analysis_kind, evidence),
        limitations=list(dict.fromkeys(limitations)),
        source_artifacts={
            "sql": state.artifact_ids.get("sql_agent", []),
            "eda": state.artifact_ids.get("eda_agent", []),
        },
        data_quality_notes=quality_notes,
        eda_profile_summaries=profiles,
        human_review=HumanReview(required=plan.requires_human_review, reason=plan.review_reason),
    )
    return result.model_dump(mode="json")


def _invoke_tool(tool_name: str, records: list[dict[str, Any]], plan) -> dict[str, Any]:
    args: dict[str, Any] = {"records": records, **plan.tool_parameters.get(tool_name, {})}
    if tool_name == "describe_metric":
        args.setdefault("metric", plan.metric)
    elif tool_name == "compare_groups":
        args.setdefault("metric", plan.metric)
        args.setdefault("dimension", plan.dimension)
    elif tool_name == "test_group_difference":
        args.setdefault("metric", plan.metric)
        args.setdefault("dimension", plan.dimension)
    elif tool_name == "measure_correlation":
        args.setdefault("columns", plan.feature_columns)
    elif tool_name == "analyze_trend":
        args.setdefault("metric", plan.metric)
        args.setdefault("time_column", plan.time_column)
    elif tool_name == "fit_regression_model":
        args.setdefault("target", plan.metric)
        args.setdefault("features", plan.feature_columns)
    elif tool_name == "fit_classification_model":
        args.setdefault("target", plan.metric)
        args.setdefault("features", plan.feature_columns)
    elif tool_name == "detect_anomalies":
        args.setdefault("features", plan.feature_columns)
    elif tool_name == "analyze_time_series":
        args.setdefault("metric", plan.metric)
        args.setdefault("time_column", plan.time_column)
    elif tool_name == "segment_entities":
        args.setdefault("features", plan.feature_columns)
    elif tool_name == "analyze_contribution":
        args.setdefault("metric", plan.metric)
        args.setdefault("dimension", plan.dimension)
    elif tool_name == "simulate_scenario":
        args.setdefault("metric", plan.metric)
    return ANALYSIS_TOOLS[tool_name].invoke(args)


def _method_summary(kind: AnalysisKind, tools: list[str]) -> str:
    names = ", ".join(tools) if tools else "no statistical tools"
    return f"Executed a {kind.value} analysis using {names}; all claims are tied to computed artifact evidence."


def _quality_notes(df: pd.DataFrame, profiles: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for profile in profiles:
        status = profile.get("quality_status")
        if status:
            notes.append(f"EDA quality status: {status}.")
        notes.extend(str(item) for item in profile.get("key_issues", []) or [])
    for column, count in df.isna().sum().items():
        if int(count) > 0:
            notes.append(f"Column {column} contains {int(count)} missing values.")
    return list(dict.fromkeys(notes))
