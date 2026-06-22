from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from DATA_Analyst_Assistant_Agent.agents.analysis.catalog import CAPABILITIES
from DATA_Analyst_Assistant_Agent.agents.analysis.prompts import (
    ANALYSIS_PLANNER_SYSTEM_PROMPT,
    TOOL_CATALOG,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisContext, AnalysisExecutionPlan
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


ALLOWED_TOOLS = {tool for capability in CAPABILITIES.values() for tool in capability.tools} | {"analyze_trend"}

COLUMN_PARAMETER_KEYS = {
    "metric",
    "target",
    "dimension",
    "time_column",
    "entity_id_column",
    "event_time_column",
    "cohort_time_column",
    "event_column",
    "customer_id_column",
    "amount_column",
    "duration_column",
    "event_observed_column",
    "text_column",
    "period_column",
    "date_column",
    "outcome_column",
    "latitude_column",
    "longitude_column",
    "item_column",
    "value_column",
    "cost_column",
    "datetime_column",
    "monetary_value_column",
}
COLUMN_LIST_PARAMETER_KEYS = {"features", "columns", "channel_columns", "control_columns"}


def build_analysis_plan(context: AnalysisContext, *, model: Any | None = None) -> AnalysisExecutionPlan:
    """Always call an LLM for a structured plan and fail on invalid output."""

    chat_model = model or get_chat_model(temperature=0)
    structured_model = chat_model.with_structured_output(AnalysisExecutionPlan)
    result = structured_model.invoke([
        SystemMessage(content=ANALYSIS_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"{TOOL_CATALOG}\n\nAnalysis context:\n{context.model_dump_json(indent=2)}"),
    ])
    plan = result if isinstance(result, AnalysisExecutionPlan) else AnalysisExecutionPlan.model_validate(result)
    return validate_analysis_plan(plan, context)


def validate_analysis_plan(plan: AnalysisExecutionPlan, context: AnalysisContext) -> AnalysisExecutionPlan:
    """Reject hallucinated tools, columns, or supervisor question-type changes."""

    unknown_tools = set(plan.tool_names) - ALLOWED_TOOLS
    if unknown_tools:
        raise ValueError(f"LLM plan selected unknown tools: {sorted(unknown_tools)}")
    if not plan.tool_names:
        raise ValueError(plan.review_reason or "LLM plan did not select an executable analysis tool.")

    canonical_type = normalize_question_type(context.question_type)
    if canonical_type and plan.question_type != canonical_type:
        raise ValueError(
            f"LLM plan changed supervisor question_type from {canonical_type} to {plan.question_type}."
        )

    _require_column("metric", plan.metric, context.columns)
    _require_column("dimension", plan.dimension, context.columns)
    _require_column("time_column", plan.time_column, context.columns)
    unknown_features = set(plan.feature_columns) - set(context.columns)
    if unknown_features:
        raise ValueError(f"LLM plan referenced unknown feature columns: {sorted(unknown_features)}")
    unknown_parameter_tools = set(plan.tool_parameters) - set(plan.tool_names)
    if unknown_parameter_tools:
        raise ValueError(
            f"LLM plan supplied parameters for unselected tools: {sorted(unknown_parameter_tools)}"
        )
    for tool_name, parameters in plan.tool_parameters.items():
        _validate_tool_parameters(tool_name, parameters, context.columns)

    plan.planner_mode = "llm"
    return plan


def normalize_question_type(value: str | None) -> str | None:
    if not value:
        return None
    aliases = {
        "predict": "prediction", "forecast": "prediction", "regression": "prediction", "\uc608\uce21": "prediction",
        "classify": "classification", "\ubd84\ub958": "classification",
        "anomaly": "anomaly_detection", "outlier": "anomaly_detection", "\uc774\uc0c1\ud0d0\uc9c0": "anomaly_detection",
        "compare": "comparison", "group_difference": "comparison", "\ube44\uad50": "comparison",
        "relationship": "correlation", "\uc0c1\uad00": "correlation", "\uad00\uacc4": "correlation",
        "time_series": "trend", "\ucd94\uc774": "trend", "\uc2dc\uacc4\uc5f4": "trend",
        "cause": "causal", "impact": "causal", "\uc778\uacfc": "causal", "\uc6d0\uc778": "causal",
        "timeseries": "time_series", "time_series_analysis": "time_series",
        "cohort_analysis": "cohort", "\ucf54\ud638\ud2b8": "cohort",
        "retention_analysis": "retention", "\ub9ac\ud150\uc158": "retention",
        "funnel_analysis": "funnel", "\ud37c\ub110": "funnel",
        "customer_journey": "journey", "path": "journey",
        "segment": "segmentation", "clustering": "segmentation", "\uc138\uadf8\uba3c\ud2b8": "segmentation",
        "customer_value": "rfm", "clv": "rfm",
        "pareto": "contribution", "concentration": "contribution", "\uae30\uc5ec\ub3c4": "contribution",
        "ab_test": "experiment", "a/b": "experiment", "\uc2e4\ud5d8": "experiment",
        "time_to_event": "survival", "\uc0dd\uc874": "survival",
        "voc": "text", "review_text": "text", "\ud14d\uc2a4\ud2b8": "text",
        "what_if": "scenario", "simulation": "scenario", "\uc2dc\ub098\ub9ac\uc624": "scenario",
        "unit_economics_analysis": "unit_economics", "\uc720\ub2db\uc774\ucf54\ub178\ubbf9\uc2a4": "unit_economics",
        "demand_forecast": "demand", "\uc218\uc694": "demand",
        "stock_analysis": "inventory", "\uc7ac\uace0": "inventory",
        "channel_attribution": "attribution", "\uc5b4\ud2b8\ub9ac\ubdf0\uc158": "attribution",
        "marketing_mix": "mmm", "marketing_mix_model": "mmm",
        "customer_lifetime_value": "clv", "lifetime_value": "clv",
        "spatial": "geospatial", "geo": "geospatial", "\uacf5\uac04\ubd84\uc11d": "geospatial",
        "allocation": "optimization", "optimize": "optimization", "\ucd5c\uc801\ud654": "optimization",
    }
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return aliases.get(normalized, normalized)


def _require_column(name: str, value: str | None, columns: list[str]) -> None:
    if value is not None and value not in columns:
        raise ValueError(f"LLM plan referenced unknown {name}: {value}")


def _validate_tool_parameters(tool_name: str, parameters: dict[str, Any], columns: list[str]) -> None:
    for key, value in parameters.items():
        if key in COLUMN_PARAMETER_KEYS and value is not None and value not in columns:
            raise ValueError(f"LLM plan referenced unknown column in {tool_name}.{key}: {value}")
        if key in COLUMN_LIST_PARAMETER_KEYS:
            unknown = set(value or []) - set(columns)
            if unknown:
                raise ValueError(
                    f"LLM plan referenced unknown columns in {tool_name}.{key}: {sorted(unknown)}"
                )
