from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.insight import build_hypotheses, evidence_from_payload
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisExecutionPlan,
    AnalysisKind,
    AnalysisResult,
    HumanReview,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.tools import ANALYSIS_TOOLS
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


def build_analysis_result(
    state: OrchestrationState,
    *,
    dataframe: pd.DataFrame | None = None,
    eda_profiles: list[dict[str, Any]] | None = None,
    question_type: str | None = None,
    execution_plan: AnalysisExecutionPlan | None = None,
    code_generator_model: Any | None = None,
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
            payload = _invoke_tool(
                tool_name,
                records,
                plan,
                state=state,
                dataframe=df,
                context=context,
                code_generator_model=code_generator_model,
            )
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
        limitations.extend(_eda_limitations(profiles))
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


def _invoke_tool(
    tool_name: str,
    records: list[dict[str, Any]],
    plan,
    *,
    state: OrchestrationState,
    dataframe: pd.DataFrame,
    context: Any,
    code_generator_model: Any | None = None,
) -> dict[str, Any]:
    if tool_name == "code_generator":
        return _run_code_generator(
            state=state,
            dataframe=dataframe,
            context=context,
            plan=plan,
            model=code_generator_model,
        )
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


def _run_code_generator(
    *,
    state: OrchestrationState,
    dataframe: pd.DataFrame,
    context: Any,
    plan: AnalysisExecutionPlan,
    model: Any | None = None,
) -> dict[str, Any]:
    chat_model = model or get_chat_model(model_env="CODE_GENERATOR_MODEL", temperature=0)
    prompt = (
        "Write concise Python code to answer the user's data-analysis request.\n"
        "Use the provided pandas DataFrame named df. pandas is available as pd.\n"
        "Do not read or write files, call networks, or mutate external state.\n"
        "Set a variable named result to a dict with keys: summary, findings, statistics, limitations.\n"
        "Return JSON only with keys python_code and rationale.\n\n"
        f"User question: {state.user_query}\n"
        f"Execution plan: {plan.model_dump_json(indent=2)}\n"
        f"Analysis context: {context.model_dump_json(indent=2)}"
    )
    response = chat_model.invoke([
        SystemMessage(content="You generate small, auditable pandas analysis code."),
        HumanMessage(content=prompt),
    ])
    content = getattr(response, "content", response)
    code_spec = _parse_code_generator_response(str(content))
    python_code = code_spec.get("python_code", "").strip()
    if not python_code:
        raise ValueError("CODE_GENERATOR_MODEL did not return python_code.")

    result = _execute_generated_analysis_code(python_code, dataframe)
    findings = result.get("findings") or []
    if isinstance(findings, str):
        findings = [findings]
    limitations = result.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    return {
        "summary": str(result.get("summary") or code_spec.get("rationale") or "Generated code analysis completed."),
        "findings": [str(item) for item in findings],
        "statistics": result.get("statistics") or {},
        "limitations": [str(item) for item in limitations],
        "generated_code": python_code,
        "model_env": "CODE_GENERATOR_MODEL",
    }


def _parse_code_generator_response(content: str) -> dict[str, Any]:
    import json

    cleaned = content.strip()
    cleaned = cleaned.replace("```json", "").replace("```python", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"python_code": cleaned, "rationale": ""}
    if not isinstance(parsed, dict):
        raise ValueError("CODE_GENERATOR_MODEL response must be a JSON object.")
    return parsed


def _execute_generated_analysis_code(python_code: str, dataframe: pd.DataFrame) -> dict[str, Any]:
    import math

    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - numpy is expected in the project env
        np = None

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
    }
    globals_dict = {"__builtins__": safe_builtins, "pd": pd, "np": np, "math": math}
    locals_dict: dict[str, Any] = {"df": dataframe.copy(), "result": None}
    try:
        exec(python_code, globals_dict, locals_dict)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"generated code failed: {exc}") from exc
    result = locals_dict.get("result")
    if not isinstance(result, dict):
        raise ValueError("generated code must set result to a dict.")
    return result


def _method_summary(kind: AnalysisKind, tools: list[str]) -> str:
    names = ", ".join(tools) if tools else "no statistical tools"
    return f"Executed a {kind.value} analysis using {names}; all claims are tied to computed artifact evidence."


def _quality_notes(df: pd.DataFrame, profiles: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for profile in profiles:
        profile_block = profile.get("profile", profile)
        status = profile_block.get("quality_status")
        if status:
            notes.append(f"EDA quality status: {status}.")
        notes.extend(str(item) for item in profile_block.get("key_issues", []) or [])
        for caution in profile.get("cautions", []) or []:
            if isinstance(caution, dict) and caution.get("message_ko"):
                notes.append(f"EDA caution ({caution.get('severity', 'unknown')}): {caution['message_ko']}")
    for column, count in df.isna().sum().items():
        if int(count) > 0:
            notes.append(f"Column {column} contains {int(count)} missing values.")
    return list(dict.fromkeys(notes))


def _eda_limitations(profiles: list[dict[str, Any]]) -> list[str]:
    limitations: list[str] = []
    for profile in profiles:
        data_level = profile.get("data_level", {}) or {}
        if data_level.get("is_aggregated"):
            limitations.append(
                "EDA indicates aggregated data; avoid individual customer/order/product-level interpretation."
            )
        for constraint in profile.get("analysis_constraints", []) or []:
            blocked = ", ".join(str(item) for item in constraint.get("blocked_operations", []) or [])
            reason = constraint.get("reason_ko") or "EDA analysis constraint applies."
            if blocked:
                limitations.append(f"{reason} Blocked operations: {blocked}.")
        for caution in profile.get("cautions", []) or []:
            if isinstance(caution, dict) and caution.get("implication") == "avoid_causal_claims":
                limitations.append("Observed associations should not be phrased as causal effects.")
    return list(dict.fromkeys(limitations))
