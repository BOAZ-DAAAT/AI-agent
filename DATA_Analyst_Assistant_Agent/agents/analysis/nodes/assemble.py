"""Bridge the analysis loop's AnalysisOutcome to the stable AnalysisResult.

The output contract (AnalysisResult) is consumed by chart/finalize/report and by
agent.py's preview, so it must stay shape-compatible. This module packs the
generated-code outcome into that contract and fills the new codegen fields
(intent/generated_code/code_critique/codegen_attempts) additively.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.analyze import AnalysisOutcome
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.coverage import build_answer_coverage
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisEvidence,
    AnalysisExecutionPlan,
    AnalysisIntent,
    AnalysisKind,
    AnalysisResult,
    HumanReview,
    ReviewRequest,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState

GENERATED_TOOL_NAME = "generated_analysis"


def _plan_from_intent(intent: AnalysisIntent) -> AnalysisExecutionPlan:
    """Minimal plan so downstream readers of result.plan keep working."""

    return AnalysisExecutionPlan(
        objective=intent.objective,
        question_type="general_task",
        analysis_kind=AnalysisKind.general_task,
        analysis_subtype=f"generated::{intent.domain}",
        tool_names=[GENERATED_TOOL_NAME],
        metric=intent.metric_hints[0] if intent.metric_hints else None,
        dimension=intent.dimension_hints[0] if intent.dimension_hints else None,
        time_column=intent.time_column,
        feature_columns=list(intent.metric_hints),
        requires_human_review=intent.requires_human_review,
        review_reason=intent.review_reason,
        planner_mode="llm",
    )


def _evidence_from_outcome(outcome: AnalysisOutcome) -> AnalysisEvidence:
    result = outcome.result or {}
    findings = result.get("findings") or []
    finding = findings[0] if findings else (result.get("summary") or "No finding produced.")
    status = {
        "passed": "success",
        "review_required": "review_required",
        "failed": "failed",
    }.get(outcome.status, "failed")
    return AnalysisEvidence(
        tool_name=GENERATED_TOOL_NAME,
        status=status,
        method="LLM-generated analysis code",
        inputs={},
        statistics=result.get("statistics") or {},
        finding=str(finding),
        caveats=[str(item) for item in (result.get("limitations") or [])],
    )


def build_result_from_outcome(
    state: OrchestrationState,
    context: AnalysisContext,
    intent: AnalysisIntent,
    outcome: AnalysisOutcome,
    dataframe: pd.DataFrame,
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    result_payload = outcome.result or {}
    evidence = _evidence_from_outcome(outcome)

    findings: list[str] = [str(item) for item in (result_payload.get("findings") or [])]
    data_load_note = ""
    if not dataframe.empty:
        data_load_note = f"SQL result contains {len(dataframe)} rows and {len(dataframe.columns)} columns."

    limitations = [str(item) for item in (result_payload.get("limitations") or [])]
    method_notes = [str(item) for item in (result_payload.get("method_notes") or [])]
    review_request = _review_request_from_payload(result_payload.get("review_request"))
    if outcome.status == "failed":
        reason = outcome.critique.feedback if outcome.critique else "did not pass method review"
        limitations.append(f"Analysis did not pass method review after {outcome.attempts} attempts: {reason}")
        if outcome.early_stop_reason:
            limitations.append(f"Analysis stopped early: {outcome.early_stop_reason}.")
    elif outcome.status == "review_required" and outcome.critique and review_request is None:
        reason = outcome.critique.feedback or "; ".join(outcome.critique.method_issues)
        if reason:
            method_notes.append(reason)
    limitations.extend(_eda_limitations(profiles))

    review_required = intent.requires_human_review or review_request is not None or outcome.status == "failed"
    review_reason = _review_reason(intent, outcome, review_request)
    status = {
        "passed": "success",
        "review_required": "review_required" if review_request is not None else "success",
        "failed": "failed",
    }.get(outcome.status, "failed")
    quality_notes = _quality_notes(dataframe, profiles)
    if data_load_note:
        quality_notes.insert(0, data_load_note)

    result = AnalysisResult(
        run_id=state.run_id,
        goal=context.goal,
        plan=_plan_from_intent(intent),
        method_summary=(
            result_payload.get("summary")
            or f"Generated analysis for a {intent.domain} question ({outcome.status})."
        ),
        key_findings=findings or ["No usable findings were produced."],
        evidence=[evidence],
        hypotheses=_hypothesis_summaries(result_payload.get("hypothesis_tests") or []),
        limitations=list(dict.fromkeys(limitations)) or ["No limitations were recorded."],
        source_artifacts={
            "sql": state.artifact_ids.get("sql_agent", []),
            "eda": state.artifact_ids.get("eda_agent", []),
        },
        data_quality_notes=list(dict.fromkeys(quality_notes)),
        eda_profile_summaries=profiles,
        human_review=HumanReview(required=review_required, reason=review_reason),
        answer_coverage=build_answer_coverage(intent, context, outcome.code, outcome.result),
        status=status,
        title=_title_for_intent(intent),
        executive_summary=str(result_payload.get("summary") or ""),
        hypothesis_tests=result_payload.get("hypothesis_tests") or [],
        evidence_tables=result_payload.get("evidence_tables") or [],
        interpretation=[str(item) for item in (result_payload.get("interpretation") or [])],
        review_request=review_request,
        method_notes=list(dict.fromkeys(method_notes)),
        intent=intent,
        generated_code=(outcome.code.code if outcome.code else ""),
        code_critique=outcome.critique,
        codegen_attempts=outcome.attempts,
    )
    return result.model_dump(mode="json")


def _review_request_from_payload(value: Any) -> ReviewRequest | None:
    if not isinstance(value, dict):
        return None
    try:
        request = ReviewRequest.model_validate(value)
    except Exception:
        return None
    if not request.question.strip() or not request.proposal.strip():
        return None
    if not any(option.strip() for option in request.options):
        return None
    return request


def _review_reason(
    intent: AnalysisIntent,
    outcome: AnalysisOutcome,
    review_request: ReviewRequest | None,
) -> str:
    if intent.review_reason:
        return intent.review_reason
    if review_request is not None and review_request.question.strip():
        return review_request.question
    if outcome.status == "failed":
        return "Generated analysis failed method review; human check recommended."
    return ""


def _title_for_intent(intent: AnalysisIntent) -> str:
    domain = (intent.domain or "general").replace("_", " ").strip().title()
    return f"{domain} Analysis"


def _hypothesis_summaries(tests: list[dict[str, Any]]) -> list[Any]:
    summaries: list[Any] = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        decision = item.get("decision")
        if decision not in {"supported", "inconclusive", "not_supported"}:
            decision = "inconclusive"
        summaries.append({
            "null_hypothesis": str(item.get("null_hypothesis") or ""),
            "alternative_hypothesis": str(item.get("alternative_hypothesis") or item.get("hypothesis") or ""),
            "decision": decision,
            "rationale": _hypothesis_rationale(item),
        })
    return summaries


def _hypothesis_rationale(item: dict[str, Any]) -> str:
    parts = [str(item.get("test_name") or "hypothesis test")]
    if item.get("p_value") is not None:
        parts.append(f"p={item['p_value']}")
    if item.get("effect_size") is not None:
        parts.append(f"effect_size={item['effect_size']}")
    caveats = item.get("caveats") or []
    if caveats:
        parts.append("; ".join(str(c) for c in caveats))
    return "; ".join(parts)


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
