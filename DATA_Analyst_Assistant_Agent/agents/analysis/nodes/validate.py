from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisResult
from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck


def run_analysis_self_check(result: dict) -> list[LocalCheck]:
    try:
        parsed = AnalysisResult.model_validate(result)
    except Exception as exc:
        return [LocalCheck(name="structured_output_valid", passed=False, severity="error", detail=str(exc))]

    evidence_names = {item.tool_name for item in parsed.evidence}
    planned_names = set(parsed.plan.tool_names)
    expected_kinds = {
        "descriptive": {"descriptive"},
        "comparison": {"group_comparison"},
        "correlation": {"correlation"},
        "trend": {"trend", "group_comparison"},
        "time_series": {"time_series"},
        "prediction": {"regression", "classification"},
        "classification": {"classification"},
        "anomaly_detection": {"anomaly_detection"},
        "causal": {"correlation"},
        "experiment": {"group_comparison"},
        "churn": {"classification"},
        "cohort": {"cohort"},
        "retention": {"cohort"},
        "funnel": {"funnel"},
        "journey": {"journey"},
        "segmentation": {"segmentation"},
        "rfm": {"customer_value"},
        "contribution": {"contribution"},
        "mix_shift": {"mix_shift"},
        "profitability": {"contribution", "group_comparison"},
        "root_cause": {"correlation", "contribution"},
        "survival": {"survival"},
        "text": {"text"},
        "scenario": {"scenario"},
        "unit_economics": {"contribution", "group_comparison"},
        "demand": {"time_series", "regression"},
        "inventory": {"time_series", "anomaly_detection"},
        "attribution": {"contribution"},
        "mmm": {"mmm"},
        "clv": {"clv"},
        "geospatial": {"geospatial"},
        "optimization": {"optimization"},
    }
    allowed_kinds = expected_kinds.get(parsed.plan.question_type)
    return [
        LocalCheck(name="structured_output_valid", passed=True),
        LocalCheck(name="method_summary_present", passed=bool(parsed.method_summary)),
        LocalCheck(name="key_findings_present", passed=bool(parsed.key_findings)),
        LocalCheck(name="limitations_present", passed=bool(parsed.limitations)),
        LocalCheck(
            name="findings_traceable_to_tools",
            passed=not parsed.evidence or evidence_names <= planned_names,
            severity="error",
            detail="Every evidence item must reference a tool selected by the validated plan.",
        ),
        LocalCheck(
            name="all_planned_tools_executed",
            passed=planned_names <= evidence_names,
            severity="error",
            detail="Every tool selected by the LLM plan must produce an evidence item.",
        ),
        LocalCheck(
            name="question_type_matches_analysis_kind",
            passed=allowed_kinds is None or parsed.plan.analysis_kind.value in allowed_kinds,
            severity="error",
            detail=(
                f"question_type={parsed.plan.question_type}, "
                f"analysis_kind={parsed.plan.analysis_kind.value}"
            ),
        ),
        LocalCheck(
            name="human_review_reason_present",
            passed=not parsed.human_review.required or bool(parsed.human_review.reason),
            severity="error",
        ),
    ]
