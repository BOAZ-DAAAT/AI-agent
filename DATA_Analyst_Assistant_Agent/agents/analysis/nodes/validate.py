from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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
        "general_task": {"general_task"},
    }
    allowed_kinds = expected_kinds.get(parsed.plan.question_type)
    checks = [
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
    checks.extend(_run_methodology_checks(parsed))
    return checks


def _run_methodology_checks(result: AnalysisResult) -> list[LocalCheck]:
    return [
        *_check_tool_inputs_match_plan(result),
        *_check_method_requirements(result),
        *_check_claims_are_evidence_backed(result),
        *_check_limitations_match_method(result),
    ]


def _check_tool_inputs_match_plan(result: AnalysisResult) -> list[LocalCheck]:
    mismatches: list[str] = []
    expected = {
        "metric": result.plan.metric,
        "dimension": result.plan.dimension,
        "time_column": result.plan.time_column,
        "feature_columns": result.plan.feature_columns,
    }
    for evidence in result.evidence:
        for key, expected_value in expected.items():
            if expected_value in (None, [], {}):
                continue
            if evidence.inputs.get(key) != expected_value:
                mismatches.append(f"{evidence.tool_name}.{key}")
    return [
        LocalCheck(
            name="tool_parameters_match_plan",
            passed=not mismatches,
            severity="error",
            detail=(
                "Evidence inputs must preserve the validated analysis plan parameters. "
                f"Mismatches: {', '.join(mismatches)}"
            ),
        )
    ]


def _check_method_requirements(result: AnalysisResult) -> list[LocalCheck]:
    evidence_by_tool = {item.tool_name: item for item in result.evidence}
    requirements = _method_requirements()
    relevant = requirements.get(result.plan.analysis_kind.value, {})
    missing: list[str] = []
    for tool_name, required_keys in relevant.items():
        evidence = evidence_by_tool.get(tool_name)
        if evidence is None:
            if tool_name in result.plan.tool_names:
                missing.append(f"{tool_name}: evidence missing")
            continue
        absent = sorted(key for key in required_keys if not _has_path(evidence.statistics, key))
        if absent:
            missing.append(f"{tool_name}: {', '.join(absent)}")

    return [
        LocalCheck(
            name="analysis_method_requirements_met",
            passed=not missing,
            severity="error",
            detail=(
                "Executed evidence must contain the minimum statistics expected for the planned "
                f"{result.plan.analysis_kind.value} method. Missing: {'; '.join(missing)}"
            ),
        )
    ]


def _check_claims_are_evidence_backed(result: AnalysisResult) -> list[LocalCheck]:
    evidence_findings = [item.finding for item in result.evidence if item.finding]
    unsupported = [
        finding
        for finding in evidence_findings
        if not _contains_normalized(result.key_findings, finding)
    ]
    return [
        LocalCheck(
            name="finding_claims_supported",
            passed=not unsupported,
            severity="error",
            detail=(
                "Every computed evidence finding should be carried into key_findings without "
                f"being dropped. Missing findings: {' | '.join(unsupported)}"
            ),
        )
    ]


def _check_limitations_match_method(result: AnalysisResult) -> list[LocalCheck]:
    text = " ".join([
        *result.limitations,
        *(caveat for evidence in result.evidence for caveat in evidence.caveats),
    ]).casefold()
    expectations = {
        "correlation": ("causation", "causal"),
        "regression": ("holdout", "validation"),
        "classification": ("holdout", "validation"),
        "anomaly_detection": ("review", "proof"),
        "time_series": ("frequency", "seasonality", "history"),
        "cohort": ("entity identifier", "event logging"),
        "funnel": ("step", "event ordering"),
        "segmentation": ("stability", "descriptive"),
        "customer_value": ("historical", "forecast"),
        "contribution": ("causal", "causality"),
        "mix_shift": ("causal", "causality"),
        "survival": ("censoring",),
        "text": ("semantic", "interpretation"),
        "scenario": ("assumption",),
        "geospatial": ("spatial scale", "privacy", "neighbor"),
        "optimization": ("approval", "operational"),
    }
    expected_terms = expectations.get(result.plan.analysis_kind.value, ())
    passed = not expected_terms or any(term in text for term in expected_terms)
    return [
        LocalCheck(
            name="limitations_match_analysis_kind",
            passed=passed,
            severity="warning",
            detail=(
                "Method-specific caveats should be present in limitations or evidence caveats. "
                f"Expected one of: {', '.join(expected_terms)}"
            ),
        )
    ]


def _method_requirements() -> dict[str, dict[str, set[str]]]:
    return {
        "descriptive": {"describe_metric": {"count", "mean", "min", "max"}},
        "group_comparison": {
            "compare_groups": {"groups", "group_count"},
            "test_group_difference": {"method", "p_value", "alpha", "significant"},
        },
        "correlation": {"measure_correlation": {"pairs", "observation_count"}},
        "trend": {"analyze_trend": {"direction", "slope_p_value", "r_squared", "observation_count"}},
        "regression": {"fit_regression_model": {"evaluation_scope", "test_r2", "test_mae", "test_rmse"}},
        "classification": {
            "fit_classification_model": {
                "evaluation_scope",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_f1_weighted",
            }
        },
        "anomaly_detection": {"detect_anomalies": {"anomaly_count", "anomaly_rate", "top_anomaly_candidates"}},
        "time_series": {"analyze_time_series": {"period_count", "frequency", "trend_p_value"}},
        "cohort": {"analyze_cohort_retention": {"cohort_count", "entity_count", "retention_matrix"}},
        "funnel": {"analyze_funnel": {"steps", "overall_conversion_rate"}},
        "journey": {"analyze_journey": {"entity_count", "top_paths"}},
        "segmentation": {"segment_entities": {"n_clusters", "entity_count", "silhouette_score"}},
        "customer_value": {"analyze_rfm": {"customer_count", "segments"}},
        "contribution": {"analyze_contribution": {"group_count", "pareto_80_group_count"}},
        "mix_shift": {"analyze_mix_shift": {"previous_period", "current_period", "total_delta"}},
        "survival": {"analyze_survival": {"event_count", "observation_count"}},
        "text": {"analyze_text": {"document_count", "token_count", "top_terms"}},
        "scenario": {"simulate_scenario": {"baseline", "projected", "change_percent", "assumption"}},
        "mmm": {"run_bayesian_mmm": {"period_count", "channel_contribution_shares"}},
        "clv": {"estimate_probabilistic_clv": {"customer_count", "mean_expected_clv"}},
        "geospatial": {"analyze_geospatial_hotspots": {"point_count", "global_moran_i", "global_p_value_simulated"}},
        "optimization": {"optimize_business_allocation": {"status", "objective_value", "budget_used", "allocations"}},
        "general_task": {"code_generator": {"statistics"}},
    }


def _has_path(payload: dict[str, Any], path: str) -> bool:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return current not in (None, [], {})


def _contains_normalized(items: Iterable[str], needle: str) -> bool:
    normalized = _normalize_text(needle)
    return any(normalized in _normalize_text(item) for item in items)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).casefold().split())
