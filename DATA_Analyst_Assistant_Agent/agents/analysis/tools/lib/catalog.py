from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisCapability:
    question_type: str
    sector: str
    purpose: str
    tools: tuple[str, ...]
    required_roles: tuple[str, ...]


CAPABILITIES: dict[str, AnalysisCapability] = {
    "descriptive": AnalysisCapability(
        "descriptive", "common", "Summarize the current state of a metric.",
        ("describe_metric",), ("metric",),
    ),
    "comparison": AnalysisCapability(
        "comparison", "common", "Compare a metric across groups.",
        ("compare_groups", "test_group_difference"), ("metric", "dimension"),
    ),
    "experiment": AnalysisCapability(
        "experiment", "product", "Evaluate an A/B or multi-arm experiment.",
        ("test_group_difference",), ("metric", "variant"),
    ),
    "correlation": AnalysisCapability(
        "correlation", "common", "Measure associations among numeric variables.",
        ("measure_correlation",), ("features",),
    ),
    "time_series": AnalysisCapability(
        "time_series", "common", "Analyze growth, trend, volatility, and seasonality over time.",
        ("analyze_time_series",), ("metric", "time"),
    ),
    "prediction": AnalysisCapability(
        "prediction", "common", "Build a holdout-evaluated numeric prediction baseline.",
        ("fit_regression_model",), ("target", "features"),
    ),
    "classification": AnalysisCapability(
        "classification", "common", "Build a holdout-evaluated classification baseline.",
        ("fit_classification_model",), ("target", "features"),
    ),
    "churn": AnalysisCapability(
        "churn", "customer", "Estimate customer churn risk.",
        ("fit_classification_model",), ("churn_target", "customer_features"),
    ),
    "anomaly_detection": AnalysisCapability(
        "anomaly_detection", "risk", "Find observations that require review.",
        ("detect_anomalies",), ("features",),
    ),
    "cohort": AnalysisCapability(
        "cohort", "customer", "Compare retained activity by acquisition cohort.",
        ("analyze_cohort_retention",), ("entity_id", "event_time"),
    ),
    "retention": AnalysisCapability(
        "retention", "customer", "Measure repeat activity and retention curves.",
        ("analyze_cohort_retention",), ("entity_id", "event_time"),
    ),
    "funnel": AnalysisCapability(
        "funnel", "product", "Measure ordered step conversion and drop-off.",
        ("analyze_funnel",), ("entity_id", "event", "event_time", "steps"),
    ),
    "journey": AnalysisCapability(
        "journey", "product", "Summarize common ordered user paths.",
        ("analyze_journey",), ("entity_id", "event", "event_time"),
    ),
    "segmentation": AnalysisCapability(
        "segmentation", "customer", "Create behavior-based entity segments.",
        ("segment_entities",), ("entity_id", "features"),
    ),
    "rfm": AnalysisCapability(
        "rfm", "customer", "Score customer recency, frequency, and monetary value.",
        ("analyze_rfm",), ("customer_id", "event_time", "amount"),
    ),
    "contribution": AnalysisCapability(
        "contribution", "sales", "Measure concentration and Pareto contribution.",
        ("analyze_contribution",), ("dimension", "metric"),
    ),
    "mix_shift": AnalysisCapability(
        "mix_shift", "sales", "Separate mix and within-group changes.",
        ("analyze_mix_shift",), ("period", "dimension", "metric"),
    ),
    "profitability": AnalysisCapability(
        "profitability", "finance", "Compare profit or margin contribution.",
        ("analyze_contribution", "compare_groups"), ("dimension", "profit_metric"),
    ),
    "unit_economics": AnalysisCapability(
        "unit_economics", "finance", "Compare unit-level revenue, cost, profit, and margin.",
        ("analyze_contribution", "compare_groups"), ("unit", "revenue_or_profit_metric"),
    ),
    "demand": AnalysisCapability(
        "demand", "operations", "Analyze demand growth, volatility, and seasonality.",
        ("analyze_time_series",), ("demand_metric", "time"),
    ),
    "inventory": AnalysisCapability(
        "inventory", "operations", "Detect inventory trends and abnormal stock observations.",
        ("analyze_time_series", "detect_anomalies"), ("inventory_metric", "time", "features"),
    ),
    "attribution": AnalysisCapability(
        "attribution", "marketing", "Describe channel contribution without causal attribution claims.",
        ("analyze_contribution",), ("channel", "outcome_metric"),
    ),
    "root_cause": AnalysisCapability(
        "root_cause", "operations", "Rank variables and groups associated with a KPI change.",
        ("measure_correlation", "analyze_contribution"), ("metric", "drivers"),
    ),
    "survival": AnalysisCapability(
        "survival", "customer", "Estimate time-to-event survival curves.",
        ("analyze_survival",), ("duration", "event_observed"),
    ),
    "text": AnalysisCapability(
        "text", "voc", "Summarize text themes and frequent terms.",
        ("analyze_text",), ("text",),
    ),
    "scenario": AnalysisCapability(
        "scenario", "strategy", "Evaluate explicit what-if metric changes.",
        ("simulate_scenario",), ("metric", "change_assumptions"),
    ),
    "causal": AnalysisCapability(
        "causal", "experiment", "Produce association evidence and require causal review.",
        ("measure_correlation",), ("outcome", "treatment", "confounders"),
    ),
    "mmm": AnalysisCapability(
        "mmm", "marketing", "Estimate Bayesian media contribution, carryover, and saturation.",
        ("run_bayesian_mmm",), ("date", "outcome", "channel_spend", "controls"),
    ),
    "clv": AnalysisCapability(
        "clv", "customer", "Estimate probabilistic purchases, spend, and customer lifetime value.",
        ("estimate_probabilistic_clv",), ("customer_id", "transaction_time", "amount"),
    ),
    "geospatial": AnalysisCapability(
        "geospatial", "location", "Measure spatial autocorrelation and hotspot candidates.",
        ("analyze_geospatial_hotspots",), ("latitude", "longitude", "metric"),
    ),
    "optimization": AnalysisCapability(
        "optimization", "strategy", "Optimize constrained business allocation decisions.",
        ("optimize_business_allocation",), ("item", "value", "cost", "budget"),
    ),
}


def tool_catalog_text() -> str:
    lines = ["Enterprise analysis capabilities:"]
    for capability in CAPABILITIES.values():
        lines.append(
            f"- {capability.question_type} [{capability.sector}]: {capability.purpose} "
            f"tools={list(capability.tools)} required_roles={list(capability.required_roles)}"
        )
    return "\n".join(lines)
