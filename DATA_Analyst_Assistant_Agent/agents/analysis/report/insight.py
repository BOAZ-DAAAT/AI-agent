from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisEvidence,
    AnalysisExecutionPlan,
    AnalysisKind,
    HypothesisSummary,
)


def evidence_from_payload(
    tool_name: str, payload: dict[str, Any], plan: AnalysisExecutionPlan
) -> AnalysisEvidence:
    caveats = ["Computed from the current run's SQL result artifact."]
    if tool_name == "describe_metric":
        finding = (
            f"{payload['metric']} ranges from {payload['min']:g} to {payload['max']:g} "
            f"with mean {payload['mean']:.2f} across {payload['count']} observations."
        )
        method = "Descriptive statistics"
    elif tool_name == "compare_groups":
        top = payload["groups"][0]
        finding = f"Top {payload['dimension']} by {payload['metric']} is {top[payload['dimension']]} ({top['sum']:g})."
        method = "Grouped aggregation"
    elif tool_name == "test_group_difference":
        significance = "statistically significant" if payload["significant"] else "not statistically significant"
        finding = (
            f"The {payload['method']} result for {payload['metric']} across {payload['dimension']} was "
            f"{significance} at alpha={payload['alpha']:.2f} (p={payload['p_value']:.4g})."
        )
        method = payload["method"]
        caveats.extend(payload["assumptions"])
    elif tool_name == "measure_correlation":
        top = payload["pairs"][0] if payload["pairs"] else None
        finding = (
            f"Strongest observed Pearson relationship is {top['left']} versus {top['right']} "
            f"(r={top['pearson_r']:.3f}, adjusted p={top['p_value_adjusted_bh']:.4g}, n={top['n']})."
            if top
            else "No estimable Pearson correlation pair was found."
        )
        method = "Pearson correlation"
        caveats.append("Correlation is not evidence of causation.")
    elif tool_name == "analyze_trend":
        finding = (
            f"{payload['metric']} has an observed {payload['direction']} direction over "
            f"{payload['observation_count']} ordered observations "
            f"(slope p={payload['slope_p_value']:.4g}, R2={payload['r_squared']:.3f})."
        )
        method = "Linear trend estimate"
    elif tool_name == "fit_regression_model":
        finding = (
            f"Linear regression baseline for {payload['target']} achieved "
            f"holdout R2={payload['test_r2']:.3f}, MAE={payload['test_mae']:.3f}, "
            f"and RMSE={payload['test_rmse']:.3f}."
        )
        method = "Linear regression baseline"
        caveats.append("A single holdout split is a baseline and should be followed by repeated validation.")
    elif tool_name == "fit_classification_model":
        auc = f" and ROC-AUC={payload['test_roc_auc']:.3f}" if "test_roc_auc" in payload else ""
        finding = (
            f"Logistic classification baseline for {payload['target']} achieved "
            f"holdout accuracy={payload['test_accuracy']:.3f}, balanced accuracy="
            f"{payload['test_balanced_accuracy']:.3f}{auc}."
        )
        method = "Logistic classification baseline"
        caveats.append("A single holdout split is a baseline and should be followed by repeated validation.")
    elif tool_name == "detect_anomalies":
        finding = (
            f"Isolation Forest flagged {payload['anomaly_count']} of {payload['observation_count']} rows "
            f"as anomalies ({payload['anomaly_rate']:.1%})."
        )
        method = "Isolation Forest anomaly detection"
        caveats.append("Anomaly labels are candidates for review, not proof of data errors or fraud.")
    elif tool_name == "analyze_time_series":
        growth = payload.get("total_growth_rate")
        growth_text = f"{growth:.1%}" if growth is not None else "not estimable"
        finding = (
            f"{payload['metric']} changed by {growth_text} across {payload['period_count']} "
            f"{payload['frequency']} periods (trend p={payload['trend_p_value']:.4g})."
        )
        method = "Aggregated business time-series analysis"
        caveats.append("Trend and seasonality depend on aggregation frequency and available history.")
    elif tool_name == "analyze_cohort_retention":
        finding = (
            f"Cohort analysis covered {payload['entity_count']} entities across "
            f"{payload['cohort_count']} cohorts."
        )
        method = "Cohort retention matrix"
        caveats.append("Retention requires a stable entity identifier and consistent event logging.")
    elif tool_name == "analyze_funnel":
        finding = (
            f"Strict ordered funnel conversion was {payload['overall_conversion_rate']:.1%} "
            f"across {len(payload['steps'])} steps."
        )
        method = "Strict ordered funnel"
        caveats.append("Step definitions and event ordering materially affect funnel conversion.")
    elif tool_name == "analyze_journey":
        top_path = payload["top_paths"][0] if payload["top_paths"] else {"path": [], "entities": 0}
        finding = (
            f"Journey analysis covered {payload['entity_count']} entities; the most common path "
            f"was {' -> '.join(top_path['path'])} ({top_path['entities']} entities)."
        )
        method = "Ordered customer journey paths"
        caveats.append("Event taxonomy and identity stitching affect observed paths.")
    elif tool_name == "segment_entities":
        finding = (
            f"K-means produced {payload['n_clusters']} segments for {payload['entity_count']} entities "
            f"with silhouette={payload['silhouette_score']:.3f}."
        )
        method = "Behavioral segmentation"
        caveats.append("Segments are descriptive and require business naming and stability checks.")
    elif tool_name == "analyze_rfm":
        finding = f"RFM scoring segmented {payload['customer_count']} customers by historical value signals."
        method = "RFM customer value scoring"
        caveats.append("RFM is historical and is not a probabilistic lifetime-value forecast.")
    elif tool_name == "analyze_contribution":
        finding = (
            f"The top {payload['pareto_80_group_count']} of {payload['group_count']} groups account "
            f"for approximately 80% of {payload['metric']}."
        )
        method = "Pareto contribution analysis"
        caveats.append("Contribution does not identify causal drivers of the metric.")
    elif tool_name == "analyze_mix_shift":
        growth = payload.get("growth_rate")
        growth_text = f"{growth:.1%}" if growth is not None else "not estimable"
        finding = (
            f"{payload['previous_period']} to {payload['current_period']} total change was "
            f"{payload['total_delta']:.3f} ({growth_text})."
        )
        method = "Period contribution change"
        caveats.append("Driver contribution is descriptive and does not establish causality.")
    elif tool_name == "analyze_survival":
        median = payload.get("median_survival_time")
        finding = (
            f"Kaplan-Meier analysis observed {payload['event_count']} events among "
            f"{payload['observation_count']} records; median survival={median}."
        )
        method = "Kaplan-Meier survival analysis"
        caveats.append("Censoring must be correctly encoded for survival estimates to be valid.")
    elif tool_name == "analyze_text":
        finding = (
            f"Text profiling summarized {payload['document_count']} documents and "
            f"{payload['token_count']} tokens."
        )
        method = "Frequency-based text profile"
        caveats.append("Frequent terms are not semantic topics; LLM or human interpretation is required.")
    elif tool_name == "simulate_scenario":
        finding = (
            f"A {payload['change_percent']:.1f}% what-if change moves {payload['metric']} from "
            f"{payload['baseline']:.3f} to {payload['projected']:.3f}."
        )
        method = "Deterministic what-if scenario"
        caveats.append(payload["assumption"])
    elif tool_name == "run_bayesian_mmm":
        top_channel = max(
            payload["channel_contribution_shares"],
            key=payload["channel_contribution_shares"].get,
            default="unknown",
        )
        finding = (
            f"Bayesian MMM fitted {payload['period_count']} periods; the largest modeled media "
            f"contribution share was {top_channel}."
        )
        method = "PyMC-Marketing Bayesian MMM"
        caveats.append("Media contribution depends on priors, controls, adstock, saturation, and spend variation.")
    elif tool_name == "estimate_probabilistic_clv":
        finding = (
            f"Probabilistic CLV modeled {payload['customer_count']} customers with mean expected "
            f"CLV={payload['mean_expected_clv']:.3f}."
        )
        method = "BG/NBD and Gamma-Gamma probabilistic CLV"
        caveats.append("CLV depends on repeat-purchase and spend-model assumptions and calibration quality.")
    elif tool_name == "analyze_geospatial_hotspots":
        finding = (
            f"Global Moran's I was {payload['global_moran_i']:.3f} "
            f"(simulation p={payload['global_p_value_simulated']:.4g}) across {payload['point_count']} points."
        )
        method = "Moran spatial autocorrelation"
        caveats.append("Hotspot results depend on CRS, neighbor weights, spatial scale, and privacy thresholds.")
    elif tool_name == "optimize_business_allocation":
        finding = (
            f"OR-Tools returned {payload['status']} allocation with objective="
            f"{payload['objective_value']:.3f} and budget used={payload['budget_used']:.3f}."
        )
        method = "Constrained OR-Tools allocation"
        caveats.append("Optimization is a recommendation and requires approval before operational execution.")
    else:
        raise ValueError(f"No insight formatter is registered for tool: {tool_name}")
    return AnalysisEvidence(
        tool_name=tool_name,
        method=method,
        inputs={
            "metric": plan.metric,
            "dimension": plan.dimension,
            "time_column": plan.time_column,
            "feature_columns": plan.feature_columns,
        },
        statistics=payload,
        finding=finding,
        caveats=caveats,
    )


def build_hypotheses(kind: AnalysisKind, evidence: list[AnalysisEvidence]) -> list[HypothesisSummary]:
    if kind == AnalysisKind.correlation:
        correlation = next((item for item in evidence if item.tool_name == "measure_correlation"), None)
        pairs = correlation.statistics.get("pairs", []) if correlation else []
        significant = any(float(pair.get("p_value_adjusted_bh", 1.0)) < 0.05 for pair in pairs)
        return [HypothesisSummary(
            null_hypothesis="The selected numeric variables have no linear association.",
            alternative_hypothesis="At least one selected pair has a linear association.",
            decision="supported" if significant else "not_supported" if pairs else "inconclusive",
            rationale="Decision uses Benjamini-Hochberg adjusted Pearson p-values at alpha=0.05.",
        )]
    if kind == AnalysisKind.group_comparison:
        test = next((item for item in evidence if item.tool_name == "test_group_difference"), None)
        if not test:
            return [HypothesisSummary(
                null_hypothesis="The group metric summaries do not differ.",
                alternative_hypothesis="At least one group metric summary differs.",
                decision="inconclusive",
                rationale="The data did not satisfy the minimum group-test requirements.",
            )]
        return [HypothesisSummary(
            null_hypothesis="The group metric means do not differ.",
            alternative_hypothesis="At least one group metric mean differs.",
            decision="supported" if test.statistics["significant"] else "not_supported",
            rationale=f"Decision uses {test.statistics['method']} at alpha={test.statistics['alpha']}.",
        )]
    return []
