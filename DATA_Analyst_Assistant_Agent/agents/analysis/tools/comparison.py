from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
import warnings

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.statistics import add_benjamini_hochberg_adjustment

@tool
def compare_groups(records: list[dict[str, Any]], metric: str, dimension: str) -> dict[str, Any]:
    """Compare a numeric metric across groups using count, sum, mean, and median."""

    df = frame_from_records(records)
    if metric not in df.columns or dimension not in df.columns:
        raise ValueError("Metric and dimension must both exist in the data.")
    work = df[[dimension, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        raise ValueError("No valid grouped observations are available.")
    grouped = work.groupby(dimension, dropna=False)[metric].agg(["count", "sum", "mean", "median"])
    grouped = grouped.sort_values("sum", ascending=False)
    rows = [{dimension: str(index), **{key: float(value) for key, value in row.items()}} for index, row in grouped.iterrows()]
    return {"metric": metric, "dimension": dimension, "groups": rows, "group_count": len(rows)}

@tool
def test_group_difference(
    records: list[dict[str, Any]], metric: str, dimension: str, alpha: float = 0.05
) -> dict[str, Any]:
    """Test whether a numeric metric differs across two or more independent groups."""

    from scipy.stats import levene
    from statsmodels.stats.oneway import anova_oneway
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    df = frame_from_records(records)
    if metric not in df.columns or dimension not in df.columns:
        raise ValueError("Metric and dimension must both exist in the data.")
    work = df[[dimension, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna()
    groups = [group[metric].to_numpy(dtype=float) for _, group in work.groupby(dimension) if len(group) >= 2]
    labels = [str(name) for name, group in work.groupby(dimension) if len(group) >= 2]
    if len(groups) < 2:
        raise ValueError("At least two groups with two observations each are required.")
    if len(groups) == 2:
        comparison = CompareMeans(DescrStatsW(groups[0]), DescrStatsW(groups[1]))
        statistic, p_value, degrees_of_freedom = comparison.ttest_ind(usevar="unequal")
        confidence_interval = comparison.tconfint_diff(alpha=alpha, usevar="unequal")
        method = "statsmodels_welch_t_test"
        statistic = float(statistic)
        degrees_of_freedom = float(degrees_of_freedom)
        pooled_variance = (
            ((len(groups[0]) - 1) * np.var(groups[0], ddof=1) + (len(groups[1]) - 1) * np.var(groups[1], ddof=1))
            / (len(groups[0]) + len(groups[1]) - 2)
        )
        effect_size = float((np.mean(groups[0]) - np.mean(groups[1])) / np.sqrt(pooled_variance)) if pooled_variance > 0 else 0.0
        effect_name = "cohen_d"
    else:
        tested = anova_oneway(groups, use_var="unequal", welch_correction=True)
        method = "statsmodels_welch_anova"
        statistic = float(tested.statistic)
        p_value = float(tested.pvalue)
        degrees_of_freedom = [float(value) for value in tested.df]
        confidence_interval = None
        all_values = np.concatenate(groups)
        grand_mean = float(np.mean(all_values))
        between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
        total = float(np.sum((all_values - grand_mean) ** 2))
        effect_size = float(between / total) if total > 0 else 0.0
        effect_name = "eta_squared"
    p_value = float(p_value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        variance_p_value = float(levene(*groups, center="median").pvalue)
    return {
        "method": method,
        "metric": metric,
        "dimension": dimension,
        "groups": labels,
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "mean_difference_ci_95": [float(value) for value in confidence_interval] if confidence_interval else None,
        "alpha": alpha,
        "significant": p_value < alpha,
        "effect_size_name": effect_name,
        "effect_size": effect_size,
        "levene_p_value": variance_p_value if np.isfinite(variance_p_value) else None,
        "assumptions": ["independent observations", "approximately normal residuals", "no severe outliers"],
    }
