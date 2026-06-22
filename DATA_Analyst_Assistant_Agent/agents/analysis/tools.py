from __future__ import annotations

import warnings
import re
import os
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


@tool
def describe_metric(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Calculate count, missing values, mean, median, min, max, and standard deviation for a numeric metric."""

    df = _frame(records)
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        raise ValueError(f"Metric has no numeric observations: {metric}")
    return {
        "metric": metric,
        "count": int(valid.count()),
        "missing": int(values.isna().sum()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "q1": float(valid.quantile(0.25)),
        "q3": float(valid.quantile(0.75)),
        "iqr": float(valid.quantile(0.75) - valid.quantile(0.25)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
    }


@tool
def compare_groups(records: list[dict[str, Any]], metric: str, dimension: str) -> dict[str, Any]:
    """Compare a numeric metric across groups using count, sum, mean, and median."""

    df = _frame(records)
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
def measure_correlation(records: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    """Calculate Pearson correlations for numeric columns without claiming causality."""

    df = _frame(records)
    selected = [column for column in columns if column in df.columns]
    numeric = df[selected].apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    if len(numeric.columns) < 2:
        raise ValueError("At least two numeric columns are required for correlation analysis.")
    pairs: list[dict[str, Any]] = []
    import statsmodels.api as sm

    for left_index, left in enumerate(numeric.columns):
        for right in numeric.columns[left_index + 1 :]:
            pair = numeric[[left, right]].dropna()
            if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            correlation = float(pair[left].corr(pair[right]))
            model = sm.OLS(pair[right].astype(float), sm.add_constant(pair[left].astype(float))).fit()
            confidence_interval = model.conf_int(alpha=0.05).iloc[1]
            pairs.append({
                "left": left,
                "right": right,
                "pearson_r": correlation,
                "p_value": float(model.pvalues.iloc[1]),
                "slope": float(model.params.iloc[1]),
                "slope_ci_95": [float(confidence_interval.iloc[0]), float(confidence_interval.iloc[1])],
                "n": len(pair),
            })
    pairs.sort(key=lambda item: abs(item["pearson_r"]), reverse=True)
    _add_benjamini_hochberg_adjustment(pairs)
    return {"method": "pearson", "pairs": pairs, "observation_count": len(numeric)}


@tool
def test_group_difference(
    records: list[dict[str, Any]], metric: str, dimension: str, alpha: float = 0.05
) -> dict[str, Any]:
    """Test whether a numeric metric differs across two or more independent groups."""

    from scipy.stats import levene
    from statsmodels.stats.oneway import anova_oneway
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    df = _frame(records)
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


@tool
def analyze_trend(records: list[dict[str, Any]], metric: str, time_column: str) -> dict[str, Any]:
    """Estimate a linear direction over ordered time observations for one numeric metric."""

    df = _frame(records)
    if metric not in df.columns or time_column not in df.columns:
        raise ValueError("Metric and time column must both exist in the data.")
    work = df[[time_column, metric]].copy()
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna().sort_values(time_column)
    if len(work) < 2:
        raise ValueError("At least two valid time observations are required.")
    x = np.arange(len(work), dtype=float)
    y = work[metric].to_numpy(dtype=float)
    import statsmodels.api as sm

    regression = sm.OLS(y, sm.add_constant(x)).fit()
    slope = float(regression.params[1])
    slope_interval = regression.conf_int(alpha=0.05)[1]
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    return {
        "metric": metric,
        "time_column": time_column,
        "observation_count": len(work),
        "start": float(y[0]),
        "end": float(y[-1]),
        "slope_per_observation": slope,
        "slope_p_value": float(regression.pvalues[1]),
        "r_squared": float(regression.rsquared),
        "slope_stderr": float(regression.bse[1]),
        "slope_ci_95": [float(slope_interval[0]), float(slope_interval[1])],
        "direction": direction,
    }


def _build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


@tool
def fit_regression_model(records: list[dict[str, Any]], target: str, features: list[str]) -> dict[str, Any]:
    """Fit a deterministic linear regression baseline and evaluate it on a holdout set."""

    df = _frame(records)
    selected = [column for column in features if column in df.columns and column != target]
    if target not in df.columns or not selected:
        raise ValueError("A numeric target and at least one valid feature are required.")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    x = df.loc[valid, selected]
    y = y.loc[valid]
    if len(y) < 20 or y.nunique() < 2:
        raise ValueError("Regression requires at least twenty rows and a non-constant target.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = Pipeline([("preprocess", _build_preprocessor(x, selected)), ("model", LinearRegression())])
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return {
        "method": "linear_regression",
        "target": target,
        "features": selected,
        "observation_count": len(y),
        "train_count": len(y_train),
        "test_count": len(y_test),
        "train_r2": float(r2_score(y_train, train_prediction)),
        "test_r2": float(r2_score(y_test, test_prediction)),
        "test_mae": float(mean_absolute_error(y_test, test_prediction)),
        "test_rmse": float(mean_squared_error(y_test, test_prediction) ** 0.5),
        "top_coefficients": _top_coefficients(model),
        "evaluation_scope": "fixed_holdout",
    }


@tool
def fit_classification_model(records: list[dict[str, Any]], target: str, features: list[str]) -> dict[str, Any]:
    """Fit a deterministic logistic baseline and evaluate it on a stratified holdout set."""

    df = _frame(records)
    selected = [column for column in features if column in df.columns and column != target]
    if target not in df.columns or not selected:
        raise ValueError("A target and at least one valid feature are required.")
    valid = df[target].notna()
    x = df.loc[valid, selected]
    y = df.loc[valid, target].astype(str)
    if len(y) < 20 or y.nunique() < 2 or int(y.value_counts().min()) < 4:
        raise ValueError("Classification requires twenty rows and at least four observations per class.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocess", _build_preprocessor(x, selected)),
        ("model", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    result: dict[str, Any] = {
        "method": "logistic_regression",
        "target": target,
        "features": selected,
        "classes": sorted(y.unique().tolist()),
        "observation_count": len(y),
        "train_count": len(y_train),
        "test_count": len(y_test),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "test_f1_weighted": float(f1_score(y_test, prediction, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=model.named_steps["model"].classes_).tolist(),
        "top_coefficients": _top_coefficients(model),
        "evaluation_scope": "stratified_fixed_holdout",
    }
    if y.nunique() == 2:
        probability = model.predict_proba(x_test)[:, 1]
        positive = model.named_steps["model"].classes_[1]
        result["test_roc_auc"] = float(roc_auc_score((y_test == positive).astype(int), probability))
    return result


@tool
def detect_anomalies(
    records: list[dict[str, Any]], features: list[str], contamination: float = 0.05
) -> dict[str, Any]:
    """Detect multivariate numeric outliers with a deterministic Isolation Forest baseline."""

    df = _frame(records)
    selected = [column for column in features if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]
    if not selected:
        raise ValueError("Anomaly detection requires at least one numeric feature.")
    x = df[selected].apply(pd.to_numeric, errors="coerce")
    if len(x) < 10:
        raise ValueError("Anomaly detection requires at least ten observations.")
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(x)
    model = IsolationForest(contamination=contamination, random_state=0)
    labels = model.fit_predict(values)
    scores = model.decision_function(values)
    anomaly_positions = np.flatnonzero(labels == -1).tolist()
    return {
        "method": "isolation_forest",
        "features": selected,
        "observation_count": len(x),
        "contamination": contamination,
        "anomaly_count": len(anomaly_positions),
        "anomaly_rate": float(len(anomaly_positions) / len(x)),
        "anomaly_row_positions": anomaly_positions,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "top_anomaly_candidates": [
            {"row_position": int(position), "score": float(scores[position])}
            for position in sorted(anomaly_positions, key=lambda item: scores[item])[:10]
        ],
    }


@tool
def analyze_time_series(
    records: list[dict[str, Any]],
    metric: str,
    time_column: str,
    frequency: str = "M",
    aggregation: str = "sum",
    seasonal_period: int | None = None,
) -> dict[str, Any]:
    """Analyze an aggregated business time series for growth, trend, volatility, and seasonality."""

    import statsmodels.api as sm

    df = _frame(records)
    if metric not in df.columns or time_column not in df.columns:
        raise ValueError("Metric and time column must both exist in the data.")
    work = df[[time_column, metric]].copy()
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna().sort_values(time_column)
    if len(work) < 4:
        raise ValueError("Time-series analysis requires at least four valid observations.")
    rule = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}.get(frequency.upper(), frequency)
    indexed = work.set_index(time_column)[metric]
    if aggregation == "mean":
        series = indexed.resample(rule).mean()
    elif aggregation == "median":
        series = indexed.resample(rule).median()
    else:
        series = indexed.resample(rule).sum(min_count=1)
    series = series.dropna()
    if len(series) < 4:
        raise ValueError("Aggregation produced fewer than four time periods.")
    x = np.arange(len(series), dtype=float)
    regression = sm.OLS(series.to_numpy(dtype=float), sm.add_constant(x)).fit()
    slope_interval = regression.conf_int(alpha=0.05)[1]
    growth = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    window = min(max(2, seasonal_period or 3), len(series))
    rolling = series.rolling(window=window, min_periods=1).mean()
    seasonality_strength = None
    if seasonal_period and len(series) >= seasonal_period * 2:
        phase_means = series.groupby(np.arange(len(series)) % seasonal_period).mean()
        residual = series.to_numpy() - np.take(phase_means.to_numpy(), np.arange(len(series)) % seasonal_period)
        total_variance = float(np.var(series.to_numpy()))
        seasonality_strength = 1.0 - float(np.var(residual)) / total_variance if total_variance > 0 else 0.0
    return {
        "method": "aggregated_time_series",
        "metric": metric,
        "time_column": time_column,
        "frequency": frequency.upper(),
        "aggregation": aggregation,
        "period_count": len(series),
        "start_value": float(series.iloc[0]),
        "end_value": float(series.iloc[-1]),
        "total_growth_rate": float(series.iloc[-1] / series.iloc[0] - 1) if series.iloc[0] != 0 else None,
        "average_period_growth_rate": float(growth.mean()) if not growth.empty else None,
        "growth_volatility": float(growth.std(ddof=1)) if len(growth) > 1 else 0.0,
        "trend_slope": float(regression.params[1]),
        "trend_p_value": float(regression.pvalues[1]),
        "trend_r_squared": float(regression.rsquared),
        "trend_slope_ci_95": [float(slope_interval[0]), float(slope_interval[1])],
        "seasonal_period": seasonal_period,
        "seasonality_strength": seasonality_strength,
        "series": [
            {"period": str(index), "value": float(value), "rolling_mean": float(rolling.loc[index])}
            for index, value in series.tail(120).items()
        ],
    }


@tool
def analyze_cohort_retention(
    records: list[dict[str, Any]],
    entity_id_column: str,
    event_time_column: str,
    cohort_time_column: str | None = None,
    period_unit: str = "month",
) -> dict[str, Any]:
    """Build acquisition cohorts and calculate period-by-period entity retention."""

    df = _frame(records)
    required = [entity_id_column, event_time_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Entity ID and event time columns are required for cohort analysis.")
    work = df[required + ([cohort_time_column] if cohort_time_column else [])].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work = work.dropna(subset=[entity_id_column, event_time_column])
    if cohort_time_column:
        work[cohort_time_column] = pd.to_datetime(work[cohort_time_column], errors="coerce")
        work["_cohort_time"] = work[cohort_time_column]
    else:
        work["_cohort_time"] = work.groupby(entity_id_column)[event_time_column].transform("min")
    if period_unit == "week":
        work["_cohort"] = work["_cohort_time"].dt.to_period("W").astype(str)
        work["_period_index"] = ((work[event_time_column] - work["_cohort_time"]).dt.days // 7).astype(int)
    else:
        work["_cohort"] = work["_cohort_time"].dt.to_period("M").astype(str)
        work["_period_index"] = (
            (work[event_time_column].dt.year - work["_cohort_time"].dt.year) * 12
            + work[event_time_column].dt.month - work["_cohort_time"].dt.month
        ).astype(int)
    work = work[work["_period_index"] >= 0]
    active = work.groupby(["_cohort", "_period_index"])[entity_id_column].nunique()
    cohort_sizes = active.groupby(level=0).first()
    matrix: list[dict[str, Any]] = []
    for (cohort, period), count in active.items():
        size = int(cohort_sizes.loc[cohort])
        matrix.append({
            "cohort": str(cohort),
            "period_index": int(period),
            "active_entities": int(count),
            "cohort_size": size,
            "retention_rate": float(count / size) if size else 0.0,
        })
    return {
        "method": "cohort_retention",
        "entity_id_column": entity_id_column,
        "event_time_column": event_time_column,
        "period_unit": period_unit,
        "cohort_count": int(len(cohort_sizes)),
        "entity_count": int(work[entity_id_column].nunique()),
        "retention_matrix": matrix,
    }


@tool
def analyze_funnel(
    records: list[dict[str, Any]],
    entity_id_column: str,
    event_column: str,
    event_time_column: str,
    steps: list[str],
) -> dict[str, Any]:
    """Measure strict ordered funnel conversion and drop-off by entity."""

    df = _frame(records)
    if not steps or len(steps) < 2:
        raise ValueError("Funnel analysis requires at least two ordered steps.")
    required = [entity_id_column, event_column, event_time_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Entity, event, and event-time columns are required for funnel analysis.")
    work = df[required].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work = work.dropna().sort_values([entity_id_column, event_time_column])
    reached = [0] * len(steps)
    for _, group in work.groupby(entity_id_column):
        next_step = 0
        for event in group[event_column].astype(str):
            if next_step < len(steps) and event == steps[next_step]:
                reached[next_step] += 1
                next_step += 1
    first = reached[0]
    result_steps = []
    for index, (step, count) in enumerate(zip(steps, reached)):
        previous = reached[index - 1] if index > 0 else count
        result_steps.append({
            "step": step,
            "entities": int(count),
            "conversion_from_start": float(count / first) if first else 0.0,
            "conversion_from_previous": float(count / previous) if previous else 0.0,
            "dropoff_from_previous": int(previous - count) if index > 0 else 0,
        })
    return {
        "method": "strict_ordered_funnel",
        "entity_count": int(work[entity_id_column].nunique()),
        "steps": result_steps,
        "overall_conversion_rate": float(reached[-1] / first) if first else 0.0,
    }


@tool
def analyze_journey(
    records: list[dict[str, Any]],
    entity_id_column: str,
    event_column: str,
    event_time_column: str,
    max_steps: int = 10,
    top_n: int = 20,
) -> dict[str, Any]:
    """Summarize common ordered entity paths and event-to-event transitions."""

    df = _frame(records)
    required = [entity_id_column, event_column, event_time_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Entity, event, and event-time columns are required for journey analysis.")
    work = df[required].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work = work.dropna().sort_values([entity_id_column, event_time_column])
    paths: Counter[tuple[str, ...]] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    for _, group in work.groupby(entity_id_column):
        events = group[event_column].astype(str).tolist()[:max_steps]
        if events:
            paths[tuple(events)] += 1
        transitions.update(zip(events, events[1:]))
    return {
        "method": "ordered_journey_paths",
        "entity_count": int(work[entity_id_column].nunique()),
        "top_paths": [
            {"path": list(path), "entities": count}
            for path, count in paths.most_common(top_n)
        ],
        "top_transitions": [
            {"from_event": pair[0], "to_event": pair[1], "count": count}
            for pair, count in transitions.most_common(top_n)
        ],
    }


@tool
def segment_entities(
    records: list[dict[str, Any]], entity_id_column: str, features: list[str], n_clusters: int = 4
) -> dict[str, Any]:
    """Cluster entities using standardized numeric behavior features."""

    df = _frame(records)
    selected = [column for column in features if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]
    if entity_id_column not in df.columns or len(selected) < 2:
        raise ValueError("Segmentation requires an entity ID and at least two numeric features.")
    entity_frame = df.groupby(entity_id_column, as_index=False)[selected].mean()
    if len(entity_frame) <= n_clusters:
        raise ValueError("Entity count must be larger than n_clusters.")
    values = SimpleImputer(strategy="median").fit_transform(entity_frame[selected])
    values = StandardScaler().fit_transform(values)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(values)
    entity_frame["segment"] = labels
    profiles = entity_frame.groupby("segment")[selected].mean().reset_index().to_dict(orient="records")
    sizes = entity_frame["segment"].value_counts().sort_index().to_dict()
    return {
        "method": "kmeans_segmentation",
        "entity_id_column": entity_id_column,
        "features": selected,
        "n_clusters": n_clusters,
        "entity_count": len(entity_frame),
        "silhouette_score": float(silhouette_score(values, labels)),
        "segment_sizes": {str(key): int(value) for key, value in sizes.items()},
        "segment_profiles": profiles,
    }


@tool
def analyze_rfm(
    records: list[dict[str, Any]], customer_id_column: str, event_time_column: str, amount_column: str
) -> dict[str, Any]:
    """Calculate recency, frequency, monetary scores and customer segment counts."""

    df = _frame(records)
    required = [customer_id_column, event_time_column, amount_column]
    if any(column not in df.columns for column in required):
        raise ValueError("Customer ID, event time, and amount columns are required for RFM analysis.")
    work = df[required].copy()
    work[event_time_column] = pd.to_datetime(work[event_time_column], errors="coerce")
    work[amount_column] = pd.to_numeric(work[amount_column], errors="coerce")
    work = work.dropna()
    reference = work[event_time_column].max() + pd.Timedelta(days=1)
    rfm = work.groupby(customer_id_column).agg(
        recency=(event_time_column, lambda value: int((reference - value.max()).days)),
        frequency=(event_time_column, "count"),
        monetary=(amount_column, "sum"),
    )
    if len(rfm) < 5:
        raise ValueError("RFM analysis requires at least five customers.")
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["score"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)
    rfm["segment"] = pd.cut(
        rfm["score"], bins=[0, 6, 9, 12, 15], labels=["at_risk", "regular", "loyal", "champion"]
    )
    return {
        "method": "rfm_quintile_scoring",
        "customer_count": len(rfm),
        "reference_date": str(reference.date()),
        "segment_counts": {str(key): int(value) for key, value in rfm["segment"].value_counts().items()},
        "summary": rfm[["recency", "frequency", "monetary", "score"]].describe().to_dict(),
    }


@tool
def analyze_contribution(
    records: list[dict[str, Any]], dimension: str, metric: str
) -> dict[str, Any]:
    """Calculate ranked contribution, concentration, and Pareto coverage by dimension."""

    df = _frame(records)
    if dimension not in df.columns or metric not in df.columns:
        raise ValueError("Dimension and metric columns are required for contribution analysis.")
    work = df[[dimension, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    grouped = work.dropna(subset=[metric]).groupby(dimension, dropna=False)[metric].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    if grouped.empty or total == 0:
        raise ValueError("Contribution analysis requires a non-zero aggregated metric.")
    share = grouped / total
    cumulative = share.cumsum()
    pareto_count = int((cumulative < 0.8).sum() + 1)
    rows = [
        {"dimension_value": str(key), "metric": float(value), "share": float(share.loc[key]), "cumulative_share": float(cumulative.loc[key])}
        for key, value in grouped.head(100).items()
    ]
    return {
        "method": "pareto_contribution",
        "dimension": dimension,
        "metric": metric,
        "group_count": len(grouped),
        "top_1_share": float(share.iloc[0]),
        "top_5_share": float(share.head(5).sum()),
        "pareto_80_group_count": pareto_count,
        "pareto_80_group_rate": float(pareto_count / len(grouped)),
        "contributions": rows,
    }


@tool
def analyze_mix_shift(
    records: list[dict[str, Any]], period_column: str, dimension: str, metric: str
) -> dict[str, Any]:
    """Explain the total metric change between the latest two periods by dimension contribution."""

    df = _frame(records)
    required = [period_column, dimension, metric]
    if any(column not in df.columns for column in required):
        raise ValueError("Period, dimension, and metric columns are required for mix-shift analysis.")
    work = df[required].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[period_column, dimension, metric])
    periods = sorted(work[period_column].astype(str).unique())
    if len(periods) < 2:
        raise ValueError("Mix-shift analysis requires at least two periods.")
    previous_period, current_period = periods[-2], periods[-1]
    pivot = (
        work[work[period_column].astype(str).isin([previous_period, current_period])]
        .assign(_period=work[period_column].astype(str))
        .pivot_table(index=dimension, columns="_period", values=metric, aggfunc="sum", fill_value=0.0)
    )
    for period in (previous_period, current_period):
        if period not in pivot.columns:
            pivot[period] = 0.0
    pivot["delta"] = pivot[current_period] - pivot[previous_period]
    total_previous = float(pivot[previous_period].sum())
    total_current = float(pivot[current_period].sum())
    total_delta = total_current - total_previous
    pivot = pivot.sort_values("delta", ascending=False)
    drivers = [
        {
            "dimension_value": str(index),
            "previous": float(row[previous_period]),
            "current": float(row[current_period]),
            "delta": float(row["delta"]),
            "share_of_total_change": float(row["delta"] / total_delta) if total_delta else None,
        }
        for index, row in pivot.head(100).iterrows()
    ]
    return {
        "method": "period_contribution_change",
        "previous_period": previous_period,
        "current_period": current_period,
        "previous_total": total_previous,
        "current_total": total_current,
        "total_delta": total_delta,
        "growth_rate": float(total_delta / total_previous) if total_previous else None,
        "dimension_drivers": drivers,
    }


@tool
def analyze_survival(
    records: list[dict[str, Any]], duration_column: str, event_observed_column: str
) -> dict[str, Any]:
    """Estimate a Kaplan-Meier survival curve from duration and event-observed columns."""

    df = _frame(records)
    if duration_column not in df.columns or event_observed_column not in df.columns:
        raise ValueError("Duration and event-observed columns are required for survival analysis.")
    duration = pd.to_numeric(df[duration_column], errors="coerce")
    observed = df[event_observed_column].astype(str).str.casefold().map(
        {"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0}
    )
    work = pd.DataFrame({"duration": duration, "observed": observed}).dropna()
    work = work[work["duration"] >= 0]
    if len(work) < 10:
        raise ValueError("Survival analysis requires at least ten valid observations.")
    survival = 1.0
    curve = [{"time": 0.0, "survival_probability": 1.0, "at_risk": len(work), "events": 0}]
    for time in sorted(work.loc[work["observed"] == 1, "duration"].unique()):
        at_risk = int((work["duration"] >= time).sum())
        events = int(((work["duration"] == time) & (work["observed"] == 1)).sum())
        if at_risk:
            survival *= 1.0 - events / at_risk
        curve.append({
            "time": float(time),
            "survival_probability": float(survival),
            "at_risk": at_risk,
            "events": events,
        })
    median = next((item["time"] for item in curve if item["survival_probability"] <= 0.5), None)
    return {
        "method": "kaplan_meier",
        "observation_count": len(work),
        "event_count": int(work["observed"].sum()),
        "censored_count": int((work["observed"] == 0).sum()),
        "median_survival_time": median,
        "survival_curve": curve[:200],
    }


@tool
def analyze_text(
    records: list[dict[str, Any]], text_column: str, top_n: int = 30
) -> dict[str, Any]:
    """Summarize text coverage and frequent terms without inventing semantic labels."""

    df = _frame(records)
    if text_column not in df.columns:
        raise ValueError("Text column is required for text analysis.")
    texts = df[text_column].dropna().astype(str)
    if texts.empty:
        raise ValueError("Text analysis requires non-empty text values.")
    tokens = [
        token.casefold()
        for text in texts
        for token in re.findall(r"\b[^\W\d_]{2,}\b", text, flags=re.UNICODE)
    ]
    counts = Counter(tokens)
    lengths = texts.str.len()
    return {
        "method": "frequency_based_text_profile",
        "document_count": len(texts),
        "token_count": len(tokens),
        "average_text_length": float(lengths.mean()),
        "top_terms": [{"term": term, "count": count} for term, count in counts.most_common(top_n)],
        "semantic_interpretation_required": True,
    }


@tool
def simulate_scenario(
    records: list[dict[str, Any]], metric: str, change_percent: float, aggregation: str = "sum"
) -> dict[str, Any]:
    """Apply an explicit percentage change to a metric for transparent what-if analysis."""

    df = _frame(records)
    if metric not in df.columns:
        raise ValueError("Metric column is required for scenario analysis.")
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if values.empty:
        raise ValueError("Scenario analysis requires numeric metric values.")
    baseline = float(values.mean() if aggregation == "mean" else values.sum())
    projected = baseline * (1.0 + change_percent / 100.0)
    return {
        "method": "deterministic_what_if",
        "metric": metric,
        "aggregation": aggregation,
        "change_percent": change_percent,
        "baseline": baseline,
        "projected": projected,
        "absolute_change": projected - baseline,
        "assumption": "All other factors are held constant.",
    }


def _add_benjamini_hochberg_adjustment(pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        return
    ranked = sorted(enumerate(pairs), key=lambda item: item[1]["p_value"])
    total = len(ranked)
    adjusted = [1.0] * total
    running = 1.0
    for reverse_rank in range(total - 1, -1, -1):
        original_index, pair = ranked[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(pair["p_value"]) * total / rank)
        adjusted[original_index] = min(1.0, running)
    for index, pair in enumerate(pairs):
        pair["p_value_adjusted_bh"] = adjusted[index]


def _top_coefficients(model: Pipeline, limit: int = 10) -> list[dict[str, Any]]:
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocess.get_feature_names_out().tolist()
    coefficients = np.asarray(estimator.coef_)
    if coefficients.ndim > 1:
        coefficients = np.mean(np.abs(coefficients), axis=0)
    else:
        coefficients = np.abs(coefficients)
    ranked = sorted(zip(names, coefficients), key=lambda item: float(item[1]), reverse=True)[:limit]
    return [{"feature": name, "absolute_coefficient": float(value)} for name, value in ranked]


from DATA_Analyst_Assistant_Agent.agents.analysis.advanced_tools import (
    analyze_geospatial_hotspots,
    estimate_probabilistic_clv,
    optimize_business_allocation,
    run_bayesian_mmm,
)


ANALYSIS_TOOLS = {
    item.name: item
    for item in (
        describe_metric,
        compare_groups,
        measure_correlation,
        test_group_difference,
        analyze_trend,
        fit_regression_model,
        fit_classification_model,
        detect_anomalies,
        analyze_time_series,
        analyze_cohort_retention,
        analyze_funnel,
        analyze_journey,
        segment_entities,
        analyze_rfm,
        analyze_contribution,
        analyze_mix_shift,
        analyze_survival,
        analyze_text,
        simulate_scenario,
        run_bayesian_mmm,
        estimate_probabilistic_clv,
        analyze_geospatial_hotspots,
        optimize_business_allocation,
    )
}
