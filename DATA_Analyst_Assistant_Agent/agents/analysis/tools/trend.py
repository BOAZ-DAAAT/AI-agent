from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records

@tool
def analyze_trend(records: list[dict[str, Any]], metric: str, time_column: str) -> dict[str, Any]:
    """Estimate a linear direction over ordered time observations for one numeric metric."""

    df = frame_from_records(records)
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

    df = frame_from_records(records)
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
