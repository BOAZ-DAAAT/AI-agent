from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records

@tool
def describe_metric(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Calculate count, missing values, mean, median, min, max, and standard deviation for a numeric metric."""

    df = frame_from_records(records)
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
