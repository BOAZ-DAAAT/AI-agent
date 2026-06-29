from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.statistics import add_benjamini_hochberg_adjustment

@tool
def measure_correlation(records: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    """Calculate Pearson correlations for numeric columns without claiming causality."""

    df = frame_from_records(records)
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
    add_benjamini_hochberg_adjustment(pairs)
    return {"method": "pearson", "pairs": pairs, "observation_count": len(numeric)}
