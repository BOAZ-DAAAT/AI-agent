from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records

@tool
def analyze_survival(
    records: list[dict[str, Any]], duration_column: str, event_observed_column: str
) -> dict[str, Any]:
    """Estimate a Kaplan-Meier survival curve from duration and event-observed columns."""

    df = frame_from_records(records)
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
