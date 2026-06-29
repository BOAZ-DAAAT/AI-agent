from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

@tool
def detect_anomalies(
    records: list[dict[str, Any]], features: list[str], contamination: float = 0.05
) -> dict[str, Any]:
    """Detect multivariate numeric outliers with a deterministic Isolation Forest baseline."""

    df = frame_from_records(records)
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
