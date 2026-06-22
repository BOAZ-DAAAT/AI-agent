from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def select_chart_type(state: OrchestrationState, dataframe: pd.DataFrame | None = None) -> str:
    route_kind = state.route_kind or (state.plan.route_kind if state.plan else "")
    if route_kind == "trend":
        return "line"
    query = state.user_query.casefold()
    if any(token in query for token in ("monthly", "trend", "\uc6d4\ubcc4", "\ucd94\uc774")):
        return "line"
    if any(token in query for token in ("compare", "category", "top", "\ube44\uad50", "\uce74\ud14c\uace0\ub9ac")):
        return "bar"
    if any(token in query for token in ("distribution", "histogram", "\ubd84\ud3ec")):
        return "histogram"
    if dataframe is not None and not dataframe.empty:
        numeric_cols = list(dataframe.select_dtypes(include="number").columns)
        non_numeric_cols = [column for column in dataframe.columns if column not in numeric_cols]
        if numeric_cols and non_numeric_cols:
            return "bar"
        if numeric_cols:
            return "histogram" if len(dataframe) > 1 else "table"
    return "table"


def build_chart_config(
    state: OrchestrationState,
    *,
    dataframe: pd.DataFrame | None = None,
    analysis_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataframe = dataframe if dataframe is not None else pd.DataFrame()
    chart_type = select_chart_type(state, dataframe)
    plan = state.plan
    x_field, y_field = _select_fields(dataframe, chart_type, plan.dimension if plan else None, plan.metric if plan else None)
    x_type = "temporal" if chart_type == "line" else "nominal"
    values = dataframe.head(100).where(pd.notna(dataframe), None).to_dict(orient="records") if not dataframe.empty else []
    spec = _vega_lite_spec(chart_type, x_field, y_field, x_type, values)
    return {
        "chart_type": chart_type,
        "title": _title_for(chart_type, x_field, y_field),
        "vega_lite_spec": spec,
        "encoding": {
            "x": {"field": x_field, "type": x_type, "label": _label_for(x_field), "unit": "unknown"},
            "y": {"field": y_field, "type": "quantitative", "label": _label_for(y_field), "unit": "unknown"},
        },
        "data_reference": state.artifact_ids.get("sql_agent", []),
        "analysis_reference": state.artifact_ids.get("analysis_agent", []),
        "analysis_findings": (analysis_result or {}).get("key_findings", []),
        "axis": {
            "x_label": _label_for(x_field),
            "y_label": _label_for(y_field),
            "unit_note": "Units are inferred from artifact columns until source metadata is available.",
        },
    }


def _select_fields(
    df: pd.DataFrame,
    chart_type: str,
    dimension_hint: str | None,
    metric_hint: str | None,
) -> tuple[str, str]:
    columns = list(df.columns)
    numeric_cols = list(df.select_dtypes(include="number").columns) if not df.empty else []
    non_numeric_cols = [column for column in columns if column not in numeric_cols]
    metric = metric_hint if metric_hint in columns else (numeric_cols[0] if numeric_cols else (columns[-1] if columns else "metric"))
    dimension = dimension_hint if dimension_hint in columns else (non_numeric_cols[0] if non_numeric_cols else (columns[0] if columns else "dimension"))
    if chart_type == "histogram":
        return metric, "count"
    if chart_type == "table":
        return dimension, metric
    return dimension, metric


def _vega_lite_spec(chart_type: str, x_field: str, y_field: str, x_type: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    if chart_type == "table":
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Tabular preview of SQL result artifact data.",
            "data": {"values": values},
            "mark": "text",
            "encoding": {"text": {"field": x_field, "type": "nominal"}},
        }
    if chart_type == "histogram":
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": values},
            "mark": "bar",
            "encoding": {
                "x": {"field": x_field, "type": "quantitative", "bin": True},
                "y": {"aggregate": "count", "type": "quantitative"},
            },
        }
    mark = "line" if chart_type == "line" else "bar"
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "mark": mark,
        "encoding": {
            "x": {"field": x_field, "type": x_type},
            "y": {"field": y_field, "type": "quantitative"},
        },
    }


def _title_for(chart_type: str, x_field: str, y_field: str) -> str:
    if chart_type == "line":
        return f"{y_field.title()} trend by {x_field.title()}"
    if chart_type == "bar":
        return f"{y_field.title()} comparison by {x_field.title()}"
    if chart_type == "histogram":
        return f"{y_field.title()} distribution"
    return "SQL result preview"


def _label_for(field: str) -> str:
    return field.replace("_", " ").title()
