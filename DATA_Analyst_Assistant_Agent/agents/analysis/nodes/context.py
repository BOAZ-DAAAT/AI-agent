from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisContext
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def build_analysis_context(
    state: OrchestrationState,
    dataframe: pd.DataFrame,
    eda_profiles: list[dict[str, Any]],
    *,
    question_type: str | None = None,
    sample_limit: int = 5,
) -> AnalysisContext:
    """Expose only task-relevant data and a small sample to an optional LLM planner."""

    numeric = list(dataframe.select_dtypes(include="number").columns)
    categorical = [column for column in dataframe.columns if column not in numeric]
    temporal = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
        or any(token in column.casefold() for token in ("date", "time", "month", "year"))
    ]
    samples = []
    if not dataframe.empty:
        samples = dataframe.head(sample_limit).where(pd.notna(dataframe), None).to_dict(orient="records")
    column_profiles = _column_profiles(dataframe, numeric)

    quality_statuses: list[str] = []
    issues: list[str] = []
    for profile in eda_profiles:
        status = profile.get("quality_status")
        if status:
            quality_statuses.append(str(status))
        issues.extend(str(item) for item in profile.get("key_issues", []) or [])

    plan = state.plan
    return AnalysisContext(
        user_question=state.user_query,
        goal=state.goal or (plan.goal if plan else state.user_query),
        route_kind=state.route_kind or (plan.route_kind if plan else "simple"),
        question_type=question_type or _question_type_from_state(state),
        metric_hint=plan.metric if plan else None,
        dimension_hint=plan.dimension if plan else None,
        row_count=len(dataframe),
        columns=list(dataframe.columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        temporal_columns=temporal,
        column_profiles=column_profiles,
        sample_rows=samples,
        eda_quality_statuses=quality_statuses,
        eda_key_issues=list(dict.fromkeys(issues)),
        source_artifact_ids=[item for ids in state.artifact_ids.values() for item in ids],
    )


def _question_type_from_state(state: OrchestrationState) -> str | None:
    """Forward-compatible hook for a supervisor-owned question type contract."""

    direct = getattr(state, "question_type", None)
    if direct:
        return str(direct)
    if state.plan:
        planned = getattr(state.plan, "question_type", None)
        if planned:
            return str(planned)
    for source in (state.retry_context or {}, state.error_state or {}):
        value = source.get("question_type")
        if value:
            return str(value)
    return None


def _column_profiles(dataframe: pd.DataFrame, numeric_columns: list[str]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for column in dataframe.columns:
        series = dataframe[column]
        profile: dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if column in numeric_columns:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                profile.update(
                    min=float(values.min()),
                    max=float(values.max()),
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                )
        else:
            profile["top_values"] = {
                str(key): int(value) for key, value in series.value_counts(dropna=False).head(5).items()
            }
        profiles[column] = profile
    return profiles
