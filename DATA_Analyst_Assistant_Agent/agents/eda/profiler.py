from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.artifact_data import CsvArtifactData


def profile_from_csv_artifacts(csvs: list[CsvArtifactData]) -> dict[str, Any]:
    key_issues: list[str] = []
    source_artifacts = [csv.artifact_id for csv in csvs]
    read_errors = {csv.artifact_id: csv.error for csv in csvs if csv.error}

    if not csvs:
        key_issues.append("No SQL result artifact was available for EDA.")
        df = pd.DataFrame()
    else:
        frames = [csv.dataframe for csv in csvs if csv.error is None]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if read_errors:
        key_issues.extend(f"{artifact_id}: {error}" for artifact_id, error in read_errors.items())
    if csvs and len(df.columns) == 0:
        key_issues.append("SQL result CSV did not include columns.")
    if csvs and len(df) == 0:
        key_issues.append("SQL result CSV contains zero data rows.")

    null_counts = {column: int(count) for column, count in df.isna().sum().items()} if len(df.columns) else {}
    unique_counts = {column: int(df[column].nunique(dropna=True)) for column in df.columns}
    numeric_summary = _numeric_summary(df)
    categorical_top_values = _categorical_top_values(df)

    if not csvs:
        quality_status = "unavailable"
        recommended_next_steps = ["run_sql_preview", "clarify_data_source"]
    elif key_issues:
        quality_status = "needs_review"
        recommended_next_steps = ["review_csv_artifact", "rerun_or_refine_sql"]
    else:
        quality_status = "usable"
        recommended_next_steps = ["continue_to_analysis", "document_limitations"]

    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": null_counts,
        "unique_counts": unique_counts,
        "numeric_summary": numeric_summary,
        "categorical_top_values": categorical_top_values,
        "sample_available": len(df) > 0,
        "quality_status": quality_status,
        "key_issues": key_issues,
        "recommended_next_steps": recommended_next_steps,
        "source_artifacts": source_artifacts,
    }


def _numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    summary = numeric_df.describe().to_dict()
    return {
        column: {
            stat: (None if pd.isna(value) else float(value))
            for stat, value in stats.items()
            if stat in {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
        }
        for column, stats in summary.items()
    }


def _categorical_top_values(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    categorical: dict[str, dict[str, int]] = {}
    for column in df.select_dtypes(exclude="number").columns:
        counts = df[column].fillna("<NA>").astype(str).value_counts(dropna=False).head(5)
        categorical[column] = {str(value): int(count) for value, count in counts.items()}
    return categorical
