from __future__ import annotations

from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def build_analysis_result(
    state: OrchestrationState,
    *,
    dataframe: pd.DataFrame | None = None,
    eda_profiles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = state.plan
    route_kind = state.route_kind or (plan.route_kind if plan else "simple")
    df = dataframe if dataframe is not None else pd.DataFrame()
    method_summary = "SQL 실행 결과 데이터를 기준으로 주요 지표와 구조를 요약했습니다."
    if route_kind == "trend":
        method_summary = "SQL 실행 결과를 바탕으로 시간 흐름 또는 방향성을 확인하는 기술 분석을 수행했습니다."
    elif route_kind == "eda":
        method_summary = "EDA 프로파일 결과를 함께 참고해 데이터 품질을 고려한 기술 분석을 수행했습니다."
    elif route_kind == "comprehensive":
        method_summary = "SQL 결과, EDA 프로파일, 시각화 산출물을 함께 사용해 종합 분석을 수행했습니다."

    metric = plan.metric if plan else None
    dimension = plan.dimension if plan else None
    target = " and ".join(item for item in [dimension, metric] if item)
    finding_subject = target or "요청한 SQL 결과"
    eda_profiles = eda_profiles or []
    has_eda = bool(state.artifact_ids.get("eda_agent"))

    key_findings = [
        f"{finding_subject}를 SQL 실행 결과 산출물을 기준으로 분석했습니다.",
    ]
    limitations = [
        "이번 결과는 현재 실행에서 생성된 SQL 결과 행을 기준으로 해석해야 합니다.",
        "분석 결과는 관측된 패턴을 설명하는 것이며 인과관계를 의미하지 않습니다.",
    ]
    data_quality_notes: list[str] = []

    if route_kind == "trend":
        key_findings.insert(0, "선택된 시간 기준에 대해 방향성 검토가 가능하도록 결과를 정리했습니다.")
    if has_eda:
        key_findings.append("분석 요약을 만들기 전에 EDA 프로파일의 데이터 품질 신호를 함께 검토했습니다.")
        limitations.append("EDA 프로파일은 핵심 품질 신호를 요약한 것이며 전체 데이터 진단을 완전히 대체하지는 않습니다.")
    for profile in eda_profiles:
        quality_status = profile.get("quality_status")
        if quality_status:
            data_quality_notes.append(f"EDA 품질 상태는 {quality_status}입니다.")
        if quality_status and quality_status != "usable":
            limitations.append(f"EDA 품질 상태가 {quality_status}이므로 결과 해석 전에 주요 이슈를 확인해야 합니다.")
        data_quality_notes.extend(profile.get("key_issues", []) or [])

    if df.empty:
        key_findings.append("SQL 결과 산출물에 지표 계산에 사용할 데이터 행이 없습니다.")
        limitations.append("SQL 결과가 비어 있어 행 단위 통계는 계산하지 못했습니다.")
    else:
        key_findings.extend(_data_findings(df, metric=metric, dimension=dimension))
        data_quality_notes.extend(_null_quality_notes(df))

    return {
        "run_id": state.run_id,
        "goal": state.goal,
        "method_summary": method_summary,
        "key_findings": key_findings,
        "limitations": limitations,
        "source_artifacts": {
            "sql": state.artifact_ids.get("sql_agent", []),
            "eda": state.artifact_ids.get("eda_agent", []),
        },
        "data_quality_notes": data_quality_notes,
        "eda_profile_summaries": eda_profiles,
    }


def _data_findings(df: pd.DataFrame, *, metric: str | None, dimension: str | None) -> list[str]:
    findings = [f"SQL 결과는 {len(df)}개 행과 {len(df.columns)}개 컬럼으로 구성되어 있습니다."]
    numeric_cols = list(df.select_dtypes(include="number").columns)
    metric_col = metric if metric in df.columns else (numeric_cols[0] if numeric_cols else None)
    dimension_col = dimension if dimension in df.columns else _first_non_numeric_column(df, numeric_cols)

    if metric_col:
        series = df[metric_col].dropna()
        if not series.empty:
            findings.append(
                f"{metric_col} 값은 최소 {series.min():g}, 최대 {series.max():g}, 평균 {series.mean():.2f}입니다."
            )
    if metric_col and dimension_col and dimension_col in df.columns:
        grouped = df.groupby(dimension_col, dropna=False)[metric_col].sum(numeric_only=True).sort_values(ascending=False)
        if not grouped.empty:
            top_key = str(grouped.index[0])
            findings.append(f"{metric_col} 기준 가장 높은 {dimension_col} 값은 {top_key}이며, 값은 {grouped.iloc[0]:g}입니다.")
    elif dimension_col and dimension_col in df.columns:
        top = df[dimension_col].fillna("<NA>").astype(str).value_counts().head(1)
        if not top.empty:
            findings.append(f"{dimension_col}에서 가장 자주 나타난 값은 {top.index[0]}이며, {int(top.iloc[0])}개 행입니다.")
    return findings


def _first_non_numeric_column(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    for column in df.columns:
        if column not in numeric_cols:
            return column
    return None


def _null_quality_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    null_counts = df.isna().sum()
    for column, count in null_counts.items():
        if int(count) > 0:
            notes.append(f"{column} 컬럼에 결측치가 {int(count)}개 있습니다.")
    return notes
