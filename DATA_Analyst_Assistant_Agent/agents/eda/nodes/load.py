"""load_mart 노드 + 컬럼 의미 분류 헬퍼.

원본 `eda_agent/eda_agent.py` 의 load_mart_node / _classify_columns /
_select_best_key_col / _is_meaningless_id 를 분리한 것.

[변경점] 원본은 DB 마트 테이블을 직접 로드했으나, 현재 아키텍처에서는
SQL 에이전트가 넘긴 CSV 아티팩트로부터 만들어진 DataFrame 을 wrapper(agent.py)가
EdaContext.df 에 미리 채워 넣는다. 이 노드는 그 df 를 받아 컬럼 의미를 분류하고
key/measure/time/count 컬럼을 확정해 컨텍스트를 갱신한다.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.dtype_utils import categorical_object_columns, usable_time_columns
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState

_TIME_NAME_RE = re.compile(
    r"(^|_)(time|ts|dt|day|month|year)($|_)|(_at$)",
    re.IGNORECASE,
)


def _candidate_time_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        dtype = str(df[col].dtype).lower()
        name = str(col)
        lowered = name.lower()
        if (
            "datetime" in dtype
            or "date" in lowered
            or "datetime" in lowered
            or "timestamp" in lowered
            or _TIME_NAME_RE.search(name)
        ):
            candidates.append(col)
    return candidates


def _safe_usable_time_columns(df: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[str], str, str]:
    try:
        cols = usable_time_columns(df, candidate_cols)
    except Exception as exc:  # noqa: BLE001
        return [], "skipped_due_error", str(exc)
    if cols:
        return cols, "ok", ""
    return [], "no_usable_time_columns", "No column was suitable for time-series charting."


def _classify_columns(df: pd.DataFrame, measure_cols: list) -> dict:
    """Classify load-time columns without an LLM call."""
    time_cols, time_status, time_reason = _safe_usable_time_columns(df, _candidate_time_columns(df))
    return {
        "time_columns": time_cols,
        "time_detection_status": time_status,
        "time_skip_reason": time_reason,
        "count_column": _select_count_column(df, measure_cols or []),
    }


def _select_count_column(df: pd.DataFrame, measure_cols: list) -> str:
    measure_set = set(measure_cols or [])
    markers = ("count", "cnt", "frequency", "qty", "quantity", "num_", "n_")
    for col in df.columns:
        lowered = str(col).lower()
        if (
            col in measure_set
            and any(marker in lowered for marker in markers)
            and pd.api.types.is_numeric_dtype(df[col])
        ):
            return str(col)
    for col in df.columns:
        lowered = str(col).lower()
        if any(marker in lowered for marker in markers) and pd.api.types.is_numeric_dtype(df[col]):
            return str(col)
    return ""

def _is_meaningless_id(df: pd.DataFrame, col: str) -> bool:
    """컬럼이 UUID/해시처럼 시각화에 무의미한 고유 ID인지 판별."""
    if col not in df.columns:
        return False
    n_unique = df[col].nunique()
    n_rows = len(df)
    if n_rows and n_unique >= n_rows * 0.5:
        return True
    sample = df[col].dropna().astype(str).head(10)
    if len(sample) and sample.str.match(r'^[0-9a-f]{32,}$').mean() >= 0.8:
        return True
    return False


def _select_best_key_col(df: pd.DataFrame, key_columns: list, measure_cols: list) -> Optional[str]:
    """key_columns 중 시각화에 의미 있는 컬럼을 자동 선택."""
    candidates = [c for c in key_columns if c in df.columns]
    measure_set = set(measure_cols or [])
    for col in candidates:
        if col in measure_set:
            continue
        if not _is_meaningless_id(df, col):
            return col
    non_measure = [c for c in candidates if c not in measure_set]
    if non_measure:
        return min(non_measure, key=lambda c: df[c].nunique())
    return candidates[0] if candidates else None


def load_mart_node(state: EDAState) -> dict:
    """wrapper 가 채워둔 EdaContext.df 를 받아 컬럼 의미를 확정한다."""
    ctx = get_context()
    df = ctx.df
    if df is None or df.empty:
        raise RuntimeError("EDA: 분석할 DataFrame이 비어 있습니다 (CSV 아티팩트 없음).")

    # 입력계약(fixture/향후 SQL)이 컬럼 역할을 미리 채워뒀으면 LLM 분류를 건너뛴다(토큰 절감).
    if ctx.measure_cols and ctx.key_col:
        time_cols, time_status, time_reason = _safe_usable_time_columns(df, _candidate_time_columns(df))
        ctx.time_cols = ctx.time_cols or time_cols
        ctx.count_col = ctx.count_col or ""
        ctx.question_type = state.get("question_type", "") or ctx.question_type
        ctx.priority_metrics = []
        return {
            "time_columns":    ctx.time_cols,
            "count_column":    ctx.count_col,
            "has_time_column": len(ctx.time_cols) > 0,
            "time_detection_status": time_status if not ctx.time_cols else "ok",
            "time_skip_reason": "" if ctx.time_cols else time_reason,
            "error_log":       [],
        }

    mart_design = state.get("mart_design", {}) or {}
    contract = state.get("analysis_data_contract", {}) or {}
    key_columns    = mart_design.get("key_columns", [])
    dimension_cols = mart_design.get("dimension_columns", [])
    measure_cols   = mart_design.get("measure_columns") or None
    if contract:
        key_columns = key_columns or list(contract.get("grain_columns") or contract.get("entity_keys") or [])
        derived_columns = [
            item for item in (contract.get("derived_columns") or []) if isinstance(item, dict)
        ]
        dimension_cols = dimension_cols or [
            str(item.get("output_column"))
            for item in derived_columns
            if item.get("role") in {"dimension", "attribute"} and item.get("output_column")
        ]
        measure_cols = measure_cols or [
            str(item.get("output_column"))
            for item in derived_columns
            if item.get("role") == "measure" and item.get("output_column")
        ] or None

    # measure_cols 미지정 시 수치형 컬럼으로 추론
    if not measure_cols:
        measure_cols = list(df.select_dtypes(include=["float64", "int64"]).columns) or None

    # key_col 선택: dimension 우선 → key_columns 중 유의미한 것 → 범주형 폴백
    if dimension_cols:
        key_col = dimension_cols[0]
    else:
        key_col = _select_best_key_col(df, key_columns, measure_cols or [])
    if key_col is None:
        cat_cols = categorical_object_columns(df)
        key_col = cat_cols[0] if cat_cols else None

    col_meta = _classify_columns(df, measure_cols or [])

    # 컨텍스트 갱신 (downstream 노드/툴이 참조)
    ctx.key_col       = key_col
    ctx.measure_cols  = measure_cols
    ctx.time_cols     = col_meta["time_columns"]
    ctx.count_col     = col_meta["count_column"]
    # target: 앞단 plan_metric이 실제 컬럼이면 채택(대개 None — 그땐 가설 노드가 priority로 폴백)
    pm = state.get("plan_metric")
    ctx.target_col    = pm if (pm and pm in df.columns) else None
    ctx.question_type = state.get("question_type", "")
    ctx.priority_metrics = []  # planner 실행 후 갱신됨

    return {
        "time_columns":    ctx.time_cols,
        "count_column":    ctx.count_col,
        "has_time_column": len(ctx.time_cols) > 0,
        "time_detection_status": col_meta["time_detection_status"],
        "time_skip_reason": col_meta["time_skip_reason"],
        "error_log":       [],
    }
