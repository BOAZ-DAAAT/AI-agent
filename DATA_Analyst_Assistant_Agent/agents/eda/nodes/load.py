"""load_mart 노드 + 컬럼 의미 분류 헬퍼.

원본 `eda_agent/eda_agent.py` 의 load_mart_node / _classify_columns /
_select_best_key_col / _is_meaningless_id 를 분리한 것.

[변경점] 원본은 DB 마트 테이블을 직접 로드했으나, 현재 아키텍처에서는
SQL 에이전트가 넘긴 CSV 아티팩트로부터 만들어진 DataFrame 을 wrapper(agent.py)가
EdaContext.df 에 미리 채워 넣는다. 이 노드는 그 df 를 받아 컬럼 의미를 분류하고
key/measure/time/count 컬럼을 확정해 컨텍스트를 갱신한다.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import classify_columns_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def _classify_columns(df: pd.DataFrame, measure_cols: list) -> dict:
    """LLM이 컬럼명 + 샘플값을 보고 시간 컬럼 / 표본 수 컬럼을 의미 기반으로 분류."""
    # 1차: dtype으로 확실한 시간 컬럼 추출
    obvious_time = [c for c in df.columns if "datetime" in str(df[c].dtype)]

    candidate_cols = [c for c in df.columns if c not in (measure_cols or [])]
    if not candidate_cols:
        return {"time_columns": obvious_time, "count_column": ""}

    sample = df[candidate_cols].head(3).to_dict(orient="list")
    prompt = classify_columns_prompt(list(df.columns), measure_cols, sample)

    try:
        response = get_llm().invoke(prompt).content.strip()
        result = safe_json_parse(response, {})
        time_cols = list(set(obvious_time + result.get("time_columns", [])))
        count_col = result.get("count_column", "")
        time_cols = [c for c in time_cols if c in df.columns]
        count_col = count_col if count_col in df.columns else ""
    except Exception:
        time_cols = obvious_time
        count_col = ""

    return {"time_columns": time_cols, "count_column": count_col}


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
        obvious_time = [c for c in df.columns if "datetime" in str(df[c].dtype)]
        ctx.time_cols = ctx.time_cols or obvious_time
        ctx.count_col = ctx.count_col or ""
        ctx.question_type = state.get("question_type", "") or ctx.question_type
        ctx.priority_metrics = []
        return {
            "time_columns":    ctx.time_cols,
            "count_column":    ctx.count_col,
            "has_time_column": len(ctx.time_cols) > 0,
            "error_log":       [],
        }

    mart_design = state.get("mart_design", {}) or {}
    key_columns    = mart_design.get("key_columns", [])
    dimension_cols = mart_design.get("dimension_columns", [])
    measure_cols   = mart_design.get("measure_columns") or None

    # measure_cols 미지정 시 수치형 컬럼으로 추론
    if not measure_cols:
        measure_cols = list(df.select_dtypes(include=["float64", "int64"]).columns) or None

    # key_col 선택: dimension 우선 → key_columns 중 유의미한 것 → 범주형 폴백
    if dimension_cols:
        key_col = dimension_cols[0]
    else:
        key_col = _select_best_key_col(df, key_columns, measure_cols or [])
    if key_col is None:
        cat_cols = list(df.select_dtypes(include=["object"]).columns)
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
        "error_log":       [],
    }
