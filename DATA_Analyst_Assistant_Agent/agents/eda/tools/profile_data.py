"""@tool profile_data — 데이터 기본 구조/기초통계."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.profiling import get_basic_profile


@tool
def profile_data() -> str:
    """데이터 기본 구조(shape, 컬럼 타입, 카디널리티, 시간 컬럼 여부, 기초통계)를 반환한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    df = ctx.df
    result = get_basic_profile(df)
    cardinality = {col: int(df[col].nunique()) for col in df.columns}
    time_cols = [c for c in df.columns if "datetime" in str(df[c].dtype) or "date" in c.lower()]
    return json.dumps({
        "shape": result["shape"],
        "dtypes": result["dtypes"],
        "cardinality": cardinality,
        "time_columns": time_cols,
        "describe": {c: {str(k): str(v) for k, v in s.items()} for c, s in result["describe"].items()},
    }, ensure_ascii=False)
