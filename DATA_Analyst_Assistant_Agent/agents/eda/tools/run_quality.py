"""@tool run_quality — 결측/이상치/중복/표본 신뢰도 점검."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.data_quality_skill import run_data_quality_skill


@tool
def run_quality() -> str:
    """data_quality_skill: 결측치 / 이상치 / 중복 / 표본 신뢰도를 한 번에 점검한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(
        run_data_quality_skill(ctx.df, key_col=ctx.key_col, measure_cols=ctx.measure_cols, count_col=ctx.count_col),
        ensure_ascii=False,
    )
