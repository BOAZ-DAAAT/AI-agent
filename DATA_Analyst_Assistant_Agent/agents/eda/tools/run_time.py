"""@tool run_time — 시계열 추세 + 시즌성."""

from __future__ import annotations

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.time_skill import run_time_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools._common import _emit_and_dump


@tool
def run_time() -> str:
    """time_skill: 시계열 추세 + 시즌성 분석을 한 번에 수행한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_time_skill(
        ctx.df, measure_cols=ctx.measure_cols, time_cols=ctx.time_cols, key_col=ctx.key_col))
