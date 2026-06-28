"""@tool run_distribution — 분포 분석(히스토그램/박스/범주 빈도)."""

from __future__ import annotations

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.distribution_skill import run_distribution_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools._common import _emit_and_dump


@tool
def run_distribution() -> str:
    """distribution_skill: 히스토그램 / 박스플롯 / 범주형 빈도 분포를 한 번에 분석한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_distribution_skill(
        ctx.df, measure_cols=ctx.measure_cols, question_type=ctx.question_type,
        priority_metrics=ctx.priority_metrics, key_col=ctx.key_col, target_col=ctx.target_col))
