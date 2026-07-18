"""@tool run_comparison — 카테고리별 비교(막대/히트맵 등)."""

from __future__ import annotations

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.comparison_skill import run_comparison_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools._common import _emit_and_dump


@tool
def run_comparison() -> str:
    """comparison_skill: 카테고리별 상위/하위 barplot + 히트맵 비교를 한 번에 수행한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_comparison_skill(
        ctx.df, key_col=ctx.key_col, measure_cols=ctx.measure_cols, question_type=ctx.question_type,
        priority_metrics=ctx.priority_metrics, user_question=ctx.user_question))
