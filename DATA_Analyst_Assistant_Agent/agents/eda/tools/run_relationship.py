"""@tool run_relationship — 변수 간 관계(상관 히트맵/산점도)."""

from __future__ import annotations

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.relationship_skill import run_relationship_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools._common import _emit_and_dump


@tool
def run_relationship() -> str:
    """relationship_skill: 상관관계 히트맵 + scatter plot을 한 번에 분석한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_relationship_skill(
        ctx.df, measure_cols=ctx.measure_cols, question_type=ctx.question_type, target_col=ctx.target_col))
