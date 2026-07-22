"""@tool run_clustering -- K-means cluster structure discovery."""

from __future__ import annotations

from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.clustering_skill import run_clustering_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools._common import _emit_and_dump


@tool
def run_clustering() -> str:
    """clustering_skill: run K-means clustering and return cluster diagnostics."""
    ctx = get_context()
    if ctx.df is None:
        return '{"skip": true, "reason": "dataframe is not loaded"}'
    return _emit_and_dump(
        ctx,
        run_clustering_skill(
            df=ctx.df,
            measure_cols=ctx.measure_cols,
            key_col=ctx.key_col,
            question_type=ctx.question_type,
        ),
    )
