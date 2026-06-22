"""clustering 노드 — K-means 군집 (best-effort, 실패해도 EDA 중단 X)."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import accumulate_chart_requests, append_errors, get_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.clustering_skill import run_clustering_skill


def clustering_node(state: EDAState) -> dict:
    ctx = get_context()
    result, err = run_node_with_retry(
        lambda: run_clustering_skill(
            df=ctx.df,
            measure_cols=ctx.measure_cols,
            key_col=ctx.key_col,
            question_type=ctx.question_type,
        ),
        "clustering",
        fallback={"skip": True, "reason": "클러스터링 오류로 스킵"},
    )
    if isinstance(result, dict):
        accumulate_chart_requests(ctx, result.pop("chart_requests", []))
    return {"clustering_result": result, "error_log": append_errors(state, err)}
