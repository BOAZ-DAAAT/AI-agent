"""chart_selector 노드 — 생성된 차트 중 핵심을 LLM이 선별해 key/ 폴더로 복사."""

from __future__ import annotations

import glob
import os
import shutil

from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools import visualize  # OUTPUT_DIR/KEY_DIR 동적 반영(set_output_dirs)
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.chart_selector_skill import run_chart_selector_skill

try:  # 제공자에 따라 openai 예외가 없을 수 있어 방어적으로 import
    from openai import RateLimitError
except Exception:  # noqa: BLE001
    class RateLimitError(Exception):
        pass


def chart_selector_node(state: EDAState) -> dict:
    all_charts = sorted(glob.glob(os.path.join(visualize.OUTPUT_DIR, "*.png")))
    if not all_charts:
        return {"key_charts": []}

    analysis_results = {
        "inspect":      state.get("inspect_result", ""),
        "quality":      state.get("quality_result", ""),
        "distribution": state.get("distribution_result", ""),
        "comparison":   state.get("comparison_result", ""),
        "relationship": state.get("relationship_result", ""),
        "time":         state.get("time_result", ""),
    }
    stat = state.get("statistical_metadata", {})

    def _run(ar, st):
        return run_chart_selector_skill(
            chart_paths=all_charts,
            user_question=state["user_question"],
            analysis_results=ar,
            question_type=state.get("question_type", ""),
            statistical_metadata=st,
            priority_metrics=state.get("analysis_plan", {}).get("priority_metrics", []),
        )

    try:
        key_charts = _run(analysis_results, stat)
    except RateLimitError:
        truncated = {k: (v[:300] + "...") if isinstance(v, str) and len(v) > 300 else v
                     for k, v in analysis_results.items()}
        clustering = stat.get("clustering", {})
        slim_stat = {"clustering": {k: v for k, v in clustering.items() if k != "cluster_labels"}}
        key_charts = _run(truncated, slim_stat)

    # key/ 폴더 초기화 후 선별 차트 복사
    for f in glob.glob(os.path.join(visualize.KEY_DIR, "*.png")):
        os.remove(f)
    for src in key_charts:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(visualize.KEY_DIR, os.path.basename(src)))

    return {"key_charts": key_charts}
