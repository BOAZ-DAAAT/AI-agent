"""chart_selector 노드 — 생성된 차트 중 핵심을 LLM이 선별해 key/ 폴더로 복사."""

from __future__ import annotations

import glob
import os
import shutil

from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize  # OUTPUT_DIR/KEY_DIR 동적 반영(set_output_dirs)
from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_selector_skill import run_chart_selector_skill

try:  # 제공자에 따라 openai 예외가 없을 수 있어 방어적으로 import
    from openai import RateLimitError
except Exception:  # noqa: BLE001
    class RateLimitError(Exception):
        pass


def chart_selector_node(state: EDAState) -> dict:
    all_charts = sorted(glob.glob(os.path.join(visualize.OUTPUT_DIR, "*.png")))
    if not all_charts:
        return {"key_charts": [], "key_chart_captions": {}}

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
            hypotheses=state.get("hypotheses", ""),   # 가설 근거 차트 우선 유지 (#71 B)
        )

    try:
        key_charts, captions, visual_debug = _run(analysis_results, stat)
    except RateLimitError:
        truncated = {k: (v[:300] + "...") if isinstance(v, str) and len(v) > 300 else v
                     for k, v in analysis_results.items()}
        clustering = stat.get("clustering", {})
        slim_stat = {"clustering": {k: v for k, v in clustering.items() if k != "cluster_labels"}}
        key_charts, captions, visual_debug = _run(truncated, slim_stat)

    # key/ 폴더 초기화 후 선별 차트 복사
    for f in glob.glob(os.path.join(visualize.KEY_DIR, "*.png")):
        os.remove(f)
    for src in key_charts:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(visualize.KEY_DIR, os.path.basename(src)))

    update: dict = {"key_charts": key_charts, "key_chart_captions": captions}

    # 멀티모달 시각점검이 드롭했거나(왜 빠졌는지) 점검 자체가 실패했으면(꺼진 채 몰랐던 상태
    # 방지) cautions로 흘려서 report까지 추적 가능하게 한다 — validator의 기존 관례와 동일.
    dropped = visual_debug.get("dropped") or []
    check_failures = visual_debug.get("check_failures") or 0
    if dropped or check_failures:
        update["cautions"] = list(state.get("cautions", []) or []) + [{
            "code": "CHART_VISUAL_CHECK_ISSUE",
            "source": "eda_chart_selector",
            "severity": "low" if not dropped else "medium",
            "message_ko": (
                f"멀티모달 렌더링 점검에서 {len(dropped)}장 드롭"
                + (f", 점검 자체 실패 {check_failures}건" if check_failures else "")
            ),
            "recommended_action": ["review_key_charts_visually"] if dropped else [],
            "details": {"dropped": dropped, "check_failures": check_failures},
        }]

    # 캡션(선정 이유)은 아티팩트 메타데이터로 실려 분석 에이전트의 차트 읽기(멀티모달)를 돕는다
    return update
