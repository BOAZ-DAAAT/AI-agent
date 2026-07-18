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


def _slim_stat_for_curation(stat: dict) -> dict:
    """차트 큐레이션 프롬프트엔 statistical_metadata 전체가 아니라 실제로 쓰는 필드만 넘긴다.

    _call_llm_remove의 큐레이션 가이드(질문직답/가설근거/종합비교/순위/클러스터)는
    correlation_pairs(관계 차트 판단)·group_comparison(bar/heatmap 우선순위)·
    clustering(cluster_chart_rule 근거)만 참조한다. 컬럼별 distribution 전체 블록·
    cautions·analysis_constraints 등은 큐레이션 판단에 안 쓰이는데 토큰만 크게
    차지해서(#194) 여기선 뺀다 — insight/hypothesis로 가는 원본 state는 안 건드림.
    """
    clustering = stat.get("clustering", {}) or {}
    return {
        "correlation_pairs": stat.get("correlation_pairs", {}),
        "group_comparison":  stat.get("group_comparison", {}),
        "clustering":        {k: v for k, v in clustering.items() if k != "cluster_labels"},
    }


_ANALYSIS_RESULT_CHAR_LIMIT = 300  # 노드당 원문 요약 상한 — 차트 큐레이션엔 전체 서술이 불필요(#194)


def _truncate(text: str, limit: int = _ANALYSIS_RESULT_CHAR_LIMIT) -> str:
    return (text[:limit] + "...") if isinstance(text, str) and len(text) > limit else text


def chart_selector_node(state: EDAState) -> dict:
    all_charts = sorted(glob.glob(os.path.join(visualize.OUTPUT_DIR, "*.png")))
    if not all_charts:
        return {"key_charts": [], "key_chart_captions": {}}

    # 각 분석 노드의 원문 요약을 통째로 넘기지 않는다 — 큐레이션 판단엔 핵심 몇 문장이면
    # 충분한데 전체 서술을 다시 먹이면 토큰만 커진다(#194, RateLimitError 폴백에만 있던
    # 절단 패턴을 기본 경로에도 적용).
    analysis_results = {
        "inspect":      _truncate(state.get("inspect_result", "")),
        "quality":      _truncate(state.get("quality_result", "")),
        "distribution": _truncate(state.get("distribution_result", "")),
        "comparison":   _truncate(state.get("comparison_result", "")),
        "relationship": _truncate(state.get("relationship_result", "")),
        "time":         _truncate(state.get("time_result", "")),
    }
    stat = _slim_stat_for_curation(state.get("statistical_metadata", {}))
    # 가설도 마찬가지 — 차트 선정엔 "무엇을 검증하는지"만 필요하지 H0/H1/검증방법 전문은 불필요.
    hypotheses = _truncate(state.get("hypotheses", ""), limit=600)

    def _run(ar, st, hyp):
        return run_chart_selector_skill(
            chart_paths=all_charts,
            user_question=state["user_question"],
            analysis_results=ar,
            question_type=state.get("question_type", ""),
            statistical_metadata=st,
            priority_metrics=state.get("analysis_plan", {}).get("priority_metrics", []),
            hypotheses=hyp,   # 가설 근거 차트 우선 유지 (#71 B)
        )

    try:
        key_charts, captions, visual_debug = _run(analysis_results, stat, hypotheses)
    except RateLimitError:
        slim_stat = {"clustering": stat.get("clustering", {})}
        key_charts, captions, visual_debug = _run(analysis_results, slim_stat, hypotheses)

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
