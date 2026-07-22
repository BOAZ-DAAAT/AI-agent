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


def _is_branch_instruction(question: str) -> bool:
    return "[추가 지시사항]" in (question or "")


def _explicit_boxplot_requested(question: str) -> bool:
    q = (question or "").lower()
    return (
        "boxplot" in q
        or "box plot" in q
        or "박스플롯" in q
        or "박스 플롯" in q
        or "상자그림" in q
    )


def _requested_boxplot_tokens(question: str) -> list[str]:
    q = (question or "").lower()
    tokens: list[str] = []
    if "review_score" in q or "리뷰" in q or "평점" in q:
        tokens.append("review_score")
    if "delivery_days" in q or "배송 소요일" in q or "배송소요일" in q:
        tokens.append("delivery_days")
    if "delivery_delay_days" in q or "지연일" in q:
        tokens.append("delivery_delay_days")
    return tokens


def _requested_boxplot_prefer_grouped(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in ("초과", "지연 여부", "지연여부", "is_delivery_delayed", "그룹", "군별", "별로"))


def _requested_boxplot_caption(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith("groupedbox_"):
        return "사용자가 분기 지시에서 요청한 박스플롯입니다. 그룹별 리뷰 점수 분포 차이를 확인하기 위해 key chart에 포함했습니다."
    return "사용자가 분기 지시에서 요청한 박스플롯입니다. 리뷰 점수의 중앙값, 사분위 범위, 이상치 분포를 확인하기 위해 key chart에 포함했습니다."


def _requested_key_charts(all_charts: list[str], question: str) -> tuple[list[str], dict[str, str]]:
    if not (_is_branch_instruction(question) and _explicit_boxplot_requested(question)):
        return [], {}

    tokens = _requested_boxplot_tokens(question)
    candidates = [p for p in all_charts if "box" in os.path.basename(p).lower()]
    if tokens:
        candidates = [
            p for p in candidates
            if any(token in os.path.basename(p).lower() for token in tokens)
        ]
    if not candidates:
        return [], {}

    prefer_grouped = _requested_boxplot_prefer_grouped(question)

    def _rank(path: str) -> tuple[int, str]:
        name = os.path.basename(path).lower()
        grouped = name.startswith("groupedbox_")
        if prefer_grouped:
            return (0 if grouped else 1, name)
        return (0 if name.startswith("box_") else 1, name)

    selected = sorted(candidates, key=_rank)[:1]
    captions = {os.path.basename(path): _requested_boxplot_caption(path) for path in selected}
    return selected, captions


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
    requested_charts, requested_captions = _requested_key_charts(all_charts, state["user_question"])
    seen = {os.path.abspath(path) for path in key_charts}
    for path in requested_charts:
        if os.path.abspath(path) not in seen:
            key_charts.append(path)
            seen.add(os.path.abspath(path))
    captions = {**captions, **requested_captions}

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
