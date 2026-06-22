"""EDA 플래너 (적응형 컨트롤러 루프).

★ 이 에이전트의 '두뇌'. 매 라운드:
  1) 데이터 모양으로 실행 가능한(전제조건 충족) + 아직 안 돌린 분석만 추림  ← 결정론 가드레일
  2) 지금까지 결과를 LLM에 보여주고 다음 분석 1개(또는 done)를 고름        ← 에이전시
  3) 멈춤(done / 도구 소진 / 라운드 캡 / 재선택 방지) 판단

분석 노드(nodes/) 로직은 건드리지 않는다. 플래너는 '무엇을 언제 돌릴지'만 정한다.
graph.py 에서 'planner' 노드로 등록되어 분석 노드들과 순환(⇄)한다.
"""

from __future__ import annotations

from typing import Any, Dict

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import planner_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState

MAX_ANALYSES = 5  # 최대 분석 실행 횟수 (하드 캡 — 배회 방지)

# ─────────────────────────────
# 분석 도구 레지스트리 (분석 family = 카드)
#   precond(shape) → bool : 데이터 모양으로 실행 가능 여부 (결정론 가드레일)
#   result_field          : 실행 결과가 저장되는 state 키 (완료 추론용)
# ─────────────────────────────
ANALYSIS_TOOLS = [
    {"name": "quality",      "desc": "결측·이상치·중복·표본 신뢰도 점검. 분석 신뢰도 확인용",
     "precond": lambda s: True,                                  "result_field": "quality_result"},
    {"name": "distribution", "desc": "각 지표의 분포·치우침·이상치 파악",
     "precond": lambda s: s["n_numeric"] >= 1,                   "result_field": "distribution_result"},
    {"name": "comparison",   "desc": "카테고리별 그룹 비교 (상위/하위·히트맵)",
     "precond": lambda s: s["n_cat"] >= 1 and s["n_numeric"] >= 1, "result_field": "comparison_result"},
    {"name": "relationship", "desc": "변수 간 상관·trade-off 탐색",
     "precond": lambda s: s["n_numeric"] >= 2,                   "result_field": "relationship_result"},
    {"name": "time",         "desc": "시계열 추세·계절성 분석",
     "precond": lambda s: s["n_time"] >= 1,                      "result_field": "time_result"},
    {"name": "clustering",   "desc": "다지표 군집 구조 발견 (K-means)",
     "precond": lambda s: s["n_numeric"] >= 2 and s["row_count"] >= 10, "result_field": "clustering_result"},
]
ANALYSIS_NAMES = [t["name"] for t in ANALYSIS_TOOLS]


def _data_shape(ctx) -> Dict[str, Any]:
    df = ctx.df
    if df is None:
        return {"n_numeric": 0, "n_cat": 0, "n_time": 0, "row_count": 0,
                "numeric_cols": [], "cat_cols": [], "time_cols": []}
    if ctx.measure_cols:
        numeric = [c for c in ctx.measure_cols if c in df.columns]
    else:
        numeric = list(df.select_dtypes(include=["float64", "int64"]).columns)
    cat = list(df.select_dtypes(include=["object"]).columns)
    time = ctx.time_cols or []
    return {"n_numeric": len(numeric), "n_cat": len(cat), "n_time": len(time),
            "row_count": len(df), "numeric_cols": numeric, "cat_cols": cat, "time_cols": time}


def planner_node(state: EDAState) -> dict:
    ctx = get_context()
    shape = _data_shape(ctx)
    log = state.get("controller_log", [])
    round_idx = state.get("round", 0)

    # 이미 시도한 분석 (결과 비어도 재선택 방지)
    attempted = {e["choice"] for e in log if e.get("choice") and e["choice"] != "done"}
    feasible = [t for t in ANALYSIS_TOOLS if t["precond"](shape) and t["name"] not in attempted]

    # 지금까지 결과 요약 (LLM 입력용)
    completed_findings = {}
    for t in ANALYSIS_TOOLS:
        if t["name"] in attempted:
            val = state.get(t["result_field"])
            completed_findings[t["name"]] = val if isinstance(val, str) else str(val)

    # ── 하드 멈춤: 도구 소진 / 라운드 캡 ──
    if not feasible or len(attempted) >= MAX_ANALYSES:
        choice, reason = "done", ("실행 가능한 분석 소진" if not feasible else f"최대 분석 {MAX_ANALYSES}회 도달")
    else:
        need_priority = round_idx == 0 or not ctx.priority_metrics
        prompt = planner_prompt(
            user_question=state["user_question"],
            question_type=state.get("question_type", ""),
            data_shape=shape,
            feasible_cards=[{"name": t["name"], "desc": t["desc"]} for t in feasible],
            completed_findings=completed_findings,
            round_idx=round_idx,
            max_rounds=MAX_ANALYSES,
            need_priority=need_priority,
        )
        fb = state.get("validation_feedback")
        if fb:
            prompt += f"\n[직전 검증 지적 — 이를 보완할 분석을 우선 고려하라]\n{fb}\n"
        try:
            raw = get_llm().invoke(prompt).content.strip()
            decision = safe_json_parse(raw, {"next": "done", "reason": "파싱 실패로 종료"})
        except Exception as exc:  # noqa: BLE001
            decision = {"next": "done", "reason": f"플래너 오류로 종료: {exc}"}

        choice = decision.get("next", "done")
        reason = decision.get("reason", "")
        # 환각 방지: feasible 목록 밖 선택이면 done
        if choice != "done" and choice not in {t["name"] for t in feasible}:
            choice, reason = "done", f"유효하지 않은 선택({choice}) → 종료"

        # 첫 라운드: priority_metrics 확정 (df에 실제 있는 컬럼만)
        if need_priority and decision.get("priority_metrics"):
            df_cols = set(ctx.df.columns) if ctx.df is not None else set()
            ctx.priority_metrics = [
                m for m in decision["priority_metrics"]
                if isinstance(m, dict) and m.get("metric") in df_cols
            ]

    entry = {"round": round_idx, "choice": choice, "reason": reason}
    return {
        "next_analysis": choice,
        "round": round_idx + 1,
        "controller_log": log + [entry],
        # 하위호환: 다운스트림(report 등)이 보던 analysis_plan 에 priority_metrics 보관
        "analysis_plan": {"priority_metrics": ctx.priority_metrics, "controller_log": log + [entry]},
    }


def route_after_planner(state: EDAState):
    """플래너 선택에 따라 분석 노드로 가거나, 종료하면 insight로."""
    choice = state.get("next_analysis", "done")
    if choice in ANALYSIS_NAMES:
        return choice
    return "insight"
