"""EDA 검증기 (자기검증 노드).

★ 이 에이전트의 '비평가'. planner(옹호자)가 만든 결과를 깐깐하게 감사한다.
  1) 결정론 체크 — 에러/빈 출력/분석 유무 (코드, 빠름)
  2) LLM 감사 — 질문 정합성·환각·과장·누락 (깐깐하게)
  3) 문제가 있으면 '어디를' 고칠지 판정 → 그 노드로 되돌림 (타겟 재시도)

되돌림 대상: planner(분석 부족) / insight(해석·환각) / hypothesis(가설 부실).
총 2회까지만 재시도하고, 소진되면 verdict만 기록하고 통과한다(밖의 validator가 다시 봄).
"""

from __future__ import annotations

from typing import Any, Dict

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import validator_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState

MAX_VALIDATION_RETRIES = 2          # 총 재시도 캡 (무한 루프 방지)
_RETRY_TARGETS = {"planner", "insight", "hypothesis"}
_FALLBACK_TEXTS = {"", "인사이트 생성 실패", "가설 생성 실패", "요약 생성 실패", "분석 스킵 (오류로 인해 생략됨)"}


def _completed_analyses(state: EDAState) -> list:
    log = state.get("controller_log", []) or []
    return sorted({e["choice"] for e in log if e.get("choice") and e["choice"] != "done"})


def _deterministic_fail(state: EDAState):
    """코드로 잡히는 명백한 실패 → (retry_target, reason) 또는 None."""
    insight = (state.get("insight_result") or "").strip()
    hyp = (state.get("hypotheses") or "").strip()
    if insight in _FALLBACK_TEXTS:
        return "insight", "insight가 생성되지 않음(빈 값/실패)"
    if hyp in _FALLBACK_TEXTS:
        return "hypothesis", "가설이 생성되지 않음(빈 값/실패)"
    if not _completed_analyses(state):
        return "planner", "분석이 하나도 실행되지 않음"
    if not state.get("statistical_metadata"):
        return "planner", "통계 메타데이터가 비어있음"
    return None


def _check_chart_requests(state: EDAState) -> list:
    """차트 주문서를 코드로 검증한다(LLM 없음): 필수 필드 + columns가 실제 df에 존재하는지 대조.
    그림 렌더는 불필요. 문제를 리스트로 반환(파이프라인은 막지 않고 기록만 한다)."""
    issues: list = []
    reqs = state.get("chart_requests", []) or []
    df = getattr(get_context(), "df", None)
    df_cols = set(df.columns) if df is not None else None

    for i, r in enumerate(reqs):
        cols = (r.get("columns") or {})
        if not r.get("intent"):
            issues.append(f"#{i}: intent 누락")
        if not cols:
            issues.append(f"#{i}: columns 누락")
        elif df_cols is not None:
            missing = [c for c in cols if c not in df_cols]
            if missing:
                issues.append(f"#{i}: 존재하지 않는 컬럼 참조 {missing}")
        if not r.get("stats"):
            issues.append(f"#{i}: stats 비어있음")
    return issues


def validator_node(state: EDAState) -> dict:
    retries = state.get("validation_retries", 0)
    cap_reached = retries >= MAX_VALIDATION_RETRIES

    # ── 1) 결정론 체크 ──
    det = _deterministic_fail(state)
    if det and not cap_reached:
        target, reason = det
        verdict = {"status": "retry", "retry_target": target, "reason": reason, "feedback": reason}
    elif det:  # 실패지만 재시도 소진 → 기록만 하고 통과
        verdict = {"status": "pass", "retry_target": "none",
                   "reason": f"검증 미통과(재시도 소진): {det[1]}", "feedback": ""}
    else:
        # ── 2) LLM 감사 (깐깐하게) ──
        prompt = validator_prompt(
            user_question=state["user_question"],
            question_type=state.get("question_type", ""),
            completed_analyses=_completed_analyses(state),
            statistical_metadata=state.get("statistical_metadata", {}),
            insight_result=state.get("insight_result", ""),
            hypotheses=state.get("hypotheses", ""),
            final_summary=state.get("final_summary", ""),
        )
        try:
            raw = get_llm().invoke(prompt).content.strip()
            verdict = safe_json_parse(raw, {"status": "pass", "retry_target": "none",
                                            "reason": "검증 파싱 실패 → 통과", "feedback": ""})
        except Exception as exc:  # noqa: BLE001
            verdict = {"status": "pass", "retry_target": "none",
                       "reason": f"검증 오류 → 통과: {exc}", "feedback": ""}

        # ── 3) verdict 정규화 + 캡 적용 ──
        if verdict.get("status") == "retry":
            if cap_reached:
                verdict = {"status": "pass", "retry_target": "none",
                           "reason": "재시도 소진 후 통과", "feedback": ""}
            elif verdict.get("retry_target") not in _RETRY_TARGETS:
                verdict["retry_target"] = "planner"  # 타겟 불명확 시 기본값

    # ── 차트 주문서 코드 검증(LLM 없음) — 기록만, 재시도 트리거 아님 ──
    verdict["chart_issues"] = _check_chart_requests(state)

    # ── 결과 반영 ──
    update: Dict[str, Any] = {"validation_result": verdict}
    if verdict["status"] == "retry":
        update["validation_retries"] = retries + 1
        update["validation_feedback"] = verdict.get("feedback") or verdict.get("reason", "")
    else:
        update["validation_feedback"] = ""  # 통과 시 피드백 초기화
    return update


def route_after_validator(state: EDAState):
    """검증 결과에 따라 타겟 노드로 되돌리거나, 통과면 chart_selector로."""
    verdict = state.get("validation_result", {}) or {}
    if verdict.get("status") == "retry":
        target = verdict.get("retry_target", "planner")
        if target in _RETRY_TARGETS:
            return target
    return "chart_selector"
