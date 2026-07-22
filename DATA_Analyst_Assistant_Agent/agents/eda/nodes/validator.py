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

MAX_VALIDATION_RETRIES = 1          # 총 재시도 캡 (무한 루프 방지, #194 — 2회는 비용 대비 효과가 낮아 축소)
_RETRY_TARGETS = {"planner", "insight", "hypothesis"}
_FALLBACK_TEXTS = {"", "인사이트 생성 실패", "가설 생성 실패", "요약 생성 실패", "분석 스킵 (오류로 인해 생략됨)"}

# 재시도 소진 시 강제 통과되는 결정론적 실패 중, EDA 전체를 다시 돌리면 나아질 가능성이
# 큰 것만(LLM 호출성 fallback) supervisor에게 "retryable"로 알린다. planner 계열(분석
# 미실행/통계메타 없음)은 데이터·라우팅 구조 문제일 가능성이 커서 비재시도로 둔다.
_RETRYABLE_FAILURE_CODES = {"insight_fallback", "hypothesis_fallback"}


def _completed_analyses(state: EDAState) -> list:
    log = state.get("controller_log", []) or []
    return sorted({e["choice"] for e in log if e.get("choice") and e["choice"] != "done"})


_FACTS_FIELDS = ["inspect_facts", "quality_facts", "distribution_facts",
                 "comparison_facts", "relationship_facts", "time_facts",
                 "clustering_facts"]


def _all_analysis_facts(state: EDAState) -> dict:
    """분석노드 6개가 결정론적으로 계산해 뽑은 facts를 모은다 — statistical_metadata에
    안 흘러들어간 그룹별 수치(예: 분포노드의 grouped_box)도 여기엔 있어, validator가
    이걸 '없는 값'으로 오판(false positive)하지 않게 한다."""
    return {k: state.get(k) for k in _FACTS_FIELDS if state.get(k)}


def _deterministic_fail(state: EDAState):
    """코드로 잡히는 명백한 실패 → (retry_target, reason, failure_code) 또는 None."""
    insight = (state.get("insight_result") or "").strip()
    hyp = (state.get("hypotheses") or "").strip()
    if insight in _FALLBACK_TEXTS:
        return "insight", "insight가 생성되지 않음(빈 값/실패)", "insight_fallback"
    if hyp in _FALLBACK_TEXTS:
        return "hypothesis", "가설이 생성되지 않음(빈 값/실패)", "hypothesis_fallback"
    if not _completed_analyses(state):
        return "planner", "분석이 하나도 실행되지 않음", "no_completed_analyses"
    if not state.get("statistical_metadata"):
        return "planner", "통계 메타데이터가 비어있음", "missing_statistical_metadata"
    return None


def _classify_deterministic_failure(failure_code: str) -> Dict[str, Any]:
    """failure_code로 재시도 가치를 판단한다: {failure_code, retryable, suggested_action}."""
    retryable = failure_code in _RETRYABLE_FAILURE_CODES
    return {
        "failure_code": failure_code,
        "retryable": retryable,
        "suggested_action": "rerun_eda_agent" if retryable else "manual_review",
    }


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
    stopped_failure = None
    if det:
        target, reason, code = det
        classification = _classify_deterministic_failure(code)
        if not cap_reached and classification["retryable"]:
            # 다시 해볼 가치가 있는 실패(insight/hypothesis fallback)만 내부 재시도.
            verdict = {"status": "retry", "retry_target": target, "reason": reason, "feedback": reason}
        else:
            # 캡 소진 또는 구조적 실패(분석 0건/통계없음) → 헛재시도 없이 즉시 통과+신호.
            # 구조적 실패는 cap_reached가 아니어도 여기로 온다(1회차 얼리스탑).
            stop_reason = "재시도 소진" if classification["retryable"] else "재시도 무의미(구조적 실패)"
            stopped_failure = {"reason": reason, **classification}
            verdict = {"status": "pass", "retry_target": "none",
                       "reason": f"검증 미통과({stop_reason}): {reason}", "feedback": "",
                       **classification}
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
            analysis_facts=_all_analysis_facts(state),
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
                # 캡 소진으로 강제통과 시켜도, 지적됐던 사유(환각 등)를 caution으로 남겨야
                # 분석에이전트/리포트가 "검증 미완료인 채로 넘어왔다"를 알 수 있다(#194 —
                # 이전엔 결정론적 실패만 caution이 남고 이 경로는 조용히 사라졌음).
                original_reason = verdict.get("reason") or "품질 감사 지적 사항 미해결"
                stopped_failure = {"reason": original_reason, "failure_code": "llm_audit_unresolved",
                                   "retryable": False}
                verdict = {"status": "pass", "retry_target": "none",
                           "reason": f"검증 미통과(재시도 소진): {original_reason}", "feedback": "",
                           "failure_code": "llm_audit_unresolved", "retryable": False,
                           "suggested_action": "manual_review"}
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

    # 결정론적 실패로 재시도를 멈췄을 때(캡 소진 또는 구조적 실패 얼리스탑), 신호를 죽이지 않고
    # cautions로 흘려서 supervisor까지 도달하게 한다(cautions → local_checks → validation_errors).
    # failure_code/retryable은 EDAAgent.run()이 retry_hint를 세팅할 때 참고한다.
    if stopped_failure:
        update["cautions"] = list(state.get("cautions", []) or []) + [{
            "code": "EDA_SELF_VALIDATION_FAILED",
            "source": "eda_validator",
            "severity": "high",
            "message_ko": f"EDA 자체 검증 실패: {stopped_failure['reason']}",
            "recommended_action": ["review_eda_before_use"],
            "details": {
                "failure_code": stopped_failure["failure_code"],
                "retryable": stopped_failure["retryable"],
            },
        }]

    return update


def route_after_validator(state: EDAState):
    """검증 결과에 따라 타겟 노드로 되돌리거나, 통과면 chart_selector로."""
    verdict = state.get("validation_result", {}) or {}
    if verdict.get("status") == "retry":
        target = verdict.get("retry_target", "planner")
        if target in _RETRY_TARGETS:
            return target
    return "chart_selector"
