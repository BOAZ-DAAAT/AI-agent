"""hypothesis 노드 — 검증 가능한 가설 3개 + 다음 에이전트용 핸드오프 요약."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import (
    append_errors, get_context, get_llm, split_marked_json,
)
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import hypothesis_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.hypothesis import (
    FINAL_SUMMARY_MARKER, PRIMARY_HYPOTHESIS_MARKER,
)
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState


def _resolve_target(state: EDAState) -> str:
    """가설 6유형의 앵커가 될 target 컬럼을 확정한다(LLM 없음).
    우선순위: 앞단 plan_metric → planner의 priority_metrics → measure_cols.
    실제 df 컬럼인 것만 채택하고, 하나도 못 찾으면 ""(가설 노드가 인사이트로 추론)."""
    ctx = get_context()
    df = getattr(ctx, "df", None)
    cols = set(df.columns) if df is not None else set()

    candidates = []
    if state.get("plan_metric"):
        candidates.append(state["plan_metric"])
    candidates += list(state.get("analysis_plan", {}).get("priority_metrics", []) or [])
    candidates += list(getattr(ctx, "measure_cols", None) or [])

    for c in candidates:
        # priority_metrics는 {"metric": ...} dict, plan/measure는 문자열 — 둘 다 컬럼명으로 정규화
        name = c.get("metric") if isinstance(c, dict) else c
        if name and name in cols:
            return name
    return ""


def hypothesis_node(state: EDAState) -> dict:
    llm = get_llm()
    target = _resolve_target(state)
    data_level = state.get("data_level", {}) or {}
    low_n_groups = (state.get("statistical_metadata", {}) or {}).get("sample_reliability", {}).get("low_n_groups", [])
    summary_facts = state.get("summary_facts", []) or []
    if not summary_facts:
        summary_facts = [(state.get("insight_result", "") or "")[:800]]
    prompt = hypothesis_prompt(
        state["user_question"], state.get("insight_result", ""),
        target_hint=target, data_level=data_level, low_n_groups=low_n_groups,
        summary_facts=summary_facts)
    fb = state.get("validation_feedback")
    if fb:
        prompt += f"\n[직전 검증 지적 — 반드시 보완하라]\n{fb}\n"
    raw_result, err = run_node_with_retry(
        lambda: llm.invoke(prompt).content.strip(), "hypothesis", fallback="가설 생성 실패"
    )

    # 마지막 마커부터 떼어내면 가설 프로즈, 1순위 가설, 전달 요약을 한 응답에서 안전하게 분리할 수 있다.
    hypotheses_with_primary, summary_obj = split_marked_json(raw_result, FINAL_SUMMARY_MARKER)
    hypotheses, primary_obj = split_marked_json(hypotheses_with_primary, PRIMARY_HYPOTHESIS_MARKER)
    primary_hypothesis = primary_obj if isinstance(primary_obj, dict) else {}
    final_summary = str(summary_obj.get("summary", "")).strip() if isinstance(summary_obj, dict) else ""
    if not final_summary:
        final_summary = " ".join(str(f).strip() for f in summary_facts[:3] if str(f).strip()) or "요약 생성 실패"

    # 결정론 게이트(LLM 아님): 불가능한 검정(집계본에 ANOVA/t검정)을 사실 기준으로 교정.
    # 요약 만들기 전에 고쳐야 가설+요약이 일관됨.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import correct_hypothesis_feasibility
    hypotheses, _feas_fixes = correct_hypothesis_feasibility(get_context().df, hypotheses, data_level=data_level)

    # 사후 재검증(LLM 아님): 이미 계산된 통계 숫자(correlation_pairs·clustering)로 가설 강도를
    # 태그하고, 강한 순 재정렬 + 명백 무상관만 드롭(최소 1개 보존). 요약 전에 실행해 일관되게.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.hypothesis_screening import screen_hypotheses
    hypotheses, hypothesis_signals = screen_hypotheses(hypotheses, state.get("statistical_metadata", {}) or {})

    return {
        "hypotheses": hypotheses,
        "hypothesis_signals": hypothesis_signals,
        "primary_hypothesis": primary_hypothesis,
        "final_summary": final_summary,
        "analysis_target": target,
        "error_log": append_errors(state, err),
    }
