"""hypothesis 노드 — 검증 가능한 가설 3개 + 다음 에이전트용 핸드오프 요약."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import append_errors, get_context, get_llm
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.react import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import handoff_summary_prompt, hypothesis_prompt
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
        if c and c in cols:
            return c
    return ""


def hypothesis_node(state: EDAState) -> dict:
    llm = get_llm()
    target = _resolve_target(state)
    data_level = state.get("data_level", {}) or {}
    low_n_groups = (state.get("statistical_metadata", {}) or {}).get("sample_reliability", {}).get("low_n_groups", [])
    prompt = hypothesis_prompt(
        state["user_question"], state.get("insight_result", ""),
        target_hint=target, data_level=data_level, low_n_groups=low_n_groups)
    fb = state.get("validation_feedback")
    if fb:
        prompt += f"\n[직전 검증 지적 — 반드시 보완하라]\n{fb}\n"
    hypotheses, err1 = run_node_with_retry(
        lambda: llm.invoke(prompt).content.strip(), "hypothesis", fallback="가설 생성 실패"
    )

    # 결정론 게이트(LLM 아님): 불가능한 검정(집계본에 ANOVA/t검정)을 사실 기준으로 교정.
    # 요약 만들기 전에 고쳐야 가설+요약이 일관됨.
    from DATA_Analyst_Assistant_Agent.agents.eda.tools.reliability import correct_hypothesis_feasibility
    hypotheses, _feas_fixes = correct_hypothesis_feasibility(get_context().df, hypotheses, data_level=data_level)

    summary_prompt = handoff_summary_prompt(state.get("insight_result", ""), hypotheses)
    final_summary, err2 = run_node_with_retry(
        lambda: llm.invoke(summary_prompt).content.strip(), "final_summary", fallback="요약 생성 실패"
    )
    return {
        "hypotheses": hypotheses,
        "final_summary": final_summary,
        "analysis_target": target,
        "error_log": append_errors(state, err1, err2),
    }
