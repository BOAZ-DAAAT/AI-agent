"""Mini-ReAct 헬퍼 + skill 단위 툴.

원본 `eda_agent/eda_agent.py` 의 @tool 정의, 툴 그룹, run_mini_react,
run_mini_react_with_retry, run_node_with_retry 를 분리한 것.

원본은 모듈 전역 `_df / _key_col ...` 를 직접 참조했으나, 여기서는
`_runtime.get_context()` 로 동일 정보를 읽는다(동작 동일).
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import (
    MAX_NODE_RETRIES,
    accumulate_chart_requests,
    get_context,
    get_llm,
)
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import react_fix_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.comparison_skill import run_comparison_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.data_quality_skill import run_data_quality_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.distribution_skill import run_distribution_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.relationship_skill import run_relationship_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.skills.time_skill import run_time_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.profiling import get_basic_profile


# ─────────────────────────────
# Tools
# ─────────────────────────────
def _emit_and_dump(ctx, result) -> str:
    """skill 결과에서 차트 주문서(chart_requests)를 분리해 ctx에 누적하고,
    나머지만 JSON으로 반환한다(주문서는 LLM에 노출하지 않아 토큰 낭비 없음)."""
    if isinstance(result, dict):
        accumulate_chart_requests(ctx, result.pop("chart_requests", []))
    return json.dumps(result, ensure_ascii=False)


@tool
def profile_data() -> str:
    """데이터 기본 구조(shape, 컬럼 타입, 카디널리티, 시간 컬럼 여부, 기초통계)를 반환한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    df = ctx.df
    result = get_basic_profile(df)
    cardinality = {col: int(df[col].nunique()) for col in df.columns}
    time_cols = [c for c in df.columns if "datetime" in str(df[c].dtype) or "date" in c.lower()]
    return json.dumps({
        "shape": result["shape"],
        "dtypes": result["dtypes"],
        "cardinality": cardinality,
        "time_columns": time_cols,
        "describe": {c: {str(k): str(v) for k, v in s.items()} for c, s in result["describe"].items()},
    }, ensure_ascii=False)


@tool
def run_quality() -> str:
    """data_quality_skill: 결측치 / 이상치 / 중복 / 표본 신뢰도를 한 번에 점검한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(
        run_data_quality_skill(ctx.df, key_col=ctx.key_col, measure_cols=ctx.measure_cols, count_col=ctx.count_col),
        ensure_ascii=False,
    )


@tool
def run_distribution() -> str:
    """distribution_skill: 히스토그램 / 박스플롯 / 범주형 빈도 분포를 한 번에 분석한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_distribution_skill(
        ctx.df, measure_cols=ctx.measure_cols, question_type=ctx.question_type,
        priority_metrics=ctx.priority_metrics, key_col=ctx.key_col, target_col=ctx.target_col))


@tool
def run_comparison() -> str:
    """comparison_skill: 카테고리별 상위/하위 barplot + 히트맵 비교를 한 번에 수행한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_comparison_skill(
        ctx.df, key_col=ctx.key_col, measure_cols=ctx.measure_cols, question_type=ctx.question_type))


@tool
def run_relationship() -> str:
    """relationship_skill: 상관관계 히트맵 + scatter plot을 한 번에 분석한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_relationship_skill(
        ctx.df, measure_cols=ctx.measure_cols, question_type=ctx.question_type, target_col=ctx.target_col))


@tool
def run_time() -> str:
    """time_skill: 시계열 추세 + 시즌성 분석을 한 번에 수행한다."""
    ctx = get_context()
    if ctx.df is None:
        return "데이터가 로드되지 않았습니다."
    return _emit_and_dump(ctx, run_time_skill(
        ctx.df, measure_cols=ctx.measure_cols, time_cols=ctx.time_cols, key_col=ctx.key_col))


# ─────────────────────────────
# 툴 그룹
# ─────────────────────────────
INSPECT_TOOLS      = [profile_data]
QUALITY_TOOLS      = [run_quality]
DISTRIBUTION_TOOLS = [run_distribution]
COMPARISON_TOOLS   = [run_comparison]
RELATIONSHIP_TOOLS = [run_relationship]
TIME_TOOLS         = [run_time]


# ─────────────────────────────
# Mini-ReAct 루프
# ─────────────────────────────
def run_mini_react(tools_list: list, system_prompt: str, max_iter: int = 8) -> str:
    """주어진 툴 목록 안에서만 LLM이 선택·호출하는 mini-ReAct 루프."""
    tools_dict = {t.name: t for t in tools_list}
    node_llm = get_llm().bind_tools(tools_list)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="분석을 시작하라."),
    ]

    for _ in range(max_iter):
        response = node_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            fn = tools_dict.get(tc["name"])
            result = fn.invoke(tc["args"]) if fn else "툴을 찾을 수 없습니다."
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages[-1].content


# ─────────────────────────────
# 재시도 헬퍼
# ─────────────────────────────
def run_node_with_retry(fn, node_name: str, fallback="분석 스킵 (오류로 인해 생략됨)", max_retries: int = MAX_NODE_RETRIES):
    """노드 실행 함수를 감싸 에러 시 재시도하고, 모두 실패하면 fallback을 반환한다.
    반환값: (result, error_message or None)"""
    last_error = None
    for _ in range(max_retries + 1):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
    return fallback, f"[{node_name}] {last_error}"


def run_mini_react_with_retry(
    tools_list: list,
    system_prompt: str,
    node_name: str,
    fallback: str = "분석 스킵 (오류로 인해 생략됨)",
    max_retries: int = MAX_NODE_RETRIES,
    max_iter: int = 8,
) -> tuple:
    """run_mini_react 실행 중 에러 발생 시 에러 내용을 LLM에게 피드백으로 넘겨 재시도한다.
    반환값: (result, error_message or None)"""
    llm_feedback = get_llm()
    last_error = None
    current_prompt = system_prompt

    for attempt in range(max_retries + 1):
        try:
            return run_mini_react(tools_list, current_prompt, max_iter=max_iter), None
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if attempt < max_retries:
                fix = react_fix_prompt(node_name, last_error, current_prompt)
                current_prompt = llm_feedback.invoke(fix).content.strip()

    return fallback, f"[{node_name}] {last_error}"
