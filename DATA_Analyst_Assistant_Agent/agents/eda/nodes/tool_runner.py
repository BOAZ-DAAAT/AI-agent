"""Mini-ReAct 루프 엔진 + 재시도 헬퍼.

분석 노드들이 공유하는 mini-ReAct 루프(run_mini_react)와 재시도 래퍼
(run_node_with_retry / run_mini_react_with_retry)를 담는다.
LLM에 노출되는 @tool 정의·툴 그룹은 eda/tools.py 로 분리되어 있다.

원본은 모듈 전역 `_df / _key_col ...` 를 직접 참조했으나, 여기서는
`_runtime.get_context()` 로 동일 정보를 읽는다(동작 동일).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import MAX_NODE_RETRIES, get_llm
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import react_fix_prompt


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
