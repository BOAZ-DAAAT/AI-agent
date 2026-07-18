"""노드 실행 재시도 헬퍼 + @tool 직접호출 실행기.

분석 노드들이 공유하는 재시도 래퍼(run_node_with_retry)와, @tool로 감싼 스킬을 LLM
판단 없이 직접 호출한 뒤 결과를 요약 LLM 1콜로 감싸는 run_tool_with_summary를 담는다(#194).

원본은 모듈 전역 `_df / _key_col ...` 를 직접 참조했으나, 여기서는
`_runtime.get_context()` 로 동일 정보를 읽는다(동작 동일).
"""

from __future__ import annotations

from typing import Any, Callable

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import MAX_NODE_RETRIES, get_llm


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


def run_tool_with_summary(
    tool: Any,
    prompt_builder: Callable[[str], str],
    node_name: str,
    fallback: str = "분석 스킵 (오류로 인해 생략됨)",
) -> tuple[str, str | None]:
    """@tool로 감싼 스킬을 LLM 판단 없이 직접 호출(.invoke({}))하고, 결과를 요약 LLM 1콜로 감싼다(#194).

    quality/distribution/comparison/relationship/time/inspect는 노드당 도구가 항상 1개뿐이라
    '이 도구를 부를지' LLM이 고르는 mini-ReAct 판단 라운드엔 실질적 선택지가 없었다 — 그
    판단 라운드만 생략한다. @tool 포장 자체(LangChain Tool 스키마, LangSmith에서 이름 붙은
    개체로 추적되는 것 등)는 그대로 유지한다. chart_requests 분리·ctx 누적은 @tool 내부
    (_emit_and_dump)가 이미 처리하므로 여기서 따로 다루지 않는다.
    """
    result_json = tool.invoke({})
    prompt = prompt_builder(result_json)
    return run_node_with_retry(
        lambda: get_llm().invoke(prompt).content.strip(), node_name, fallback=fallback
    )
