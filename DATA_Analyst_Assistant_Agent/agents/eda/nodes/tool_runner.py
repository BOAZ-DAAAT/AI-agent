"""노드 실행 재시도 헬퍼 + @tool 직접호출 실행기.

분석 노드들이 공유하는 재시도 래퍼(run_node_with_retry)와, @tool로 감싼 스킬을 LLM
판단 없이 직접 호출한 뒤 결과를 요약 LLM 1콜로 감싸는 run_tool_with_summary를 담는다(#194).

원본은 모듈 전역 `_df / _key_col ...` 를 직접 참조했으나, 여기서는
`_runtime.get_context()` 로 동일 정보를 읽는다(동작 동일).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import MAX_NODE_RETRIES, get_llm, split_marked_json
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.analysis import ANALYSIS_FACTS_MARKER

_FACT_MAX_COUNT = 4
_FACT_EMERGENCY_MAX_LEN = 1000  # 압축 기준 아님 — LLM이 지시를 완전히 무시한 비정상 출력 감지용
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[다요음임]\.)\s+')


# ─────────────────────────────
# 재시도 헬퍼
# ─────────────────────────────
def run_node_with_retry(fn, node_name: str, fallback="분석 스킵 (오류로 인해 생략됨)"):
    """노드 실행 함수를 감싸 에러 시 재시도하고, 모두 실패하면 fallback을 반환한다.
    반환값: (result, error_message or None)"""
    last_error = None
    for _ in range(MAX_NODE_RETRIES + 1):
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


def _first_sentences(text: str, n: int = 2) -> list[str]:
    """facts 마커가 없거나 파싱에 실패했을 때의 폴백 — 문장 경계(다./요./음./임.)에서만
    끊어 앞 n개를 반환한다. 글자 수 절단과 달리 문장 중간에서 자르지 않는다."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts[:n] if p.strip()]


def _normalize_analysis_facts(parsed: Any, prose_fallback: str) -> list[str]:
    """facts JSON을 정규화한다. 정상 문장은 절대 안 자른다 — 개수만 최대 4개로 캡.

    _FACT_EMERGENCY_MAX_LEN(1000자)은 압축 기준이 아니라, LLM이 지시를 완전히 무시하고
    거대한 텍스트를 배열 항목 하나에 욱여넣은 것 같은 명백히 비정상적인 출력을 감지하는
    최후 방어선이다 — 그런 경우 항목 하나만 조용히 버리지 않고 그 노드의 facts 전체를
    버려 원문 첫 문장 폴백으로 넘긴다.
    """
    if not isinstance(parsed, list):
        return _first_sentences(prose_fallback, 2)
    facts = [" ".join(str(item).split()) for item in parsed if str(item).strip()][:_FACT_MAX_COUNT]
    if not facts or any(len(f) > _FACT_EMERGENCY_MAX_LEN for f in facts):
        return _first_sentences(prose_fallback, 2)
    return facts


def run_tool_with_summary_and_facts(
    tool: Any,
    prompt_builder: Callable[[str], str],
    node_name: str,
    fallback: str = "분석 스킵 (오류로 인해 생략됨)",
) -> tuple[str, list[str], str | None]:
    """run_tool_with_summary에 facts 분리를 더한 버전(#194 후속).

    LLM 응답을 서술(prose)과 ANALYSIS_FACTS_MARKER 뒤 JSON 배열(facts)로 나눈다.
    원문(prose)은 그대로 result에 담아 기존 state 필드·다른 EDA 내부 노드가 계속 쓸 수
    있게 보존하고, facts는 insight 입력을 줄이는 용도로 별도 필드에 담긴다 — 원문을
    facts로 대체하는 게 아니라 병행한다.
    """
    raw_result, err = run_tool_with_summary(tool, prompt_builder, node_name, fallback)
    prose, parsed = split_marked_json(raw_result, ANALYSIS_FACTS_MARKER)
    facts = _normalize_analysis_facts(parsed, prose)
    return prose, facts, err
