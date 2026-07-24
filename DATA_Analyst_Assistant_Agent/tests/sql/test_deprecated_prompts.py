"""미사용 SQL 프롬프트의 deprecated 호환 계약 테스트."""

from __future__ import annotations

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.finalize import (
    finalize_answer_prompt,
    finalize_mart_prompt,
    finalize_rewrite_prompt,
)
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.validate import validate_prompt


DEPRECATED_PACKAGE_EXPORTS = {
    "validate_prompt",
    "finalize_mart_prompt",
    "finalize_answer_prompt",
    "finalize_rewrite_prompt",
}


def prompt_state() -> dict:
    return {
        "user_question": "주문 수를 보여줘",
        "plan": {"route_kind": "simple"},
        "mart_design": {},
        "integrity_text": "{}",
        "sql_draft": {"sql": "SELECT COUNT(*) FROM orders", "reasoning": "주문 수 집계"},
        "mart_validation_signal": {},
        "precheck_result": None,
        "sql_result": [(1,)],
        "postcheck_result": None,
        "row_count": 1,
    }


def test_deprecated_prompts_are_not_package_exports():
    assert DEPRECATED_PACKAGE_EXPORTS.isdisjoint(prompts.__all__)
    for symbol in DEPRECATED_PACKAGE_EXPORTS:
        assert not hasattr(prompts, symbol)


@pytest.mark.parametrize(
    ("builder", "args", "expected", "replacement"),
    [
        (validate_prompt, (), "SQL/데이터마트 검증기", "validate_sql_and_result"),
        (finalize_mart_prompt, (), "결과 요약기", "finalize_answer"),
        (finalize_answer_prompt, (), "데이터 분석 답변 작성기", "finalize_answer"),
        (finalize_rewrite_prompt, ("초안",), "초안 답변", "finalize_answer"),
    ],
)
def test_direct_module_prompt_calls_warn_and_keep_string_result(
    builder,
    args,
    expected,
    replacement,
):
    with pytest.warns(DeprecationWarning, match=replacement):
        result = builder(prompt_state(), *args)

    assert isinstance(result, str)
    assert expected in result
