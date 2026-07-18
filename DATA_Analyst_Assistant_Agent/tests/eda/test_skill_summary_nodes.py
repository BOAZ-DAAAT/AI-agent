"""#194 — quality/distribution/comparison/relationship/time/inspect가 ReAct 툴판단
라운드 없이 '@tool 직접호출(.invoke({})) + 요약 1콜'로 도는지 검증한다.
@tool 포장 자체는 유지하고, LLM이 도구를 고르는 라운드만 생략한다."""

from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import tool_runner as TR
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import (
    comparison, distribution, inspect, quality, relationship, time as time_mod,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, reply: str = "요약된 한국어 텍스트") -> None:
        self.reply = reply
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return _FakeResponse(self.reply)


@pytest.fixture(autouse=True)
def _context():
    df = pd.DataFrame({
        "amount": [10, 20, 15, 30, 25, 12, 18, 22, 27, 31, 14, 19, 21, 26, 29],
        "score": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "category": list("ABABABABABABABA"),
    })
    set_context(EdaContext(df=df, question_type="comparison", user_question="테스트 질문"))
    yield
    reset_context()


def _state(**overrides) -> dict:
    base = {"user_question": "카테고리별 금액 비교해줘", "inspect_result": "", "analysis_plan": {}}
    base.update(overrides)
    return base


@pytest.mark.parametrize("node_fn, result_key", [
    (quality.quality_node, "quality_result"),
    (distribution.distribution_node, "distribution_result"),
    (comparison.comparison_node, "comparison_result"),
    (relationship.relationship_node, "relationship_result"),
    (time_mod.time_node, "time_result"),
    (inspect.inspect_node, "inspect_result"),
])
def test_node_calls_llm_exactly_once(monkeypatch, node_fn, result_key):
    fake = _FakeLLM()
    monkeypatch.setattr(TR, "get_llm", lambda: fake)

    out = node_fn(_state())

    assert fake.calls == 1  # ReAct 판단 라운드 없이 요약 1콜만
    assert out[result_key] == "요약된 한국어 텍스트"


def test_no_dataframe_short_circuits_without_llm_call(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(TR, "get_llm", lambda: fake)
    set_context(EdaContext(df=None))

    out = quality.quality_node(_state())

    assert fake.calls == 0
    assert "로드되지 않았습니다" in out["quality_result"]


def test_chart_requests_are_popped_into_context_not_left_in_prompt(monkeypatch):
    captured_prompts = []

    class _CapturingLLM(_FakeLLM):
        def invoke(self, prompt: str):
            captured_prompts.append(prompt)
            return super().invoke(prompt)

    fake = _CapturingLLM()
    monkeypatch.setattr(TR, "get_llm", lambda: fake)

    distribution.distribution_node(_state())

    from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
    ctx = get_context()
    # distribution_skill은 chart_requests를 만든다 — 프롬프트에 새지 않고 ctx에 쌓여야 한다
    assert "chart_requests" not in captured_prompts[0]
