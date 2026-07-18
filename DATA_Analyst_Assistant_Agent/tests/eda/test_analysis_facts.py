"""#194 후속 — 분석노드 6개의 facts 분리 검증.

원문(quality_result 등)은 그대로 보존하면서, ANALYSIS_FACTS_MARKER 뒤 JSON을 별도
*_facts 필드로 분리하는지, 정상 문장은 절대 자르지 않고 개수(4개)·비상길이(1000자)만
캡하는지, 마커 누락/malformed JSON은 원문 첫 문장 폴백으로 우아하게 처리하는지 검증한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import tool_runner as TR
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import quality
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.analysis import ANALYSIS_FACTS_MARKER


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return _FakeResponse(self.reply)


@pytest.fixture(autouse=True)
def _context():
    df = pd.DataFrame({
        "amount": [10, 20, 15, 30, 25, 12, 18, 22, 27, 31, 14, 19, 21, 26, 29],
        "category": list("ABABABABABABABA"),
    })
    set_context(EdaContext(df=df, question_type="comparison", user_question="테스트"))
    yield
    reset_context()


def _state(**overrides) -> dict:
    base = {"user_question": "테스트 질문", "inspect_result": "", "analysis_plan": {}}
    base.update(overrides)
    return base


def test_facts_split_from_marker_and_prose_preserved(monkeypatch):
    reply = (
        "서술 원문입니다. 여러 문장이 있습니다.\n"
        f"{ANALYSIS_FACTS_MARKER}\n"
        '["결측치가 5% 존재한다.", "중복 행은 없다."]'
    )
    fake = _FakeLLM(reply)
    monkeypatch.setattr(TR, "get_llm", lambda: fake)

    out = quality.quality_node(_state())

    assert fake.calls == 1  # facts 분리 때문에 LLM이 추가로 불리진 않는다
    assert out["quality_result"] == "서술 원문입니다. 여러 문장이 있습니다."
    assert out["quality_facts"] == ["결측치가 5% 존재한다.", "중복 행은 없다."]


def test_facts_count_capped_at_four_without_truncating_content(monkeypatch):
    items = [f"사실 {i}." for i in range(7)]
    reply = f'원문\n{ANALYSIS_FACTS_MARKER}\n{items}'.replace("'", '"')
    fake = _FakeLLM(reply)
    monkeypatch.setattr(TR, "get_llm", lambda: fake)

    out = quality.quality_node(_state())

    assert len(out["quality_facts"]) == 4
    assert out["quality_facts"] == ["사실 0.", "사실 1.", "사실 2.", "사실 3."]  # 앞 4개, 내용 안 잘림


def test_marker_missing_falls_back_to_first_sentences_not_char_truncation(monkeypatch):
    reply = "첫 문장이다. 두번째 문장이다. 세번째 문장이다."
    fake = _FakeLLM(reply)
    monkeypatch.setattr(TR, "get_llm", lambda: fake)

    out = quality.quality_node(_state())

    assert out["quality_facts"] == ["첫 문장이다.", "두번째 문장이다."]
    assert out["quality_result"] == reply  # 원문은 그대로 보존


def test_emergency_length_drops_whole_batch_and_falls_back(monkeypatch):
    huge = "가" * 1500
    reply = (
        "원문 첫 문장이다. 원문 둘째 문장이다.\n"
        f"{ANALYSIS_FACTS_MARKER}\n"
        f'["정상 사실.", "{huge}"]'
    )
    fake = _FakeLLM(reply)
    monkeypatch.setattr(TR, "get_llm", lambda: fake)

    out = quality.quality_node(_state())

    # 비정상(1000자 초과) 항목이 하나라도 있으면 항목만 빼지 않고 그 노드 facts 전체를 버린다
    assert out["quality_facts"] == ["원문 첫 문장이다.", "원문 둘째 문장이다."]


def test_no_dataframe_returns_empty_facts_without_llm_call(monkeypatch):
    fake = _FakeLLM("안 불려야 함")
    monkeypatch.setattr(TR, "get_llm", lambda: fake)
    set_context(EdaContext(df=None))

    out = quality.quality_node(_state())

    assert fake.calls == 0
    assert out["quality_facts"] == []
