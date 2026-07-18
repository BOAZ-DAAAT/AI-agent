"""#194 — insight_node가 서술과 facts만 한 호출에서 생성하는지 검증한다."""

from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import insight as insight_mod
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.insight import SUMMARY_FACTS_MARKER


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.calls += 1
        self.prompts.append(prompt)
        return _FakeResponse(self.reply)


@pytest.fixture(autouse=True)
def _context():
    df = pd.DataFrame({
        "amount": [10, 20, 15, 30, 25, 12, 18, 22, 27, 31, 14, 19, 21, 26, 29],
        "score": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "category": list("ABABABABABABABA"),
    })
    set_context(EdaContext(df=df, question_type="comparison"))
    yield
    reset_context()


def _state(**overrides) -> dict:
    base = {"user_question": "카테고리별 금액 비교해줘"}
    base.update(overrides)
    return base


def _reply_with_markers(facts_json: str) -> str:
    return (
        "[핵심 패턴]\n1) 관찰 ...\n\n[구조 해석]\n요약 ...\n\n[해석 주의사항]\n주의 ...\n"
        f"\n{SUMMARY_FACTS_MARKER}\n{facts_json}"
    )


def test_insight_node_calls_llm_exactly_once(monkeypatch):
    reply = _reply_with_markers('["사실1", "사실2"]')
    fake = _FakeLLM(reply)
    monkeypatch.setattr(insight_mod, "get_llm", lambda: fake)

    out = insight_mod.insight_node(_state())

    assert fake.calls == 1  # 예전엔 cautions추론(1) + 메인서술(1) = 2콜


def test_summary_facts_and_prose_are_split_cleanly(monkeypatch):
    reply = _reply_with_markers('["총 15행", "상관 약함"]')
    fake = _FakeLLM(reply)
    monkeypatch.setattr(insight_mod, "get_llm", lambda: fake)

    out = insight_mod.insight_node(_state())

    assert out["summary_facts"] == ["총 15행", "상관 약함"]
    assert SUMMARY_FACTS_MARKER not in out["insight_result"]
    assert "[핵심 패턴]" in out["insight_result"]


def test_prompt_does_not_request_llm_inferred_cautions(monkeypatch):
    reply = _reply_with_markers("[]")
    fake = _FakeLLM(reply)
    monkeypatch.setattr(insight_mod, "get_llm", lambda: fake)

    out = insight_mod.insight_node(_state())

    assert "===LLM_CAUTIONS===" not in fake.prompts[0]
    assert all(c.get("source") != "llm_inferred" for c in out["cautions"])


def test_malformed_facts_json_falls_back_gracefully(monkeypatch):
    reply = _reply_with_markers("이건 JSON이 아님")
    fake = _FakeLLM(reply)
    monkeypatch.setattr(insight_mod, "get_llm", lambda: fake)

    out = insight_mod.insight_node(_state())

    assert out["summary_facts"] == []


def test_missing_markers_keep_full_prose_and_empty_structured_fields(monkeypatch):
    fake = _FakeLLM("마커 없는 순수 서술 텍스트")
    monkeypatch.setattr(insight_mod, "get_llm", lambda: fake)

    out = insight_mod.insight_node(_state())

    assert out["insight_result"] == "마커 없는 순수 서술 텍스트"
    assert out["summary_facts"] == []
