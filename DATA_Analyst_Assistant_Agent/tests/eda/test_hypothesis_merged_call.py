"""hypothesis가 가설과 전달 요약을 한 LLM 호출에서 생성하는지 검증한다."""

from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import hypothesis as hypothesis_mod
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.hypothesis import (
    FINAL_SUMMARY_MARKER, PRIMARY_HYPOTHESIS_MARKER,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return _FakeResponse(self.reply)


@pytest.fixture(autouse=True)
def _context():
    set_context(EdaContext(
        df=pd.DataFrame({"amount": [10, 20, 30], "score": [1, 2, 3]}),
        question_type="relationship",
        measure_cols=["amount", "score"],
    ))
    yield
    reset_context()


def test_hypothesis_and_final_summary_use_one_llm_call(monkeypatch):
    reply = (
        "[가설 1]\n관찰: amount와 score가 함께 변한다.\n"
        f"{PRIMARY_HYPOTHESIS_MARKER}\n"
        '{"target":"score","feature":"amount","method":"Spearman 상관검정"}\n'
        f"{FINAL_SUMMARY_MARKER}\n"
        '{"summary":"amount와 score 관계를 Spearman 상관검정으로 우선 검증한다."}'
    )
    fake = _FakeLLM(reply)
    monkeypatch.setattr(hypothesis_mod, "get_llm", lambda: fake)

    out = hypothesis_mod.hypothesis_node({
        "user_question": "amount와 score 관계를 봐줘",
        "insight_result": "두 변수가 함께 증가할 가능성이 있다.",
        "summary_facts": ["amount와 score가 같은 방향으로 움직였다."],
        "statistical_metadata": {},
    })

    assert len(fake.prompts) == 1
    assert out["primary_hypothesis"]["target"] == "score"
    assert out["final_summary"].startswith("amount와 score 관계")
    assert FINAL_SUMMARY_MARKER not in out["hypotheses"]
