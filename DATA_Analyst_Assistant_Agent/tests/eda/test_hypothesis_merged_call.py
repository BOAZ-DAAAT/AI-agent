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


def test_final_summary_and_primary_hypothesis_sync_with_feasibility_correction(monkeypatch):
    """run-019f7437 실사례 재현 — 그룹당 1행인 feature에 Mann-Whitney를 추천하면
    hypotheses뿐 아니라 final_summary·primary_hypothesis도 교정 신호를 반영해야 한다.
    (전엔 hypotheses만 고쳐지고 final_summary는 원래 값 그대로 analysis_agent에게 전달됐다.)"""
    reply = (
        "[가설 1]\n관찰: 세그먼트별 고객 수 차이가 있다.\n유형: 그룹차이\n"
        "H0: 세그먼트 간 차이가 없다.\nH1: 세그먼트 간 차이가 있다.\n"
        "검증방법: Mann-Whitney U 검정\n필요변수: target=customer_count, feature=segment\n"
        "현재데이터: 현재 마트로 검증 가능하다\n"
        f"{PRIMARY_HYPOTHESIS_MARKER}\n"
        '{"target":"customer_count","feature":"segment","method":"Mann-Whitney U 검정"}\n'
        f"{FINAL_SUMMARY_MARKER}\n"
        '{"summary":"세그먼트 간 고객 수 차이를 Mann-Whitney U 검정으로 우선 검증한다."}'
    )
    fake = _FakeLLM(reply)
    monkeypatch.setattr(hypothesis_mod, "get_llm", lambda: fake)

    set_context(EdaContext(
        df=pd.DataFrame({
            "segment": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "customer_count": [93099, 2997, 100, 200, 300, 400],
        }),
        question_type="comparison",
    ))

    out = hypothesis_mod.hypothesis_node({
        "user_question": "세그먼트별 고객 수 차이가 유의한지 검정해줘",
        "insight_result": "세그먼트 간 고객 수 차이가 관찰된다.",
        "summary_facts": ["세그먼트별 고객 수가 다르다."],
        "statistical_metadata": {},
        "data_level": {"level": "aggregated"},
    })

    assert "추가 필요" in out["hypotheses"]
    assert "[주의:" in out["final_summary"]
    assert "Mann-Whitney" in out["final_summary"]
    assert out["primary_hypothesis"]["feasible"] is False


def test_feasible_hypothesis_leaves_final_summary_and_primary_untouched(monkeypatch):
    reply = (
        "[가설 1]\n관찰: amount와 score가 함께 변한다.\n유형: 관계추론\n"
        "H0: 상관이 없다.\nH1: 상관이 있다.\n검증방법: 스피어만 상관검정\n"
        "필요변수: target=score, feature=amount\n현재데이터: 현재 마트로 검증 가능하다\n"
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

    assert "[주의:" not in out["final_summary"]
    assert "feasible" not in out["primary_hypothesis"]
