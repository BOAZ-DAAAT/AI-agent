"""planner_node 배치실행(#194) 계약 테스트 — 1차 세트선택 + 큐 소비(LLM 호출 없음)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import planner as planner_mod


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return _FakeResponse(self.replies.pop(0))


@pytest.fixture(autouse=True)
def _context():
    # numeric 2개(amount/score) + categorical 1개(category), time 없음, 15행
    # → feasible = {quality, distribution, comparison, relationship, clustering} (time만 제외)
    df = pd.DataFrame({
        "amount": [10, 20, 15, 30, 25, 12, 18, 22, 27, 31, 14, 19, 21, 26, 29],
        "score": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "category": list("ABABABABABABABA"),
    })
    set_context(EdaContext(df=df, question_type="comparison"))
    yield
    reset_context()


def _state(**overrides) -> dict:
    base = {"user_question": "카테고리별 금액 비교해줘", "controller_log": [], "round": 0}
    base.update(overrides)
    return base


def test_round_zero_asks_for_batch_and_queues_rest(monkeypatch):
    fake_llm = _FakeLLM([json.dumps({
        "next_batch": ["comparison", "distribution"],
        "reason": "질문이 그룹비교라 comparison 먼저, 분포도 참고",
    })])
    monkeypatch.setattr(planner_mod, "get_llm", lambda: fake_llm)

    result = planner_mod.planner_node(_state())

    assert result["next_analysis"] == "comparison"
    assert result["analysis_queue"] == ["distribution"]
    assert result["round"] == 1
    assert len(fake_llm.prompts) == 1
    assert "next_batch" in fake_llm.prompts[0]  # emit_batch=True 프롬프트가 나갔는지


def test_queue_pop_does_not_call_llm(monkeypatch):
    fake_llm = _FakeLLM([])  # replies 비어있음 → 호출되면 즉시 예외로 실패
    monkeypatch.setattr(planner_mod, "get_llm", lambda: fake_llm)

    state = _state(
        round=1,
        analysis_queue=["distribution"],
        controller_log=[{"round": 0, "choice": "comparison", "reason": "..."}],
    )
    result = planner_mod.planner_node(state)

    assert result["next_analysis"] == "distribution"
    assert result["analysis_queue"] == []
    assert result["round"] == 2
    assert fake_llm.prompts == []  # LLM 호출 자체가 없었음


def test_empty_batch_falls_back_to_single_choice_codegen(monkeypatch):
    fake_llm = _FakeLLM([json.dumps({
        "next_batch": [],
        "next": "codegen",
        "reason": "파생계산이 필요함",
    })])
    monkeypatch.setattr(planner_mod, "get_llm", lambda: fake_llm)

    result = planner_mod.planner_node(_state())

    assert result["next_analysis"] == "codegen"
    assert result["analysis_queue"] == []
    assert "검정 실행이나 p-value 계산을 위해 codegen을 고르지 말고" in fake_llm.prompts[0]


def test_batch_filters_invalid_items_and_caps_at_max_analyses(monkeypatch):
    fake_llm = _FakeLLM([json.dumps({
        "next_batch": ["comparison", "not_a_real_tool", "distribution", "relationship",
                       "clustering", "quality", "time"],  # 존재하지 않는 카드 + feasible 아닌 time 섞임
        "reason": "다 보고 싶음",
    })])
    monkeypatch.setattr(planner_mod, "get_llm", lambda: fake_llm)

    result = planner_mod.planner_node(_state())

    total_planned = [result["next_analysis"]] + result["analysis_queue"]
    assert "not_a_real_tool" not in total_planned
    assert "time" not in total_planned  # time 컬럼이 없어 feasible 아님
    assert len(total_planned) <= planner_mod.MAX_ANALYSES


def test_declines_to_reselect_codegen_after_already_attempted(monkeypatch):
    # codegen이 이미 한 번(0라운드) 시도됐는데 LLM이 또 codegen을 고르면 무효 처리하고 done으로 종료
    # (route_after_codegen이 planner로 되돌릴 때 왕복 루프가 생기지 않게 하는 가드, #194)
    fake_llm = _FakeLLM([json.dumps({"next": "codegen", "reason": "그래도 codegen이 나을 것 같음"})])
    monkeypatch.setattr(planner_mod, "get_llm", lambda: fake_llm)

    state = _state(
        round=1,
        controller_log=[{"round": 0, "choice": "codegen", "reason": "..."}],
    )
    result = planner_mod.planner_node(state)

    assert result["next_analysis"] == "done"
    assert "유효하지 않은 선택" in result["controller_log"][-1]["reason"]


def test_non_first_round_does_not_emit_batch(monkeypatch):
    fake_llm = _FakeLLM([json.dumps({"next": "distribution", "reason": "재검토"})])
    monkeypatch.setattr(planner_mod, "get_llm", lambda: fake_llm)

    state = _state(
        round=2,
        controller_log=[{"round": 0, "choice": "comparison", "reason": "..."},
                        {"round": 1, "choice": "quality", "reason": "..."}],
        validation_feedback="비교가 부족함",
    )
    result = planner_mod.planner_node(state)

    assert result["next_analysis"] == "distribution"
    assert "next_batch" not in fake_llm.prompts[0]
