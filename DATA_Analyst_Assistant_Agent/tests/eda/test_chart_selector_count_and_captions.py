"""chart_selector 큐레이션 프롬프트 계약 테스트 — 최소/최대 개수 지시와 3단 캡션
(무엇을 보여주는지·왜 골랐는지·이 차트로 확인 가능한 결론) 요구가 실제로 프롬프트에
들어가는지 확인한다."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.lib import chart_selector_skill as css


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


def test_total_max_and_min_charts_values():
    assert css.TOTAL_MAX == 6
    assert css.MIN_CHARTS == 4


def test_prompt_requests_min_and_max_count(monkeypatch):
    fake = _FakeLLM('{"remove": [], "reason": {}, "keep_captions": {}}')
    monkeypatch.setattr(css, "_load_llm", lambda: fake)

    css._call_llm_remove(
        filenames=["a.png", "b.png"],
        user_question="질문",
        question_type="comparison",
        analysis_results={},
        statistical_metadata={},
    )

    prompt = fake.prompts[0]
    assert "최소 4개" in prompt
    assert "최대 6개" in prompt


def test_prompt_requests_three_part_caption(monkeypatch):
    fake = _FakeLLM('{"remove": [], "reason": {}, "keep_captions": {}}')
    monkeypatch.setattr(css, "_load_llm", lambda: fake)

    css._call_llm_remove(
        filenames=["a.png", "b.png"],
        user_question="질문",
        question_type="comparison",
        analysis_results={},
        statistical_metadata={},
    )

    prompt = fake.prompts[0]
    assert "무엇을 시각화했는지" in prompt
    assert "왜 이 차트를 골랐는지" in prompt
    assert "확인 가능한 구체적 결론" in prompt
    assert "새 숫자 생성 금지" in prompt
