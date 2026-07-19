"""가설 유형 확장(6종 → 13종) 계약 테스트 — analysis_agent가 codegen/vetted primitive로
커버하는 특화 기법(LTV/생존분석/지역분석/마케팅믹스/자원배분)과 코호트/퍼널이 '유형:' 라인으로
정확히 인식되고, 사후 재검증(screen_hypotheses)에서 안전하게(미측정으로) 처리되는지 확인한다.
"""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.lib.hypothesis_screening import (
    _TYPES,
    screen_hypotheses,
)


def _block(hyp_type: str, method: str) -> str:
    return f"""[가설 1]
관찰: 어떤 관찰
유형: {hyp_type}
H0: 귀무가설
H1: 대립가설
검증방법: {method}
필요변수: target=x, feature=y
현재데이터: 현재 마트로 검증 가능하다"""


def test_all_13_types_are_registered():
    assert _TYPES == (
        "회귀", "분류", "관계추론", "그룹차이", "군집", "시계열",
        "생애가치", "생존분석", "지역분석", "마케팅믹스", "자원배분", "코호트", "퍼널",
    )


def test_new_types_parsed_correctly_and_marked_unmeasured():
    new_types = [
        ("생애가치", "BG/NBD+Gamma-Gamma 기반 LTV 추정"),
        ("생존분석", "Kaplan-Meier 생존곡선"),
        ("지역분석", "공간자기상관(Moran's I)"),
        ("마케팅믹스", "베이지안 MMM"),
        ("자원배분", "선형계획법 최적화"),
        ("코호트", "코호트별 재구매율 추이"),
        ("퍼널", "단계별 전환율 분석"),
    ]
    for hyp_type, method in new_types:
        text = _block(hyp_type, method)
        rewritten, signals = screen_hypotheses(text, {})

        assert len(signals) == 1, hyp_type
        assert signals[0]["type"] == hyp_type
        assert signals[0]["strength"] == "미측정"  # 크래시·오분류 없이 안전하게 처리
        assert f"유형: {hyp_type}" in rewritten
        assert "사전신호: 미측정" in rewritten


def test_new_type_hypothesis_not_dropped_even_with_no_signal():
    # 관계추론/회귀만 명백무상관 드롭 대상이지, 새 유형은 신호가 없어도(미측정) 드롭 안 된다.
    text = _block("생애가치", "BG/NBD+Gamma-Gamma 기반 LTV 추정")
    rewritten, signals = screen_hypotheses(text, {})

    assert "[가설 1]" in rewritten
    assert signals[0]["drop_candidate"] is False
