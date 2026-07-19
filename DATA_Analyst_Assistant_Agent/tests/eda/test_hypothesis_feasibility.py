"""correct_hypothesis_feasibility 계약 테스트 — '유형: 그룹차이' 우선 판정, 검정 이름
목록에 없어도(Mann-Whitney 등) 잡히는지 확인한다.

run-019f7437 실제 사례: 마트가 그룹당 1행(총 2행)인데 hypothesis가 Mann-Whitney U
검정을 추천했고, 당시 _GROUP_DIFF_METHODS 목록에 Mann-Whitney가 없어 교정되지 않은 채
final_summary를 통해 analysis_agent에게 그대로 전달됐다.
"""

from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import (
    correct_hypothesis_feasibility,
)


def _df() -> pd.DataFrame:
    # 'segment'는 그룹당 1행(집계본, 불가능) / 'category'는 그룹당 여러 행(가능)
    return pd.DataFrame({
        "segment": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "customer_count": [93099, 2997, 100, 200, 300, 400],
        "category": ["A", "A", "B", "B", "C", "C"],
        "amount": [1, 2, 3, 4, 5, 6],
    })


def _block(method: str, feature: str, hyp_type: str = "그룹차이") -> str:
    return f"""[가설 1]
관찰: 어떤 관찰
유형: {hyp_type}
H0: 귀무가설
H1: 대립가설
검증방법: {method}
필요변수: target=customer_count, feature={feature}
현재데이터: 현재 마트로 검증 가능하다"""


def test_group_diff_type_with_unlisted_method_still_corrected():
    # Mann-Whitney는 _GROUP_DIFF_METHOD_HINTS 폴백 목록에도 있지만, 유형 라인이 최우선이라
    # 목록에 아예 없는 새 검정 이름이 와도(가짜 이름으로 검증) 유형만으로 잡혀야 한다.
    text = _block("Some Brand New Test 검정", "segment", hyp_type="그룹차이")

    corrected, corrections = correct_hypothesis_feasibility(_df(), text, data_level={"level": "aggregated"})

    assert len(corrections) == 1
    assert "추가 필요" in corrected
    assert "segment" in corrected


def test_mann_whitney_case_from_real_bug_is_corrected():
    text = _block("Mann-Whitney U 검정", "segment")

    corrected, corrections = correct_hypothesis_feasibility(_df(), text, data_level={"level": "aggregated"})

    assert len(corrections) == 1
    assert "현재데이터: 추가 필요" in corrected
    assert "Mann-Whitney" in corrected


def test_feasible_group_diff_not_corrected():
    text = _block("일원배치 ANOVA", "category")

    corrected, corrections = correct_hypothesis_feasibility(_df(), text, data_level={"level": "aggregated"})

    assert corrections == []
    assert "현재데이터: 현재 마트로 검증 가능하다" in corrected


def test_non_group_diff_type_skips_method_hint_fallback():
    # 유형이 명시적으로 '관계추론'이면, 검증방법 텍스트에 우연히 anova 비슷한 단어가 있어도
    # 그룹차이 폴백을 쓰지 않는다(유형 라인이 오탐 방지 역할).
    text = _block("스피어만 상관검정(anova 아님)", "segment", hyp_type="관계추론")

    corrected, corrections = correct_hypothesis_feasibility(_df(), text, data_level={"level": "aggregated"})

    assert corrections == []


def test_missing_type_line_falls_back_to_method_hint():
    # 유형 라인 자체가 없으면(HYPOTHESIS_TYPE_GUIDE=False 등) 검증방법 텍스트로 폴백 추정한다.
    block = """[가설 1]
관찰: 어떤 관찰
H0: 귀무가설
H1: 대립가설
검증방법: Kruskal-Wallis 검정
필요변수: target=customer_count, feature=segment
현재데이터: 현재 마트로 검증 가능하다"""

    corrected, corrections = correct_hypothesis_feasibility(_df(), block, data_level={"level": "aggregated"})

    assert len(corrections) == 1
    assert "추가 필요" in corrected
