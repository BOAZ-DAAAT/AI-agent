"""#194 — insight/hypothesis가 짧은 구조화 필드를 뱉고, 최종 요약이 긴 원문이 아니라
그 짧은 필드만 입력으로 받는지 검증한다."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import split_marked_json
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import output_summary_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.insight import SUMMARY_FACTS_MARKER
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.hypothesis import PRIMARY_HYPOTHESIS_MARKER


def test_split_marker_absent_keeps_prose_intact():
    prose, parsed = split_marked_json("완전한 서술만 있다", "===X===")
    assert prose == "완전한 서술만 있다"
    assert parsed is None


def test_split_summary_facts_array():
    raw = f'[핵심 패턴] ...\n\n{SUMMARY_FACTS_MARKER}\n["사실1", "사실2", "사실3"]'
    prose, parsed = split_marked_json(raw, SUMMARY_FACTS_MARKER)
    assert prose.startswith("[핵심 패턴]")
    assert SUMMARY_FACTS_MARKER not in prose  # 프로즈엔 마커/JSON 안 남음
    assert parsed == ["사실1", "사실2", "사실3"]


def test_split_primary_hypothesis_object():
    raw = (f'[가설 1] ...\n{PRIMARY_HYPOTHESIS_MARKER}\n'
           '{"target": "score", "feature": "price", "method": "Kruskal-Wallis"}')
    prose, parsed = split_marked_json(raw, PRIMARY_HYPOTHESIS_MARKER)
    assert prose.startswith("[가설 1]")
    assert parsed == {"target": "score", "feature": "price", "method": "Kruskal-Wallis"}


def test_split_malformed_json_falls_back_to_none_but_keeps_prose():
    raw = f"서술 본문\n{SUMMARY_FACTS_MARKER}\n[깨진 JSON"
    prose, parsed = split_marked_json(raw, SUMMARY_FACTS_MARKER)
    assert prose == "서술 본문"
    assert parsed is None


def test_output_summary_prompt_uses_facts_not_long_text():
    facts = ["총 6954개 제품 단위 집계", "리뷰점수 결측 1.35%"]
    ph = {"target": "review_score", "feature": "price_band", "method": "Kruskal-Wallis"}
    prompt = output_summary_prompt(facts, ph)

    # 짧은 사실이 그대로 들어가고, 마무리 문장이 1순위 가설로 구성됨
    assert "총 6954개 제품 단위 집계" in prompt
    assert "Kruskal-Wallis으로 review_score~price_band" in prompt
    # 긴 원문(insight_result/hypotheses)을 통째로 받는 옛 파라미터가 없어야 함
    assert "[핵심 사실]" in prompt
    assert "[인사이트]" not in prompt


def test_output_summary_prompt_handles_empty_primary_hypothesis():
    prompt = output_summary_prompt(["사실1"], {})
    # 가설 필드가 비어도 크래시 없이, 일반 마무리 문구로 폴백
    assert "우선 검증할 가설을 선택하라" in prompt
