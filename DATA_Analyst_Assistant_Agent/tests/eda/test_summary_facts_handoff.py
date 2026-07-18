"""#194 — insight/hypothesis가 짧은 구조화 필드를 뱉고, 최종 요약이 긴 원문이 아니라
그 짧은 필드만 입력으로 받는지 검증한다."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import split_marked_json
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
