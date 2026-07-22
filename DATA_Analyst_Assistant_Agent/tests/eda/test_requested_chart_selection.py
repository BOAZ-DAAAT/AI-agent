from __future__ import annotations

import os

from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import chart_selector as CS


def _make_png(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"fake-png-bytes")
    return str(path)


def _base_state(question: str) -> dict:
    return {
        "user_question": question,
        "inspect_result": "",
        "quality_result": "",
        "distribution_result": "",
        "comparison_result": "",
        "relationship_result": "",
        "time_result": "",
        "hypotheses": "",
        "statistical_metadata": {},
        "analysis_plan": {},
    }


def test_branch_boxplot_request_is_promoted_to_key_charts(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(visualize, "KEY_DIR", str(tmp_path / "key"))
    (tmp_path / "key").mkdir()

    scatter = _make_png(tmp_path, "scatter_delivery_days_vs_review_score.png")
    boxplot = _make_png(tmp_path, "box_review_score.png")

    def _fake_skill(**kwargs):
        return [scatter], {os.path.basename(scatter): "selected by LLM"}, {"dropped": [], "check_failures": 0}

    monkeypatch.setattr(CS, "run_chart_selector_skill", _fake_skill)

    state = _base_state("원본 질문\n\n[추가 지시사항] 리뷰 점수에 관한 박스 플롯 차트도 EDA로 보고 싶어")

    result = CS.chart_selector_node(state)

    assert scatter in result["key_charts"]
    assert boxplot in result["key_charts"]
    assert os.path.exists(tmp_path / "key" / "box_review_score.png")
    assert "요청한 박스플롯" in result["key_chart_captions"]["box_review_score.png"]


def test_boxplot_request_is_branch_only(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(visualize, "KEY_DIR", str(tmp_path / "key"))
    (tmp_path / "key").mkdir()

    scatter = _make_png(tmp_path, "scatter_delivery_days_vs_review_score.png")
    boxplot = _make_png(tmp_path, "box_review_score.png")

    def _fake_skill(**kwargs):
        return [scatter], {os.path.basename(scatter): "selected by LLM"}, {"dropped": [], "check_failures": 0}

    monkeypatch.setattr(CS, "run_chart_selector_skill", _fake_skill)

    state = _base_state("리뷰 점수에 관한 박스 플롯 차트도 EDA로 보고 싶어")

    result = CS.chart_selector_node(state)

    assert scatter in result["key_charts"]
    assert boxplot not in result["key_charts"]
    assert not os.path.exists(tmp_path / "key" / "box_review_score.png")
