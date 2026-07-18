"""#194 — chart_selector_node가 analysis_results/hypotheses 원문을 절단해서 넘기는지 검증한다."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import chart_selector as CS


def _make_png(tmp_path, name: str) -> None:
    (tmp_path / name).write_bytes(b"fake-png-bytes")


def test_long_analysis_results_and_hypotheses_are_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(visualize, "KEY_DIR", str(tmp_path / "key"))
    (tmp_path / "key").mkdir()
    _make_png(tmp_path, "dist_x.png")

    captured = {}

    def _fake_skill(**kwargs):
        captured.update(kwargs)
        return [], {}, {"dropped": [], "check_failures": 0}

    monkeypatch.setattr(CS, "run_chart_selector_skill", _fake_skill)

    long_text = "긴 서술 " * 200  # 300자 제한을 확실히 넘김
    state = {
        "user_question": "질문",
        "inspect_result": long_text,
        "quality_result": long_text,
        "distribution_result": "",
        "comparison_result": "",
        "relationship_result": "",
        "time_result": "",
        "hypotheses": long_text,
        "statistical_metadata": {},
        "analysis_plan": {},
    }

    CS.chart_selector_node(state)

    assert len(captured["analysis_results"]["inspect"]) <= 310  # 300 + "..." 여유
    assert len(captured["analysis_results"]["inspect"]) < len(long_text)
    assert len(captured["hypotheses"]) <= 610  # 600 + "..." 여유
    assert len(captured["hypotheses"]) < len(long_text)


def test_short_text_is_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(visualize, "KEY_DIR", str(tmp_path / "key"))
    (tmp_path / "key").mkdir()
    _make_png(tmp_path, "dist_x.png")

    captured = {}

    def _fake_skill(**kwargs):
        captured.update(kwargs)
        return [], {}, {"dropped": [], "check_failures": 0}

    monkeypatch.setattr(CS, "run_chart_selector_skill", _fake_skill)

    state = {
        "user_question": "질문",
        "inspect_result": "짧은 결과",
        "quality_result": "", "distribution_result": "", "comparison_result": "",
        "relationship_result": "", "time_result": "",
        "hypotheses": "짧은 가설",
        "statistical_metadata": {},
        "analysis_plan": {},
    }

    CS.chart_selector_node(state)

    assert captured["analysis_results"]["inspect"] == "짧은 결과"
    assert captured["hypotheses"] == "짧은 가설"
