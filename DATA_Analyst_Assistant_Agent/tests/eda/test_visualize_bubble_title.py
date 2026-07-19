"""plot_bubble 제목에 한글이 안 들어가는지 검증한다.

matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프가 없어 "크기"/"색" 같은 한글이
빈 박스(tofu)로 깨진다(run-019f76e4 실사례). size/color 라벨을 영어로 고정한다.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize


def _df() -> pd.DataFrame:
    # 순차 정수(range 등)는 ID류로 판단돼 numeric_cols에서 제외되므로, 비단조 값을 쓴다.
    return pd.DataFrame({
        "category": list("ABCABCABCA"),
        "metric_x": [1.5, 2.3, 1.1, 4.4, 2.2, 3.3, 1.9, 2.8, 3.1, 1.4],
        "metric_y": [10, 12, 11, 15, 13, 14, 10, 12, 13, 11],
        "metric_size": [5, 7, 6, 9, 8, 7, 5, 6, 8, 7],
        "metric_color": [100, 120, 110, 150, 130, 140, 100, 120, 130, 110],
    })


def test_bubble_title_has_no_korean_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(visualize, "OUTPUT_DIR", str(tmp_path))

    captured = {}

    def _fake_apply_style(ax, title, **kwargs):
        captured["title"] = title

    monkeypatch.setattr(visualize, "_apply_style", _fake_apply_style)

    result = visualize.plot_bubble(_df(), key_col="category")

    assert result.get("chart_path") is not None
    assert "title" in captured
    assert not re.search(r"[가-힣]", captured["title"]), captured["title"]
    assert "size:" in captured["title"]
    assert "color:" in captured["title"]
