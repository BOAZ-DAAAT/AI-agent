from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.insight.evidence import EvidencePack
from DATA_Analyst_Assistant_Agent.supervisor.insight.loop import (
    MAX_KEY_INSIGHTS,
    _build_prompt,
    _try_finish,
)


def _pack() -> EvidencePack:
    return EvidencePack(
        user_question="고객 세그먼트별 구매 패턴을 분석해줘",
        route_kind="comprehensive",
        table_summary={"rows": 10, "columns": ["segment", "sales"]},
        analysis={"method_summary": "클러스터링으로 4개 고객 군집을 도출했습니다."},
        source_artifact_ids=["artifact_analysis"],
    )


def test_try_finish_caps_key_insights_at_ten() -> None:
    result, missing = _try_finish(
        _pack(),
        {
            "answer": "군집별 구매 패턴 차이가 확인됩니다.",
            "key_insights": [f"insight {index}" for index in range(12)],
        },
        computes=[],
        charts=[],
        steps=[],
        round_idx=0,
    )

    assert missing == []
    assert result is not None
    assert len(result.key_insights) == MAX_KEY_INSIGHTS


def test_prompt_frames_insight_count_and_method_chart_purpose() -> None:
    prompt = _build_prompt(_pack(), observations=[], round_idx=0)

    assert "보통 4~6개" in prompt
    assert "최대 10개" in prompt
    assert "클러스터링의 군집 프로파일" in prompt
    assert "로지스틱 회귀의 계수/odds ratio" in prompt
