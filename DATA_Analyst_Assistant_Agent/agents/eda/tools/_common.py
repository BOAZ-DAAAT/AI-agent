"""@tool 들이 공유하는 헬퍼."""

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import accumulate_chart_requests


def _emit_and_dump(ctx, result) -> str:
    """skill 결과에서 차트 주문서(chart_requests)를 분리해 ctx에 누적하고,
    나머지만 JSON으로 반환한다(주문서는 LLM에 노출하지 않아 토큰 낭비 없음)."""
    if isinstance(result, dict):
        accumulate_chart_requests(ctx, result.pop("chart_requests", []))
    return json.dumps(result, ensure_ascii=False)
