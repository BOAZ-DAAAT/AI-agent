"""@tool 들이 공유하는 헬퍼."""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import accumulate_chart_requests


def json_safe(obj):
    """dict 키까지 포함해 JSON 직렬화 가능한 형태로 재귀 변환한다.

    json.dumps(default=...)는 값(value)에만 적용되고 dict 키에는 적용되지 않는다 —
    키가 date/datetime 등이면 default를 줘도 "keys must be str, int, float, bool or
    None" TypeError가 그대로 난다(#132). 그래서 키도 값처럼 먼저 안전화한 뒤 str()로
    감싼다(정상 문자열 키는 그대로, 문제되는 타입만 실질적으로 바뀜).
    """
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (date, datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


def _emit_and_dump(ctx, result) -> str:
    """skill 결과에서 차트 주문서(chart_requests)를 분리해 ctx에 누적하고,
    나머지만 JSON으로 반환한다(주문서는 LLM에 노출하지 않아 토큰 낭비 없음)."""
    if isinstance(result, dict):
        accumulate_chart_requests(ctx, result.pop("chart_requests", []))
    return json.dumps(json_safe(result), ensure_ascii=False)
