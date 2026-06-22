"""EDA ↔ 차트 렌더러 공유 계약.

EDA는 ChartRequest(차트 '주문서')만 발행한다 — 자연어 의도 + 검증된 수치 +
사용 컬럼 + 표준차트 힌트. encoding(축/색/마크) 결정과 렌더링은 chart/ 패키지
(Phase B)의 몫이다. EDA는 matplotlib/LLM/encoding을 일절 결정하지 않는다.

ChartSpec은 chart/가 돌려주는 렌더 가능한 결과물이다(Phase B에서 사용).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ChartRequest(BaseModel):
    """EDA가 chart/에 넘기는 주문서. encoding은 없다 — 의도와 근거만."""

    intent: str                                       # 자연어 의도: "배송일과 리뷰점수의 음의 상관을 보여줘"
    stats: Dict[str, Any] = Field(default_factory=dict)      # 검증된 수치(환각 방지 닻): {"r": -0.42, "n": 9200}
    columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # 컬럼 의미/타입: {"delivery_days": {"type": "numeric"}}
    hint: Optional[str] = None                        # 표준 차트면 종류 힌트(LLM 건너뛰기용): "heatmap"
    data_ref: Optional[str] = None                    # 데이터 참조(행이 많아 스펙에 안 실음). Phase B에서 채움


class ChartSpec(BaseModel):
    """chart/가 돌려주는 결과. UI가 이걸 렌더한다(Phase B)."""

    chart_type: str                                   # "heatmap" | "scatter" | ...
    vega_lite: Dict[str, Any]                         # 결정론적 렌더용 Vega-Lite JSON
    title: str
    source: str = "rule"                              # "rule" | "llm" (어떻게 만들었는지 추적)
