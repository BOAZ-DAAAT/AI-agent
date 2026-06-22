"""EDA 입력 계약 v1 + fixture 로더.

입력 계약 = SQL 에이전트가 EDA에 넘겨야 할 '이상적' 입력. 노션 EDA Agent 스펙의
입력부에서 꼭 필요한 것만 추림: 분석 단위(grain)·데이터 레벨·컬럼 역할.

사실상 '미리 채워진 EdaContext + grain'이다:
  numeric → measure_cols, categorical[0] → key_col, datetime → time_cols, count → count_col.
이 계약이 오면 EDA는 load 노드의 LLM 컬럼분류를 건너뛸 수 있다(토큰 절감).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class EdaInput:
    """EDA 입력 계약 v1."""

    name: str
    question: str
    question_type: str                          # route_kind (relationship/comparison/time/...)
    grain_hint: str                             # "one row per order"
    level: str                                  # "raw" | "aggregated"
    numeric: List[str] = field(default_factory=list)        # 측정값(measure)
    categorical: List[str] = field(default_factory=list)    # 그룹/차원(dimension)
    datetime: List[str] = field(default_factory=list)
    id: List[str] = field(default_factory=list)             # 분석 제외
    count: Optional[str] = None                 # 표본수/볼륨 컬럼
    target_candidates: List[str] = field(default_factory=list)

    @property
    def key_col(self) -> Optional[str]:
        """대표 그룹/차원 컬럼(EdaContext.key_col 매핑)."""
        return self.categorical[0] if self.categorical else None

    @classmethod
    def from_dict(cls, d: dict) -> "EdaInput":
        cols = d.get("columns", {})
        grain = d.get("grain", {})
        return cls(
            name=d.get("name", ""),
            question=d.get("question", ""),
            question_type=d.get("question_type", ""),
            grain_hint=grain.get("grain_hint", ""),
            level=grain.get("level", "unknown"),
            numeric=cols.get("numeric", []),
            categorical=cols.get("categorical", []),
            datetime=cols.get("datetime", []),
            id=cols.get("id", []),
            count=cols.get("count"),
            target_candidates=cols.get("target_candidates", []),
        )


def load_fixture(name: str) -> Tuple[pd.DataFrame, EdaInput]:
    """fixtures/<name>.csv + <name>.contract.json 을 읽어 (df, EdaInput) 반환."""
    csv_path = os.path.join(FIXTURE_DIR, f"{name}.csv")
    json_path = os.path.join(FIXTURE_DIR, f"{name}.contract.json")
    df = pd.read_csv(csv_path)
    # 방어: 문자열 컬럼의 잔여 공백/\r 정리(원본 olist 범주값 오염 대비)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    # datetime 컬럼 파싱
    with open(json_path, "r", encoding="utf-8") as f:
        contract = EdaInput.from_dict(json.load(f))
    for c in contract.datetime:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df, contract
