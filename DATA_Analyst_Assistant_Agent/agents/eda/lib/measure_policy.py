"""차트 생성 지표 선정 정책 — priority_metrics 우선순위 + family별 상한.

각 분석 skill이 '어떤 지표를 몇 개나 그릴지' 정할 때 쓰는 공용 정책. planner가 첫 라운드에
정하는 priority_metrics([{"metric":..,"reason":..}] 형태)는 지금까지 각 skill의 파라미터로만
받고 실제로 쓰이지 않았다 — 그 결과 numeric_cols 전체를 무조건 다 그려 지표 수가 마트 폭만큼
그대로 차트 수가 됐다(고객 RFM 마트에서 시계열만 11개 지표 × 여러 차트타입 → 43장, #166).
여기서 priority_metrics를 앞에 두고 family별 상한으로 잘라 과잉생성을 지표 선정 단계에서
막는다.
"""

from __future__ import annotations

import pandas as pd

DIST_FAMILY_CAP = 6
COMPARISON_FAMILY_CAP = 4
TIME_FAMILY_CAP = 2


def _priority_names(priority_metrics) -> list:
    """planner 산출물 [{"metric":..,"reason":..}] 또는 평문 문자열 리스트 모두 허용."""
    if not priority_metrics:
        return []
    names = []
    for m in priority_metrics:
        name = m.get("metric") if isinstance(m, dict) else m
        if name:
            names.append(name)
    return names


def select_capped_metrics(numeric_cols: list, priority_metrics=None, max_n: int = DIST_FAMILY_CAP) -> list:
    """priority_metrics를 앞에 두고(실제 존재하는 컬럼만), 나머지는 원래 순서로 이어붙인 뒤
    family 상한으로 자른다. priority_metrics가 없거나 비어 있으면 원래 순서 그대로 상한만 적용."""
    priority = [c for c in _priority_names(priority_metrics) if c in numeric_cols]
    rest = [c for c in numeric_cols if c not in priority]
    return (priority + rest)[:max_n]


def pick_distribution_chart_types(s: pd.Series) -> list:
    """지표 하나당 분포 차트 최대 2종: 항상 dist(hist) + 신호 기반 1종.

    비대칭(|skew|>1)이면 꼬리/분위수를 보여주는 ecdf, 아니면 이상치/IQR을 보여주는 box.
    violin은 dist+box(또는 dist+ecdf)와 정보가 겹쳐 기본 예산에서 제외한다(#166 과잉생성 컷).
    """
    try:
        skew = float(s.skew())
    except (TypeError, ValueError):
        skew = 0.0
    return ["dist", "ecdf" if abs(skew) > 1 else "box"]
