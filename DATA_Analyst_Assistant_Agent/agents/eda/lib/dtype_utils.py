"""컬럼 dtype 판별 공용 유틸.

MySQL DATE/DATETIME 컬럼은 pandas로 읽으면 datetime64가 아니라 object dtype에
date/datetime 값이 그대로 든 형태로 오는 경우가 흔하다. 이걸 그대로 범주형으로 취급하면
value_counts()/crosstab() 결과의 인덱스(=나중에 dict 키)가 date/datetime이 되어
JSON 직렬화 시 크래시한다("keys must be str, int, float, bool or None", #132).
"""

from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

_SAMPLE_SIZE = 20
_SNAPSHOT_NAME_RE = re.compile(r"(anchor|snapshot|as_?of)", re.IGNORECASE)
_MIN_TIME_BUCKETS = 3     # 서로 다른 시점이 이보다 적으면 상수/거의상수 — 추세·계절성 무의미
_MAX_BUCKET_RATIO = 0.5   # distinct 시점 수 / 행수. 이보다 크면 "시점당 반복관측"이 없는 엔티티 속성


def categorical_object_columns(df: pd.DataFrame, sample_size: int = _SAMPLE_SIZE) -> list[str]:
    """object dtype 컬럼 중 실제 범주형만 반환한다(date/datetime 값 컬럼은 제외).

    첫 값만 보면 혼합 타입 컬럼에서 놓칠 수 있어 앞부분 표본(sample_size개)을 전부 확인한다.
    표본 중 하나라도 date/datetime/Timestamp 인스턴스면 그 컬럼은 제외한다.
    """
    result: list[str] = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(sample_size)
        if sample.empty:
            result.append(col)
            continue
        if sample.map(lambda x: isinstance(x, (date, datetime, pd.Timestamp))).any():
            continue
        result.append(col)
    return result


def usable_time_columns(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    """시계열 추세/계절성 차트로 쓸 수 있는 datetime 컬럼만 남긴다.

    고객 단위 집계 마트에는 anchor_date(전 행 동일값인 조회 시점)나 first_purchase_date
    (고객당 1회성 속성)처럼 dtype만 datetime인 컬럼이 섞여 있다. 이런 컬럼은 시간 버킷당
    반복관측이 없어(anchor_date=버킷 1개, first_purchase_date=행마다 버킷 1개) 추세선/계절성
    집계가 상수 그림이거나 사실상 산점도라 무의미하다. 이름 신호(anchor/snapshot/as_of)로
    스냅샷류를 먼저 걸러내고, 나머지는 '서로 다른 날짜 수'가 최소치 이상이면서 행수 대비
    너무 많지 않은(=날짜당 여러 행이 실제로 쌓이는) 컬럼만 남긴다.
    """
    n_rows = len(df)
    out: list[str] = []
    for col in candidate_cols:
        if col not in df.columns or _SNAPSHOT_NAME_RE.search(str(col)):
            continue
        s = pd.to_datetime(df[col], errors="coerce").dropna()
        if s.empty or n_rows == 0:
            continue
        # Avoid dt.floor("D") here: on some Windows/Python 3.13 pandas builds it can
        # crash the interpreter for object-origin datetimes. String day buckets are
        # slower but safe, and load_mart only needs a small suitability check.
        distinct_days = s.dt.strftime("%Y-%m-%d").nunique()
        if distinct_days < _MIN_TIME_BUCKETS or distinct_days > n_rows * _MAX_BUCKET_RATIO:
            continue
        out.append(col)
    return out
