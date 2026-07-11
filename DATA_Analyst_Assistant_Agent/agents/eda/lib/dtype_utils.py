"""컬럼 dtype 판별 공용 유틸.

MySQL DATE/DATETIME 컬럼은 pandas로 읽으면 datetime64가 아니라 object dtype에
date/datetime 값이 그대로 든 형태로 오는 경우가 흔하다. 이걸 그대로 범주형으로 취급하면
value_counts()/crosstab() 결과의 인덱스(=나중에 dict 키)가 date/datetime이 되어
JSON 직렬화 시 크래시한다("keys must be str, int, float, bool or None", #132).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

_SAMPLE_SIZE = 20


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
