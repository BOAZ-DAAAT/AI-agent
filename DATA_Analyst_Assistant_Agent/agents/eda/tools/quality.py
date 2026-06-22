import pandas as pd


def check_duplicates_fn(df: pd.DataFrame) -> dict:
    """중복 행 수와 비율 반환"""
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_count": dup_count,
        "duplicate_ratio": round(dup_count / len(df), 4),
    }


def check_sample_reliability_fn(df: pd.DataFrame, min_samples: int = 30, key_col: str = None, count_col: str = None) -> dict:
    """
    집계 마트 기준으로 표본 수가 적은 그룹 탐지.
    count_col이 주어지면 우선 사용, 없으면 키워드로 탐지.
    """
    cat_cols = list(df.select_dtypes(include=["object"]).columns)
    if not cat_cols:
        return {"message": "범주형 키 컬럼이 없습니다."}

    if key_col is None or key_col not in df.columns:
        key_col = cat_cols[0]

    # LLM이 분류한 count_col 우선, 없으면 키워드 휴리스틱 폴백
    if count_col and count_col in df.columns:
        pass
    else:
        fallback = [
            c for c in df.columns
            if any(k in c.lower() for k in ["total_order", "order_count", "count", "total", "cnt"])
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        count_col = fallback[0] if fallback else None

    if not count_col:
        return {"message": "표본 수 기준 컬럼을 찾을 수 없습니다."}

    low = df[df[count_col] < min_samples][[key_col, count_col]]

    return {
        "reference_column": count_col,
        "min_threshold": min_samples,
        "low_sample_count": len(low),
        "low_sample_groups": low.head(10).to_dict(orient="records"),
        "warning": f"{len(low)}개 그룹의 {count_col}이 {min_samples} 미만 — 해당 그룹의 평균값 신뢰도 낮음",
    }
