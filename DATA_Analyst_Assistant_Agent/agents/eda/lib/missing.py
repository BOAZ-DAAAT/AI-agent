import pandas as pd


def detect_missing(df: pd.DataFrame) -> dict:
    """컬럼별 결측치 수와 비율 반환"""
    missing_count = df.isnull().sum()
    missing_ratio = (df.isnull().sum() / len(df)).round(4)
    return {
        "missing_count": missing_count.to_dict(),
        "missing_ratio": missing_ratio.to_dict(),
    }
