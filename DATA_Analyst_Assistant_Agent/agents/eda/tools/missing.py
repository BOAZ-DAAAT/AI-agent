import pandas as pd


def detect_missing(df: pd.DataFrame) -> dict:
    """컬럼별 결측치 수와 비율 반환"""
    missing_count = df.isnull().sum()
    missing_ratio = (df.isnull().sum() / len(df)).round(4)
    return {
        "missing_count": missing_count.to_dict(),
        "missing_ratio": missing_ratio.to_dict(),
    }


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 처리: 수치형은 중앙값, 문자형은 'Unknown'으로 대체"""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")
    return df
