import pandas as pd


def get_basic_profile(df: pd.DataFrame) -> dict:
    """shape, dtypes, 기초통계 반환"""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(include="all").to_dict(),
    }
