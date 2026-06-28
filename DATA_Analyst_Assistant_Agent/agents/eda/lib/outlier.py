import pandas as pd
import pandas.api.types


def detect_outliers_iqr(df: pd.DataFrame, measure_cols: list = None) -> dict:
    """IQR 기준 수치형 컬럼별 이상치 수와 비율 반환"""
    result = {}
    if measure_cols:
        numeric_cols = [c for c in measure_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        result[col] = {
            "outlier_count": int(outlier_mask.sum()),
            "outlier_ratio": round(outlier_mask.sum() / len(df), 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        }
    return result
