import pandas as pd
from DATA_Analyst_Assistant_Agent.agents.eda.tools.missing import detect_missing
from DATA_Analyst_Assistant_Agent.agents.eda.tools.outlier import detect_outliers_iqr
from DATA_Analyst_Assistant_Agent.agents.eda.tools.quality import check_duplicates_fn, check_sample_reliability_fn


def run_data_quality_skill(df: pd.DataFrame, key_col: str = None, measure_cols: list = None, count_col: str = None) -> dict:
    """
    데이터 품질 전반을 점검하는 skill.
    구성 tool:
        - detect_missing         : 결측치 수 / 비율
        - detect_outliers_iqr    : IQR 기준 이상치
        - check_duplicates_fn    : 중복 행
        - check_sample_reliability_fn : 표본 수 신뢰도
    """
    return {
        "missing":            detect_missing(df),
        "outliers":           detect_outliers_iqr(df, measure_cols=measure_cols),
        "duplicates":         check_duplicates_fn(df),
        "sample_reliability": check_sample_reliability_fn(df, key_col=key_col, count_col=count_col),
    }
