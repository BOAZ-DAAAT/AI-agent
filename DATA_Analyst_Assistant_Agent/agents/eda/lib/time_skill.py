import pandas as pd
from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_requests import from_time_skill
from DATA_Analyst_Assistant_Agent.agents.eda.lib.measure_policy import TIME_FAMILY_CAP, select_capped_metrics
from DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize import (
    _get_numeric_cols,
    plot_timeseries,
    plot_seasonality,
    plot_multiline_timeseries,
)


def run_time_skill(df: pd.DataFrame, measure_cols: list = None, time_cols: list = None,
                   key_col: str = None, priority_metrics: list = None) -> dict:
    """
    시계열 / 시즌성 분석 skill.
    구성 tool:
        - plot_timeseries  : datetime 기준 추세 + 변화율
        - plot_seasonality : 월/요일 시즌성 bar chart
        - plot_multiline   : 카테고리별 시간 추세(시간 × 범주, key_col 있을 때만)

    priority_metrics 우선 + family 상한(2개)으로 지표 수를 제한한다 — 마트의 전 수치형
    컬럼(플래그 포함)을 무조건 다 추세선으로 그리던 과잉생성 컷(#166).
    """
    numeric_pool = _get_numeric_cols(df, measure_cols, allow_flags=False)
    capped = select_capped_metrics(numeric_pool, priority_metrics, max_n=TIME_FAMILY_CAP)
    result = {
        "timeseries":  plot_timeseries(df, measure_cols=capped, time_cols=time_cols),
        "seasonality": plot_seasonality(df, measure_cols=capped, time_cols=time_cols),
    }
    if key_col:
        result["multiline"] = plot_multiline_timeseries(
            df, time_col=(time_cols or [None])[0], key_col=key_col, measure_cols=capped)
    result["chart_requests"] = from_time_skill(result, time_cols=time_cols)
    return result
