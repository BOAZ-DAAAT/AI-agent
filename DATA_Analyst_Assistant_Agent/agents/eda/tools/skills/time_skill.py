import pandas as pd
from DATA_Analyst_Assistant_Agent.agents.eda.tools.chart_requests import from_time_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.visualize import (
    plot_timeseries,
    plot_seasonality,
    plot_multiline_timeseries,
)


def run_time_skill(df: pd.DataFrame, measure_cols: list = None, time_cols: list = None,
                   key_col: str = None) -> dict:
    """
    시계열 / 시즌성 분석 skill.
    구성 tool:
        - plot_timeseries  : datetime 기준 추세 + 변화율
        - plot_seasonality : 월/요일 시즌성 bar chart
        - plot_multiline   : 카테고리별 시간 추세(시간 × 범주, key_col 있을 때만)
    """
    result = {
        "timeseries":  plot_timeseries(df, measure_cols=measure_cols, time_cols=time_cols),
        "seasonality": plot_seasonality(df, measure_cols=measure_cols, time_cols=time_cols),
    }
    if key_col:
        result["multiline"] = plot_multiline_timeseries(
            df, time_col=(time_cols or [None])[0], key_col=key_col, measure_cols=measure_cols)
    result["chart_requests"] = from_time_skill(result, time_cols=time_cols)
    return result
