import pandas as pd
from DATA_Analyst_Assistant_Agent.agents.eda.tools.chart_requests import from_relationship_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.visualize import (
    plot_correlation,
    plot_scatter_pairs,
    plot_distribution_by_target,
)


def run_relationship_skill(
    df: pd.DataFrame,
    measure_cols: list = None,
    question_type: str = "",
    target_col: str = None,
) -> dict:
    """
    변수 간 관계 분석 skill.
    question_type에 따라 생성 차트를 조정한다.

    - relationship : correlation + scatter_pairs (전체)
    - comparison   : correlation만 (지표 간 관계 파악용)
    - distribution : correlation만
    - time         : 생략
    """
    qt = question_type.lower()
    result = {}

    if qt == "time":
        return result  # 생략

    elif qt in ("comparison", "distribution"):
        result["correlation"] = plot_correlation(df, measure_cols=measure_cols)

    else:  # relationship 또는 기본값
        result["correlation"]   = plot_correlation(df, measure_cols=measure_cols)
        result["scatter_pairs"] = plot_scatter_pairs(df, measure_cols=measure_cols)

    # 쿼리 정조준: target 수준별 분포 (관계/회귀 질문의 핵심 차트, 함수가 자체 게이트)
    if target_col:
        result["dist_by_target"] = plot_distribution_by_target(df, target_col=target_col, measure_cols=measure_cols)

    result["chart_requests"] = from_relationship_skill(result)
    return result
