import pandas as pd
from DATA_Analyst_Assistant_Agent.agents.eda.tools.chart_requests import from_distribution_skill
from DATA_Analyst_Assistant_Agent.agents.eda.tools.visualize import (
    plot_distributions,
    plot_boxplots,
    plot_violins,
    plot_category_distribution,
    plot_grouped_box,
    plot_distribution_by_target,
)


def run_distribution_skill(
    df: pd.DataFrame,
    measure_cols: list = None,
    question_type: str = "",
    priority_metrics: list = None,
    key_col: str = None,
    target_col: str = None,
) -> dict:
    """
    단변량 분포 분석 skill.
    question_type에 따라 생성 차트를 조정한다.

    - distribution : hist + box + violin + catdist (전체)
    - comparison   : box만 (그룹 간 분포 파악용)
    - relationship : priority_metrics 컬럼만 hist + violin
    - time         : hist만
    """
    qt = question_type.lower()
    result = {}

    if qt == "comparison":
        # 그룹 비교 맥락 — 분포 형태보다 이상치/범위 파악이 우선
        result["boxplots"] = plot_boxplots(df, measure_cols=measure_cols)

    elif qt == "relationship":
        # 관계 분석 맥락 — 전체 컬럼 분포 형태 파악 (편향, 이상치 범위) 필요
        result["distributions"] = plot_distributions(df, measure_cols=measure_cols)
        result["violins"]       = plot_violins(df, measure_cols=measure_cols)

    elif qt == "time":
        result["distributions"] = plot_distributions(df, measure_cols=measure_cols)

    else:  # distribution 또는 기본값
        result["distributions"]         = plot_distributions(df, measure_cols=measure_cols)
        result["boxplots"]              = plot_boxplots(df, measure_cols=measure_cols)
        result["violins"]               = plot_violins(df, measure_cols=measure_cols)
        result["category_distribution"] = plot_category_distribution(df)

    # 다변수 교차: 범주별 분포 + target 수준별 분포 (원본 데이터에서만 발화 — 함수가 자체 게이트)
    result["grouped_box"] = plot_grouped_box(df, key_col=key_col, measure_cols=measure_cols)
    if target_col:
        result["dist_by_target"] = plot_distribution_by_target(df, target_col=target_col, measure_cols=measure_cols)

    result["chart_requests"] = from_distribution_skill(result)
    return result
