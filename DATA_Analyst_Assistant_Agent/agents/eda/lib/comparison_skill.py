import pandas as pd
from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_requests import from_comparison_skill
from DATA_Analyst_Assistant_Agent.agents.eda.lib.measure_policy import COMPARISON_FAMILY_CAP, select_capped_metrics
from DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize import (
    _get_numeric_cols,
    plot_top_n_barplot,
    plot_mean_ci_comparison,
    plot_segment_flag_profiles,
    plot_heatmap_matrix,
    plot_bubble,
    plot_radar,
    plot_grouped_bar,
    plot_crosstab_heatmap,
)


def run_comparison_skill(
    df: pd.DataFrame,
    key_col: str = None,
    measure_cols: list = None,
    question_type: str = "",
    priority_metrics: list = None,
) -> dict:
    """
    그룹 간 비교 분석 skill.
    question_type에 따라 생성 차트를 조정한다.

    - comparison   : 전체 (bar + heatmap + bubble + radar + grouped_bar)
    - relationship : bubble만 (주요 지표 간 포지셔닝 확인용)
    - time         : bar만 (시간대별 그룹 비교)
    - distribution : 생략 (비교 관점 불필요)
    """
    qt = question_type.lower()
    result = {}

    if qt == "distribution":
        return result  # 생략

    elif qt == "relationship":
        result["bubble"] = plot_bubble(df, key_col=key_col, measure_cols=measure_cols)

    elif qt == "time":
        result["top_n_barplot"] = plot_top_n_barplot(df, key_col=key_col, measure_cols=measure_cols)

    else:  # comparison 또는 기본값
        # priority_metrics 우선 + family 상한(4개)으로 지표 수를 제한 — "전 지표 × 7개 차트
        # 함수"로 인한 과잉생성 컷(#166). 각 함수의 key_col/자체 게이트 로직은 그대로 둔다.
        numeric_pool = _get_numeric_cols(df, measure_cols, allow_flags=False)
        capped = select_capped_metrics(numeric_pool, priority_metrics, max_n=COMPARISON_FAMILY_CAP)
        result["top_n_barplot"]  = plot_top_n_barplot(df, key_col=key_col, measure_cols=capped)
        result["mean_ci"]        = plot_mean_ci_comparison(df, key_col=key_col, measure_cols=capped)
        result["segment_profile"] = plot_segment_flag_profiles(df, measure_cols=capped)
        result["heatmap_matrix"] = plot_heatmap_matrix(df, key_col=key_col, measure_cols=capped)
        result["bubble"]         = plot_bubble(df, key_col=key_col, measure_cols=capped)
        result["radar"]          = plot_radar(df, key_col=key_col, measure_cols=capped)
        result["grouped_bar"]    = plot_grouped_bar(df, key_col=key_col, measure_cols=capped)

    # 범주 × 범주 교차표 (범주형 2개 이상일 때만 — 함수가 자체 게이트)
    if qt != "distribution":
        result["crosstab"] = plot_crosstab_heatmap(df, cat_a=key_col)

    result["chart_requests"] = from_comparison_skill(result, key_col=key_col, measure_cols=measure_cols)
    return result
