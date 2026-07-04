from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.execute import build_analysis_result
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisExecutionPlan, AnalysisKind
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def test_failed_planned_tool_is_reflected_in_evidence() -> None:
    state = OrchestrationState(run_id="run1", thread_id="thread1", user_query="월별 추이 분석", artifact_ids={})
    dataframe = pd.DataFrame(
        [
            {"sales_date": "2017-06-01", "daily_revenue": 10.0},
            {"sales_date": "2017-06-02", "daily_revenue": 12.0},
            {"sales_date": "2017-06-03", "daily_revenue": 14.0},
            {"sales_date": "2017-06-04", "daily_revenue": 16.0},
        ]
    )
    plan = AnalysisExecutionPlan(
        objective="time series",
        question_type="time_series",
        analysis_kind=AnalysisKind.time_series,
        analysis_subtype="aggregated_time_series",
        tool_names=["analyze_time_series"],
        metric="daily_revenue",
        time_column="sales_date",
    )

    result = build_analysis_result(
        state,
        dataframe=dataframe,
        eda_profiles=[],
        question_type="time_series",
        execution_plan=plan,
    )

    assert [item["tool_name"] for item in result["evidence"]] == ["analyze_time_series"]
    assert result["evidence"][0]["method"] == "failed_execution"
    assert result["evidence"][0]["statistics"]["status"] == "failed"
    assert "Aggregation produced fewer than four time periods." in result["evidence"][0]["statistics"]["error"]
