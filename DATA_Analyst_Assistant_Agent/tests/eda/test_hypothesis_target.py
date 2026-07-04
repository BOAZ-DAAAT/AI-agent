from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.hypothesis import _resolve_target


def test_resolve_target_accepts_priority_metric_dict_entries() -> None:
    set_context(
        EdaContext(
            df=pd.DataFrame({"total_revenue": [1, 2], "sales_date": ["2024-01-01", "2024-01-02"]}),
            measure_cols=["total_revenue"],
        )
    )
    try:
        state = {
            "analysis_plan": {
                "priority_metrics": [
                    {"metric": "total_revenue", "reason": "핵심 매출 지표"},
                ]
            }
        }
        assert _resolve_target(state) == "total_revenue"
    finally:
        reset_context()
