from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.classify import (
    classify_intent,
    resolve_time_grain,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisContext, AnalysisIntent
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"order_date": dates, "amount": range(len(dates))})


def test_one_month_span_resolves_to_daily() -> None:
    # "last month" style question: ~30 days -> daily, never forced to monthly.
    dates = [f"2026-06-{day:02d}" for day in range(1, 31)]
    grain, span = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "D"
    assert span == 29


def test_two_year_span_resolves_to_monthly() -> None:
    dates = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2025-12-31"]
    grain, span = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "M"
    assert span >= 700


def test_half_year_span_resolves_to_weekly() -> None:
    dates = ["2026-01-01", "2026-03-01", "2026-06-01"]
    grain, _ = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "W"


def test_many_year_span_resolves_to_coarse_grain() -> None:
    dates = ["2018-01-01", "2026-01-01"]
    grain, _ = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "Y"


def test_missing_or_unparseable_time_column_returns_none() -> None:
    assert resolve_time_grain(_frame(["2026-06-01"]), None) == (None, None)
    assert resolve_time_grain(_frame(["2026-06-01"]), "nope") == (None, None)
    bad = pd.DataFrame({"order_date": ["n/a", "unknown"], "amount": [1, 2]})
    assert resolve_time_grain(bad, "order_date") == (None, None)


class _CapturingStructuredModel:
    def __init__(self, result: AnalysisIntent) -> None:
        self.result = result
        self.messages = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.result


def test_context_builder_copies_last_failure_and_classifier_serializes_it() -> None:
    failure = {
        "reason_code": "method_review_failed",
        "failure_reason": "wrong method",
    }
    state = OrchestrationState(
        run_id="run_001",
        user_query="매출 합계를 분석해줘",
        retry_context={"last_failure": failure},
    )
    dataframe = pd.DataFrame({"amount": [10, 20]})
    context = build_analysis_context(state, dataframe, [])
    model = _CapturingStructuredModel(AnalysisIntent(objective="매출 합계"))

    classify_intent(context, dataframe, model=model)

    assert context.last_failure == failure
    human_message = model.messages[1].content
    assert '"last_failure"' in human_message
    assert "method_review_failed" in human_message
    assert "wrong method" in human_message


def test_context_builder_exposes_eda_candidates_as_optional_hints() -> None:
    state = OrchestrationState(
        run_id="run_eda_candidates",
        user_query="category revenue analysis",
    )
    dataframe = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    context = build_analysis_context(
        state,
        dataframe,
        [{
            "insight_result": "1. Category B has higher observed revenue.\n- Ignore unrelated seasonality.",
            "hypotheses": [
                {"hypothesis": "Category B revenue is higher than category A."},
                "Weekend orders may differ from weekdays.",
            ],
        }],
    )

    assert context.eda_candidate_insights == [
        "Category B has higher observed revenue.",
        "Ignore unrelated seasonality.",
    ]
    assert context.eda_candidate_hypotheses == [
        "Category B revenue is higher than category A.",
        "Weekend orders may differ from weekdays.",
    ]
