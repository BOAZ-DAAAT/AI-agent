from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.context import build_analysis_context
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def test_context_falls_back_to_semantic_agent_feedback_without_hard_failure() -> None:
    state = OrchestrationState(
        run_id="run_001",
        user_query="고객 단위 RFM 세그먼트를 만들어줘",
        retry_context={
            "agent_feedback": {
                "analysis_agent": {
                    "reason": "고객 단위가 아니라 주문 단위로 집계함",
                    "missing_evidence": ["customer_unique_id 기준 grain"],
                    "source": "semantic",
                }
            }
        },
    )
    dataframe = pd.DataFrame({"amount": [10, 20]})

    context = build_analysis_context(state, dataframe, [])

    assert context.last_failure is not None
    assert context.last_failure["reason_code"] == "semantic_validation_failed"
    assert "고객 단위가 아니라 주문 단위로 집계함" in context.last_failure["failure_reason"]
    assert "customer_unique_id 기준 grain" in context.last_failure["failure_reason"]


def test_context_prefers_hard_failure_over_semantic_agent_feedback() -> None:
    state = OrchestrationState(
        run_id="run_001",
        user_query="매출 합계를 분석해줘",
        retry_context={
            "last_failure": {"reason_code": "method_review_failed", "failure_reason": "wrong method"},
            "agent_feedback": {
                "analysis_agent": {"reason": "should not be used", "missing_evidence": []}
            },
        },
    )
    dataframe = pd.DataFrame({"amount": [10, 20]})

    context = build_analysis_context(state, dataframe, [])

    assert context.last_failure == {"reason_code": "method_review_failed", "failure_reason": "wrong method"}


def test_context_last_failure_none_without_any_retry_signal() -> None:
    state = OrchestrationState(run_id="run_001", user_query="매출 합계를 분석해줘")
    dataframe = pd.DataFrame({"amount": [10, 20]})

    context = build_analysis_context(state, dataframe, [])

    assert context.last_failure is None
