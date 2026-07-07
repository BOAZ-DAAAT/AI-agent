from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.graph import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisIntent,
    AnalysisResult,
    CodeCritique,
    GeneratedAnalysisCode,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


class _Structured:
    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def invoke(self, _messages: object) -> object:
        return self._queue.pop(0)


class _FakeModel:
    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def with_structured_output(self, _schema: object) -> _Structured:
        return _Structured(self._queue)


def _state() -> OrchestrationState:
    return OrchestrationState(run_id="run-1", user_query="sum revenue", goal="sum revenue")


def _df() -> pd.DataFrame:
    return pd.DataFrame({"revenue": [10, 20, 30]})


_GOOD_CODE = GeneratedAnalysisCode(
    rationale="sum revenue",
    code=(
        "total = int(df['revenue'].sum())\n"
        "result = {'summary': f'total revenue {total}', 'findings': [f'total revenue {total}'], "
        "'statistics': {'total_revenue': total}, 'limitations': ['single run']}\n"
    ),
)


def test_graph_produces_valid_analysis_result() -> None:
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance", metric_hints=["revenue"])])
    generate = _FakeModel([_GOOD_CODE])
    critic = _FakeModel([CodeCritique(verdict="pass")])

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)  # downstream contract stays valid
    assert terminal == "validated_result"
    assert parsed.evidence[0].statistics["total_revenue"] == 60
    assert parsed.intent.domain == "finance"
    assert parsed.generated_code
    assert parsed.codegen_attempts == 1
    assert parsed.chart_status == "not_needed"  # chart branch ran, nothing to inspect
    assert all(check.passed for check in checks)


def test_graph_failed_review_marks_human_review() -> None:
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance")])
    generate = _FakeModel([_GOOD_CODE, _GOOD_CODE, _GOOD_CODE])
    critic = _FakeModel([CodeCritique(verdict="fail", feedback="wrong")] * 3)

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "method_review_failed"
    assert parsed.human_review.required is True
    assert any(not check.passed for check in checks)
