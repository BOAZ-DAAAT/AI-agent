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
        "'statistics': {'total_revenue': total}, 'method_decision': {'selected_method': 'sum', 'rationale': 'The objective requests total revenue.', 'assumptions_checked': [], 'fallbacks_considered': []}, 'limitations': ['single run']}\n"
    ),
)

_HYPOTHESIS_CODE = GeneratedAnalysisCode(
    rationale="sum revenue with hypothesis test",
    code=(
        "total = int(df['revenue'].sum())\n"
        "result = {\n"
        "  'summary': f'total revenue {total}',\n"
        "  'findings': [f'total revenue {total}'],\n"
        "  'statistics': {'total_revenue': total},\n"
        "  'method_decision': {'selected_method': 'sum', 'rationale': 'The objective requests total revenue.', 'assumptions_checked': [], 'fallbacks_considered': []},\n"
        "  'hypothesis_tests': [{\n"
        "    'hypothesis': 'revenue is positive',\n"
        "    'test_name': 'one-sample descriptive check',\n"
        "    'null_hypothesis': 'total revenue is zero',\n"
        "    'alternative_hypothesis': 'total revenue is positive',\n"
        "    'statistic': float(total), 'p_value': 0.01, 'effect_size': 1.0,\n"
        "    'n': int(len(df)), 'decision': 'supported', 'caveats': ['fixture']\n"
        "  }],\n"
        "  'evidence_tables': [{'title': 'summary', 'columns': ['metric', 'value'], 'rows': [{'metric': 'total_revenue', 'value': total}]}],\n"
        "  'interpretation': ['Revenue is positive in this fixture.'],\n"
        "  'review_request': {\n"
        "    'decision_type': 'metric_definition',\n"
        "    'question': 'Use total revenue as the follow-up metric?',\n"
        "    'proposal': 'Use total revenue as the operational metric for the next analysis.',\n"
        "    'rationale': ['The metric is directly computed from the requested revenue column.'],\n"
        "    'evidence': {'total_revenue': total},\n"
        "    'options': [\n"
        "      {'id': 'total', 'label': 'Use total revenue', 'method': 'sum', 'assumptions': [], 'advantages': ['Measures overall scale.'], 'limitations': ['Favors larger groups.'], 'impact': 'Follow-up compares total revenue.', 'recommended': True},\n"
        "      {'id': 'average', 'label': 'Use average revenue', 'method': 'mean', 'assumptions': [], 'advantages': ['Normalizes group size.'], 'limitations': ['Does not measure total scale.'], 'impact': 'Follow-up compares average revenue.', 'recommended': False}\n"
        "    ],\n"
        "    'recommended_option_id': 'total',\n"
        "    'impact_if_approved': 'Follow-up analysis will use total revenue.',\n"
        "    'requires_followup_analysis': True\n"
        "  },\n"
        "  'limitations': ['single run']\n"
        "}\n"
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
    assert parsed.answer_coverage.coverage_status == "full"
    assert parsed.answer_coverage.used_metrics == ["revenue"]
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


def test_graph_review_required_is_validated_result_with_hypothesis_tests() -> None:
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance", metric_hints=["revenue"])])
    generate = _FakeModel([_HYPOTHESIS_CODE])
    critic = _FakeModel([
        CodeCritique(
            verdict="review_required",
            method_issues=["operational interpretation needs review"],
            feedback="review operational interpretation",
        )
    ])

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "validated_result"
    assert parsed.status == "review_required"
    assert parsed.human_review.required is True
    assert parsed.human_review.reason == "Use total revenue as the follow-up metric?"
    assert parsed.review_request is not None
    assert parsed.review_request.recommended_option_id == "total"
    assert parsed.evidence[0].status == "review_required"
    assert parsed.hypothesis_tests[0].decision == "supported"
    assert parsed.evidence_tables[0].title == "summary"
    assert all(check.passed for check in checks)


def test_graph_non_actionable_review_note_does_not_require_review() -> None:
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance", metric_hints=["revenue"])])
    generate = _FakeModel([_GOOD_CODE])
    critic = _FakeModel([CodeCritique(verdict="review_required", feedback="sample size is small")])

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "validated_result"
    assert parsed.status == "success"
    assert parsed.human_review.required is False
    assert parsed.review_request is None
    assert parsed.method_notes == ["sample size is small"]
    assert all(check.passed for check in checks)
