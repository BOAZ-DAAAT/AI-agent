from __future__ import annotations

import pandas as pd
from pydantic import ValidationError

from DATA_Analyst_Assistant_Agent.agents.analysis.graph import run_analysis_workflow
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisIntent,
    AnalysisResult,
    CodeCritique,
    GeneratedAnalysisCode,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState


def _validation_error_for(text: str) -> ValidationError:
    try:
        GeneratedAnalysisCode.model_validate_json(text)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


class _Structured:
    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def invoke(self, _messages: object) -> object:
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


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

_RECOVERABLE_CONTRACT_CODE = GeneratedAnalysisCode(
    rationale="recoverable result contract aliases",
    code=(
        "result = {\n"
        "  'summary': 'recoverable contract payload',\n"
        "  'findings': ['recoverable contract payload'],\n"
        "  'statistics': {'n': len(df)},\n"
        "  'method_decision': {'selected_method': 'summary', 'rationale': 'fixture', 'assumptions_checked': [], 'fallbacks_considered': []},\n"
        "  'limitations': [],\n"
        "  'hypothesis_tests': [{\n"
        "    'hypothesis': 'revenue is associated with fixture order',\n"
        "    'test_name': 'spearman',\n"
        "    'n': 87448.08854062065,\n"
        "    'decision': 'inconclusive'\n"
        "  }],\n"
        "  'evidence_tables': [{'name': 'risk_segment_summary', 'columns': ['metric', 'value'], 'rows': [{'metric': 'n', 'value': len(df)}]}]\n"
        "}\n"
    ),
)


_BAD_IMPORT_CODE = GeneratedAnalysisCode(
    rationale="uses a disallowed import",
    code="import os\nresult = {}\n",
)

_BAD_CONTRACT_CODE = GeneratedAnalysisCode(
    rationale="produces a malformed evidence_tables shape",
    code=(
        "result = {'summary': 'x', 'findings': ['x'], 'statistics': {'n': 1}, "
        "'limitations': [], 'evidence_tables': 'not-a-list'}\n"
    ),
)


def test_graph_repeated_generation_parse_failure_is_labeled_generation_failed() -> None:
    # run 019f7a58: openai/gpt-5.4-mini returned unparseable structured JSON
    # (too verbose, cut off / trailing noise) on both attempts, which used
    # to crash the whole analyze node as an uncaught exception instead of
    # feeding back into the retry loop like execute/critic failures do.
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance")])
    unrecoverable = _validation_error_for("this is not json at all")
    generate = _FakeModel([unrecoverable, unrecoverable])
    critic = _FakeModel([])  # must never be reached

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "generation_failed"
    assert parsed.status == "failed"
    assert parsed.error_history
    assert parsed.error_history[-1]["stage"] == "generate"
    execution_check = next(c for c in checks if c.name == "analysis_code_executed")
    assert not execution_check.passed
    assert "[generate]" in execution_check.detail


def test_graph_blocks_sql_datamart_with_incomplete_analysis_contract() -> None:
    state = _state()
    state.plan = AnalysisPlan(
        goal="seller monthly delivery trend",
        target_table="analytics.seller_month_delivery",
        generated_sql="SELECT seller_id, month, avg_delivery_days FROM mart",
        analysis_data_contract={
            "target_table": "analytics.seller_month_delivery",
            "row_grain": "",
            "generated_sql": "SELECT seller_id, month, avg_delivery_days FROM mart",
            "derived_columns": [{"output_column": "month", "source_columns": ["order_purchase_timestamp"]}],
        },
    )
    classify = _FakeModel([AssertionError("classify should not be called")])

    result, checks, terminal = run_analysis_workflow(
        state,
        pd.DataFrame({"seller_id": ["s1"], "month": ["2024-01-01"], "avg_delivery_days": [3.0]}),
        [],
        planner_model=classify,
        code_generator_model=_FakeModel([]),
        critic_model=_FakeModel([]),
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "analysis_contract_invalid"
    assert parsed.status == "failed"
    assert any("row_grain" in limitation for limitation in parsed.limitations)


def test_graph_repeated_execute_failure_is_labeled_execution_failed() -> None:
    # run 019f7970: every attempt failed at execute (blocked import), but
    # terminal_reason still said "method_review_failed" -- the critic was
    # never even called. This pins the corrected stage-specific label.
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance")])
    generate = _FakeModel([_BAD_IMPORT_CODE, _BAD_IMPORT_CODE])
    critic = _FakeModel([])  # must never be reached

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "execution_failed"
    assert parsed.status == "failed"
    assert parsed.error_history
    assert parsed.error_history[-1]["stage"] == "execute"
    assert "not allowed" in parsed.error_history[-1]["error"]
    assert any(
        "failed at the execute stage" in limitation for limitation in parsed.limitations
    )
    execution_check = next(c for c in checks if c.name == "analysis_code_executed")
    assert not execution_check.passed
    assert "not allowed" in execution_check.detail


def test_graph_repeated_result_contract_failure_is_labeled_result_contract_failed() -> None:
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance")])
    generate = _FakeModel([_BAD_CONTRACT_CODE, _BAD_CONTRACT_CODE])
    critic = _FakeModel([])  # must never be reached

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "result_contract_failed"
    assert parsed.status == "failed"
    assert parsed.error_history
    assert parsed.error_history[-1]["stage"] == "result_contract"


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


def test_graph_normalizes_recoverable_result_contract_payload() -> None:
    classify = _FakeModel([AnalysisIntent(objective="sum revenue", domain="finance", metric_hints=["revenue"])])
    generate = _FakeModel([_RECOVERABLE_CONTRACT_CODE])
    critic = _FakeModel([CodeCritique(verdict="pass")])

    result, checks, terminal = run_analysis_workflow(
        _state(), _df(), [],
        planner_model=classify,
        code_generator_model=generate,
        critic_model=critic,
    )

    parsed = AnalysisResult.model_validate(result)
    assert terminal == "validated_result"
    assert parsed.evidence_tables[0].title == "risk_segment_summary"
    assert parsed.hypothesis_tests[0].n is None
    assert "Normalized evidence_tables[0].name to title." in parsed.method_notes
    assert (
        "Cleared hypothesis_tests[0].n because sample size was a non-integer float."
        in parsed.method_notes
    )
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
