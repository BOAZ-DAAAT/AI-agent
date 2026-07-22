from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.analyze import run_analysis
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes import critic as critic_module
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    CodeCritique,
    GeneratedAnalysisCode,
)


class _Structured:
    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def invoke(self, _messages: object) -> object:
        return self._queue.pop(0)


class _FakeModel:
    """Returns queued structured outputs in order, ignoring the schema."""

    def __init__(self, queue: list[object]) -> None:
        self._queue = queue

    def with_structured_output(self, _schema: object) -> _Structured:
        return _Structured(self._queue)


def _intent() -> AnalysisIntent:
    return AnalysisIntent(objective="sum x", analysis_focus=["total x"], domain="general")


def _context() -> AnalysisContext:
    return AnalysisContext(user_question="sum x", goal="sum x", route_kind="simple", columns=["x"])


def _df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3]})


_GOOD_CODE = GeneratedAnalysisCode(
    rationale="sum",
    code=(
        "total = int(df['x'].sum())\n"
        "result = {'summary': f'sum={total}', 'findings': [f'sum={total}'], "
        "'statistics': {'sum': total}, 'method_decision': {'selected_method': 'sum', 'rationale': 'The objective requests a total.', 'assumptions_checked': [], 'fallbacks_considered': []}, 'limitations': []}\n"
    ),
)
_REVIEW_CODE = GeneratedAnalysisCode(
    rationale="sum with actionable review request",
    code=(
        "total = int(df['x'].sum())\n"
        "result = {\n"
        "  'summary': f'sum={total}',\n"
        "  'findings': [f'sum={total}'],\n"
        "  'statistics': {'sum': total},\n"
        "  'method_decision': {'selected_method': 'sum', 'rationale': 'The objective requests a total.', 'assumptions_checked': [], 'fallbacks_considered': []},\n"
        "  'limitations': [],\n"
        "  'review_request': {\n"
        "    'decision_type': 'metric_definition',\n"
        "    'question': 'Use sum(x) as the follow-up metric?',\n"
        "    'proposal': 'Use sum(x) as the operational metric.',\n"
        "    'rationale': ['The column is complete in this fixture.'],\n"
        "    'evidence': {'sum': total},\n"
        "    'options': [\n"
        "      {'id': 'sum', 'label': 'Use sum(x)', 'method': 'sum', 'assumptions': [], 'advantages': ['Measures total volume.'], 'limitations': ['Sensitive to scale.'], 'impact': 'Follow-up uses total volume.', 'recommended': True},\n"
        "      {'id': 'mean', 'label': 'Use average x instead', 'method': 'mean', 'assumptions': [], 'advantages': ['Normalizes by observations.'], 'limitations': ['Does not measure total volume.'], 'impact': 'Follow-up uses average value.', 'recommended': False}\n"
        "    ],\n"
        "    'recommended_option_id': 'sum',\n"
        "    'impact_if_approved': 'Follow-up analysis will use sum(x).',\n"
        "    'requires_followup_analysis': True\n"
        "  }\n"
        "}\n"
    ),
)
_BAD_CODE = GeneratedAnalysisCode(rationale="broken", code="result = df['missing'].sum()")
_BAD_CONTRACT_CODE = GeneratedAnalysisCode(
    rationale="bad result contract",
    code=(
        "result = {'summary': 'bad rows', 'findings': ['bad rows'], "
        "'statistics': {'n': len(df)}, 'limitations': [], "
        "'evidence_tables': [{'title': 'bad', 'rows': 'not rows'}]}\n"
    ),
)


def test_execution_failure_reflects_then_passes() -> None:
    gen = _FakeModel([_BAD_CODE, _GOOD_CODE])          # 1st errors, 2nd works
    crit = _FakeModel([CodeCritique(verdict="pass")])  # only called after 2nd
    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)
    assert outcome.status == "passed"
    assert outcome.attempts == 2
    assert outcome.result["statistics"]["sum"] == 6
    assert outcome.error_history[0]["stage"] == "execute"


def test_critic_failure_is_recorded_as_warning_and_passes() -> None:
    gen = _FakeModel([_GOOD_CODE])
    crit = _FakeModel([
        CodeCritique(verdict="fail", method_issues=["wrong method"], feedback="use median"),
    ])
    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)
    assert outcome.status == "passed"
    assert outcome.attempts == 1
    assert outcome.error_history == []
    assert "use median" in outcome.result["method_notes"][-1]


def test_fatal_result_contract_failure_is_recorded_as_warning_before_critic() -> None:
    gen = _FakeModel([_BAD_CONTRACT_CODE])
    crit = _FakeModel([CodeCritique(verdict="pass")])

    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)

    assert outcome.status == "passed"
    assert outcome.attempts == 1
    assert outcome.error_history == []
    assert any("evidence_tables" in note for note in outcome.result["method_notes"])


def test_progress_callback_reports_completed_stages() -> None:
    events: list[tuple[str, str, int]] = []
    outcome = run_analysis(
        _intent(),
        _context(),
        _df(),
        code_generator_model=_FakeModel([_GOOD_CODE]),
        critic_model=_FakeModel([CodeCritique(verdict="pass")]),
        progress_callback=lambda stage, status, attempt: events.append((stage, status, attempt)),
    )

    assert outcome.status == "passed"
    assert events == [
        ("generate", "started", 1),
        ("generate", "completed", 1),
        ("execute.preflight", "started", 1),
        ("execute.preflight", "completed", 1),
        ("execute", "started", 1),
        ("contract_check", "started", 1),
        ("contract_check", "completed", 1),
        ("execute", "completed", 1),
        ("critic", "started", 1),
        ("critic", "completed", 1),
    ]


def test_manual_run_recommended_returns_generated_code_without_exec() -> None:
    code = GeneratedAnalysisCode(
        rationale="long running",
        code=(
            "while True:\n"
            "    pass\n"
        ),
    )
    events: list[tuple[str, str, int]] = []
    outcome = run_analysis(
        _intent(),
        _context(),
        _df(),
        code_generator_model=_FakeModel([code]),
        critic_model=_FakeModel([]),
        progress_callback=lambda stage, status, attempt: events.append((stage, status, attempt)),
    )

    assert outcome.status == "passed"
    assert outcome.early_stop_reason == "manual_run_recommended"
    assert outcome.result["statistics"]["execution_decision"] == "manual_run_recommended"
    assert "while True" in outcome.result["generated_code"]
    assert ("execute", "skipped_manual_run", 1) in events


def test_review_required_preserves_result_without_retry() -> None:
    gen = _FakeModel([_REVIEW_CODE, _BAD_CODE])
    crit = _FakeModel([
        CodeCritique(
            verdict="review_required",
            method_issues=["proxy label needs operational review"],
            feedback="interpret proxy label cautiously",
        )
    ])

    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)

    assert outcome.status == "review_required"
    assert outcome.attempts == 1
    assert outcome.result["statistics"]["sum"] == 6
    assert outcome.result["review_request"]["question"] == "Use sum(x) as the follow-up metric?"
    assert outcome.critique.verdict == "review_required"


def test_review_required_without_actionable_request_becomes_method_note() -> None:
    gen = _FakeModel([_GOOD_CODE, _BAD_CODE])
    crit = _FakeModel([
        CodeCritique(
            verdict="review_required",
            method_issues=["small sample"],
            feedback="sample size is small",
        )
    ])

    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)

    assert outcome.status == "passed"
    assert outcome.attempts == 1
    assert outcome.result["statistics"]["sum"] == 6
    assert outcome.result["method_notes"] == ["sample size is small"]
    assert outcome.critique.verdict == "pass"


def test_deterministic_precheck_is_recorded_as_warning_before_critic() -> None:
    wrong_code = GeneratedAnalysisCode(
        rationale="wrong metric",
        code=(
                "total = int(df['x'].sum())\n"
                "result = {'summary': 'sum x', 'findings': ['sum x'], "
                "'statistics': {'sum_x': total}, 'method_decision': {'selected_method': 'sum', 'rationale': 'fixture', 'assumptions_checked': [], 'fallbacks_considered': []}, 'limitations': []}\n"
        ),
    )
    right_code = GeneratedAnalysisCode(
        rationale="right metric",
        code=(
                "total = int(df['revenue'].sum())\n"
                "result = {'summary': 'sum revenue', 'findings': ['sum revenue'], "
                "'statistics': {'sum_revenue': total}, 'method_decision': {'selected_method': 'sum', 'rationale': 'fixture', 'assumptions_checked': [], 'fallbacks_considered': []}, 'limitations': []}\n"
        ),
    )
    intent = AnalysisIntent(objective="sum revenue", metric_hints=["revenue"])
    context = AnalysisContext(
        user_question="sum revenue",
        goal="sum revenue",
        route_kind="simple",
        columns=["x", "revenue"],
        metric_hint="revenue",
    )
    df = pd.DataFrame({"x": [1, 2, 3], "revenue": [10, 20, 30]})
    gen = _FakeModel([wrong_code])
    crit = _FakeModel([CodeCritique(verdict="pass")])

    outcome = run_analysis(intent, context, df, code_generator_model=gen, critic_model=crit)

    assert outcome.status == "passed"
    assert outcome.attempts == 1
    assert outcome.error_history == []
    assert "metric:revenue" in outcome.result["method_notes"][-1]
    assert outcome.result["statistics"]["sum_x"] == 6


def test_contract_metric_support_failure_retries_generation() -> None:
    wrong_code = GeneratedAnalysisCode(
        rationale="row-level mean only",
        code=(
            "avg_days = float(df['delivery_days'].mean())\n"
            "result = {'summary': 'avg delivery', 'findings': ['avg delivery'], "
            "'statistics': {'avg_delivery_days': avg_days}, "
            "'method_decision': {'selected_method': 'mean', 'rationale': 'fixture'}, "
            "'limitations': []}\n"
        ),
    )
    right_code = GeneratedAnalysisCode(
        rationale="contract grain aggregation",
        code=(
            "seller_month = df.groupby(['seller_id', 'order_month'])['delivery_days'].mean().reset_index()\n"
            "result = {'summary': 'seller month delivery', 'findings': ['seller month delivery'], "
            "'statistics': {'seller_month_rows': int(len(seller_month))}, "
            "'method_decision': {'selected_method': 'groupby seller_id order_month', 'rationale': 'Matches metric_support calculation_grain.'}, "
            "'limitations': []}\n"
        ),
    )
    intent = AnalysisIntent(objective="seller monthly delivery trend")
    context = AnalysisContext(
        user_question="seller monthly delivery trend",
        goal="seller monthly delivery trend",
        route_kind="comprehensive",
        columns=["seller_id", "order_id", "order_month", "delivery_days"],
        analysis_data_contract={
            "row_grain": "seller_id x order_id",
            "grain_columns": ["seller_id", "order_id"],
            "metric_support": [{
                "metric_name": "monthly_avg_delivery_days",
                "calculation_grain": ["seller_id", "order_month"],
                "required_mart_columns": ["seller_id", "order_month", "delivery_days"],
                "downstream_calculation": "Group by seller_id and order_month before averaging delivery_days.",
            }],
        },
    )
    df = pd.DataFrame({
        "seller_id": ["s1", "s1", "s2"],
        "order_id": ["o1", "o2", "o3"],
        "order_month": ["2024-01", "2024-02", "2024-01"],
        "delivery_days": [3, 5, 7],
    })
    gen = _FakeModel([wrong_code, right_code])
    crit = _FakeModel([CodeCritique(verdict="pass")])

    outcome = run_analysis(intent, context, df, code_generator_model=gen, critic_model=crit)

    assert outcome.status == "passed"
    assert outcome.attempts == 2
    assert outcome.error_history[0]["stage"] == "critic"
    assert "contract_metric_support:monthly_avg_delivery_days" in outcome.error_history[0]["error"]
    assert outcome.result["statistics"]["seller_month_rows"] == 3


def test_partial_time_coverage_becomes_method_note_not_precheck_failure() -> None:
    intent = AnalysisIntent(
        objective="monthly revenue by category",
        metric_hints=["revenue"],
        dimension_hints=["category"],
        time_column="order_month",
        is_time_based=True,
        time_grain="M",
    )
    context = AnalysisContext(
        user_question="monthly revenue by category",
        goal="monthly revenue by category",
        route_kind="comprehensive",
        columns=["order_month", "category", "revenue"],
    )
    code = GeneratedAnalysisCode(
        rationale="category revenue",
        code=(
            "by_category = df.groupby('category')['revenue'].sum()\n"
            "result = {'summary': 'category revenue', 'findings': ['category revenue'], "
            "'statistics': {'category_revenue': by_category.to_dict()}, "
            "'method_decision': {'selected_method': 'grouped sum', 'rationale': 'fixture'}, "
            "'limitations': []}\n"
        ),
    )
    result = {
        "summary": "category revenue",
        "findings": ["category revenue"],
        "statistics": {"category_revenue": {"A": 10}},
        "method_decision": {"selected_method": "grouped sum", "rationale": "fixture"},
        "limitations": [],
    }

    critique = critic_module.deterministic_precheck(intent, context, code, result)

    assert critique is None
    assert result["method_notes"] == [
        "Time-based analysis only partially covered requested signals: time_column:order_month"
    ]


def test_missing_coverage_remains_precheck_failure() -> None:
    intent = AnalysisIntent(objective="sum revenue", metric_hints=["revenue"])
    context = AnalysisContext(
        user_question="sum revenue",
        goal="sum revenue",
        route_kind="simple",
        columns=["x", "revenue"],
        metric_hint="revenue",
    )
    result = {
        "summary": "sum x",
        "findings": ["sum x"],
        "statistics": {"sum_x": 6},
        "method_decision": {"selected_method": "sum", "rationale": "fixture"},
        "limitations": [],
    }

    critique = critic_module.deterministic_precheck(intent, context, _GOOD_CODE, result)

    assert critique is not None
    assert critique.verdict == "fail"
    assert any("metric:revenue" in issue for issue in critique.method_issues)


def test_numeric_only_review_options_are_rejected_before_critic() -> None:
    result = {
        "summary": "ok",
        "findings": ["ok"],
        "statistics": {"n": 3},
        "method_decision": {"selected_method": "summary", "rationale": "fixture"},
        "limitations": [],
        "review_request": {
            "decision_type": "threshold",
            "question": "Which threshold?",
            "proposal": "Choose a threshold.",
            "rationale": ["fixture"],
            "evidence": {"n": 3},
            "options": [
                {"id": "30", "label": "30", "method": "threshold", "impact": "same", "recommended": True},
                {"id": "50", "label": "50", "method": "threshold", "impact": "same", "recommended": False},
            ],
            "recommended_option_id": "30",
            "impact_if_approved": "Apply selected threshold.",
            "requires_followup_analysis": True,
        },
    }

    critique = critic_module.deterministic_precheck(_intent(), _context(), _GOOD_CODE, result)

    assert critique is not None
    assert critique.verdict == "fail"
    assert any("numeric thresholds" in issue for issue in critique.method_issues)


def test_critic_failure_becomes_warning_result() -> None:
    gen = _FakeModel([_GOOD_CODE])
    crit = _FakeModel([CodeCritique(verdict="fail", feedback="nope")])
    outcome = run_analysis(
        _intent(), _context(), _df(), code_generator_model=gen, critic_model=crit, max_attempts=2
    )
    assert outcome.status == "passed"
    assert outcome.attempts == 1
    assert outcome.critique.verdict == "pass"
    assert outcome.error_history == []
    assert "nope" in outcome.result["method_notes"][-1]


def test_critic_uses_dedicated_model_env(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_get_chat_model(**kwargs):
        seen.update(kwargs)
        return _FakeModel([CodeCritique(verdict="pass")])

    monkeypatch.setattr(critic_module, "get_chat_model", fake_get_chat_model)

    verdict = critic_module.critique_analysis_code(
        _intent(),
        _GOOD_CODE,
        {"summary": "ok", "findings": ["ok"], "statistics": {"sum": 6}, "limitations": []},
    )

    assert verdict.verdict == "pass"
    assert seen["model_env"] == "ANALYSIS_CRITIC_MODEL"
