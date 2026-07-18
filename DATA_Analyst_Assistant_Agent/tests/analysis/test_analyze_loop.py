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


def test_critic_failure_reflects_then_passes() -> None:
    gen = _FakeModel([_GOOD_CODE, _GOOD_CODE])
    crit = _FakeModel([
        CodeCritique(verdict="fail", method_issues=["wrong method"], feedback="use median"),
        CodeCritique(verdict="pass"),
    ])
    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)
    assert outcome.status == "passed"
    assert outcome.attempts == 2
    assert outcome.error_history[0]["stage"] == "critic"


def test_fatal_result_contract_failure_reflects_then_passes_before_critic() -> None:
    gen = _FakeModel([_BAD_CONTRACT_CODE, _GOOD_CODE])
    crit = _FakeModel([CodeCritique(verdict="pass")])

    outcome = run_analysis(_intent(), _context(), _df(), code_generator_model=gen, critic_model=crit)

    assert outcome.status == "passed"
    assert outcome.attempts == 2
    assert outcome.error_history[0]["stage"] == "result_contract"
    assert "evidence_tables" in outcome.error_history[0]["error"]


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
        ("execute", "started", 1),
        ("execute", "completed", 1),
        ("critic", "started", 1),
        ("critic", "completed", 1),
    ]


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


def test_deterministic_precheck_reflects_before_critic() -> None:
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
    gen = _FakeModel([wrong_code, right_code])
    crit = _FakeModel([CodeCritique(verdict="pass")])

    outcome = run_analysis(intent, context, df, code_generator_model=gen, critic_model=crit)

    assert outcome.status == "passed"
    assert outcome.attempts == 2
    assert outcome.error_history[0]["stage"] == "critic"
    assert "pre-check" in outcome.error_history[0]["error"]
    assert outcome.result["statistics"]["sum_revenue"] == 60


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


def test_repeated_critic_failure_stops_early() -> None:
    gen = _FakeModel([_GOOD_CODE, _GOOD_CODE])
    crit = _FakeModel([CodeCritique(verdict="fail", feedback="nope")] * 2)
    outcome = run_analysis(
        _intent(), _context(), _df(), code_generator_model=gen, critic_model=crit, max_attempts=2
    )
    assert outcome.status == "failed"
    assert outcome.attempts == 2
    assert outcome.critique.verdict == "fail"
    assert len(outcome.error_history) == 2
    assert outcome.early_stop_reason == "same critic failure repeated after regeneration"


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
