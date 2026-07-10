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
        "'statistics': {'sum': total}, 'limitations': []}\n"
    ),
)
_BAD_CODE = GeneratedAnalysisCode(rationale="broken", code="result = df['missing'].sum()")


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


def test_deterministic_precheck_reflects_before_critic() -> None:
    wrong_code = GeneratedAnalysisCode(
        rationale="wrong metric",
        code=(
            "total = int(df['x'].sum())\n"
            "result = {'summary': 'sum x', 'findings': ['sum x'], "
            "'statistics': {'sum_x': total}, 'limitations': []}\n"
        ),
    )
    right_code = GeneratedAnalysisCode(
        rationale="right metric",
        code=(
            "total = int(df['revenue'].sum())\n"
            "result = {'summary': 'sum revenue', 'findings': ['sum revenue'], "
            "'statistics': {'sum_revenue': total}, 'limitations': []}\n"
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


def test_repeated_critic_failure_stops_early() -> None:
    gen = _FakeModel([_GOOD_CODE, _GOOD_CODE, _GOOD_CODE])
    crit = _FakeModel([CodeCritique(verdict="fail", feedback="nope")] * 3)
    outcome = run_analysis(
        _intent(), _context(), _df(), code_generator_model=gen, critic_model=crit, max_attempts=3
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
