from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.generate import (
    AnalysisCodeError,
    execute_generated_code,
    generate_analysis_code,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes import generate as generate_module
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    AnalysisSelectionResponse,
    GeneratedAnalysisCode,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.1, 5.9, 8.2]})


def _code(body: str, imports: str = "") -> GeneratedAnalysisCode:
    return GeneratedAnalysisCode(rationale="test", imports=imports, code=body)


def test_statsmodels_import_is_allowed_and_runs() -> None:
    body = (
        "model = sm.OLS(df['y'], sm.add_constant(df['x'])).fit()\n"
        "result = {\n"
        "  'summary': 'ols slope estimated',\n"
        "  'findings': ['slope computed'],\n"
        "  'statistics': {'slope': float(model.params.iloc[1])},\n"
        "  'limitations': ['small n'],\n"
        "}\n"
    )
    out = execute_generated_code(_code(body, imports="import statsmodels.api as sm"), _frame())
    assert out["statistics"]["slope"] == pytest.approx(2.06, abs=0.1)


def test_disallowed_import_is_blocked() -> None:
    code = _code("import os\nresult = {}", imports="")
    with pytest.raises(AnalysisCodeError) as excinfo:
        execute_generated_code(code, _frame())
    assert "not allowed" in str(excinfo.value)


def test_result_must_be_dict_with_required_keys() -> None:
    with pytest.raises(AnalysisCodeError) as no_dict:
        execute_generated_code(_code("result = 5"), _frame())
    assert "must set `result`" in str(no_dict.value)

    with pytest.raises(AnalysisCodeError) as missing:
        execute_generated_code(_code("result = {'summary': 'x'}"), _frame())
    assert "missing required keys" in str(missing.value)


def test_primitives_namespace_is_available() -> None:
    # We don't run a heavy tool here; just prove the namespace is injected.
    body = (
        "names = sorted(primitives.keys())\n"
        "result = {'summary': 'ok', 'findings': names, 'statistics': {'n': len(names)}, 'limitations': []}\n"
    )
    out = execute_generated_code(_code(body), _frame())
    assert "analyze_survival" in out["findings"]
    assert out["statistics"]["n"] >= 1


def test_records_are_materialized_only_when_a_primitive_is_called(monkeypatch) -> None:
    calls = 0

    def fake_records(dataframe: pd.DataFrame):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(generate_module, "_records_from_dataframe", fake_records)
    code = _code(
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {'n': len(df)}, 'limitations': []}"
    )

    execute_generated_code(code, _frame())

    assert calls == 0


def test_file_and_builtins_escape_are_blocked() -> None:
    with pytest.raises(AnalysisCodeError):
        execute_generated_code(_code("data = open('x.txt')\nresult = {}"), _frame())


class _CapturingCodeModel:
    def __init__(self) -> None:
        self.messages = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        return GeneratedAnalysisCode(
            rationale="retry",
            code=(
                "result = {'summary': 'ok', 'findings': [], "
                "'statistics': {}, 'limitations': []}"
            ),
        )


def test_generate_prompt_keeps_supervisor_failure_separate_from_critic_feedback() -> None:
    context = AnalysisContext(
        user_question="매출 분석",
        goal="매출 분석",
        route_kind="simple",
        columns=["amount"],
        last_failure={
            "reason_code": "method_review_failed",
            "failure_reason": "wrong method",
        },
    )
    model = _CapturingCodeModel()

    generate_analysis_code(
        AnalysisIntent(objective="매출 분석"),
        context,
        model=model,
        feedback="critic says aggregate first",
    )

    prompt = model.messages[1].content
    assert "Previous Supervisor failure:" in prompt
    assert "method_review_failed" in prompt
    assert "wrong method" in prompt
    assert "A previous attempt was rejected" in prompt
    assert "critic says aggregate first" in prompt


def test_generate_prompt_includes_free_text_selection_as_a_binding_constraint() -> None:
    context = AnalysisContext(
        user_question="analyze revenue",
        goal="analyze revenue",
        route_kind="simple",
        columns=["amount"],
        selection_response=AnalysisSelectionResponse(free_text="Compare medians, not totals."),
    )
    model = _CapturingCodeModel()

    generate_analysis_code(AnalysisIntent(objective="analyze revenue"), context, model=model)

    assert "binding constraint" in model.messages[1].content
    assert "Compare medians, not totals." in model.messages[1].content


def test_generate_prompt_exposes_eda_candidates_without_requiring_all_of_them() -> None:
    context = AnalysisContext(
        user_question="compare category revenue",
        goal="compare category revenue",
        route_kind="comprehensive",
        columns=["category", "revenue"],
        eda_candidate_insights=["Category B has higher observed revenue."],
        eda_candidate_hypotheses=["Category B revenue is higher than category A."],
    )
    model = _CapturingCodeModel()

    generate_analysis_code(AnalysisIntent(objective="compare category revenue"), context, model=model)

    system_prompt = model.messages[0].content
    human_prompt = model.messages[1].content
    assert "exploratory candidate hints" in system_prompt
    assert "You may add new analysis hypotheses" in system_prompt
    assert "EDA exploratory candidates" in human_prompt
    assert "Category B revenue is higher than category A." in human_prompt
    assert "Do not test or report every EDA candidate by default" in human_prompt


def test_selection_response_requires_one_choice_or_free_text() -> None:
    with pytest.raises(ValueError):
        AnalysisSelectionResponse()
    with pytest.raises(ValueError):
        AnalysisSelectionResponse(selected_option_id="total", free_text="Use totals")
    assert AnalysisSelectionResponse(selected_option_id="total").constraint_text().endswith("total")
