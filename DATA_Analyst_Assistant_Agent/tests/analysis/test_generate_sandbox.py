from __future__ import annotations

import pandas as pd
import pytest

from pydantic import ValidationError

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.generate import (
    AnalysisCodeError,
    AnalysisGenerationError,
    _extract_json_object,
    execute_generated_code,
    generate_analysis_code,
    inspect_generated_code,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes import generate as generate_module
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    AnalysisSelectionResponse,
    GeneratedAnalysisCode,
    ReviewRequest,
)


def _validation_error_for(text: str) -> ValidationError:
    try:
        GeneratedAnalysisCode.model_validate_json(text)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


class _RaisingStructuredModel:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def invoke(self, _messages: object) -> object:
        raise self._exc


class _RaisingModel:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def with_structured_output(self, _schema: object) -> _RaisingStructuredModel:
        return _RaisingStructuredModel(self._exc)


def _minimal_context() -> AnalysisContext:
    return AnalysisContext(
        user_question="analyze revenue",
        goal="analyze revenue",
        route_kind="simple",
        columns=["amount"],
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


_GOOD_JSON = '{"rationale":"r","imports":"import pandas as pd","code":"result = {}"}'


def test_extract_json_object_strips_markdown_fence() -> None:
    fenced = f"```json\n{_GOOD_JSON}\n```"
    assert _extract_json_object(fenced) == _GOOD_JSON


def test_extract_json_object_strips_trailing_prose() -> None:
    trailed = f"{_GOOD_JSON}\nHope this helps!"
    assert _extract_json_object(trailed) == _GOOD_JSON


def test_extract_json_object_leaves_clean_json_untouched() -> None:
    assert _extract_json_object(_GOOD_JSON) == _GOOD_JSON


def test_generate_analysis_code_recovers_from_fenced_response() -> None:
    # run 019f7a58: openai/gpt-5.4-mini (via OpenRouter) returned a JSON
    # object wrapped in noise, which crashed structured-output parsing
    # outright. A response that is otherwise valid should not require a
    # full extra model round-trip to fix.
    exc = _validation_error_for(f"```json\n{_GOOD_JSON}\n```")
    model = _RaisingModel(exc)

    code = generate_analysis_code(
        AnalysisIntent(objective="analyze revenue"), _minimal_context(), model=model
    )

    assert code.rationale == "r"
    assert code.code == "result = {}"


def test_generate_analysis_code_raises_generation_error_when_unrecoverable() -> None:
    exc = _validation_error_for("this is not json at all")
    model = _RaisingModel(exc)

    with pytest.raises(AnalysisGenerationError):
        generate_analysis_code(
            AnalysisIntent(objective="analyze revenue"), _minimal_context(), model=model
        )


def test_timestamp_strftime_does_not_trip_the_import_guard() -> None:
    # pandas Timestamp.strftime() imports the stdlib `time` module internally
    # even when the generated code never writes `import time` itself
    # (run 019f7970: every retry failed identically because of this).
    body = (
        "label = df['ts'].iloc[0].strftime('%Y-%m')\n"
        "result = {'summary': label, 'findings': [label], 'statistics': {'n': 1}, 'limitations': []}\n"
    )
    frame = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    out = execute_generated_code(_code(body), frame)
    assert out["summary"] == "2024-01"


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


def test_preflight_allows_simple_groupby() -> None:
    code = _code(
        "summary = df.groupby('category')['value'].sum().to_dict()\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': summary, 'limitations': []}"
    )
    frame = pd.DataFrame({"category": ["a", "a", "b"], "value": [1, 2, 3]})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "auto_run"
    assert plan.risk_level == "low"


def test_preflight_allows_small_statsmodels_ols() -> None:
    code = _code(
        "model = sm.OLS(df['y'], sm.add_constant(df[['x']])).fit()\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {'n': len(df)}, 'limitations': []}",
        imports="import statsmodels.api as sm",
    )
    frame = pd.DataFrame({"x": range(50), "y": range(50)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "auto_run"


def test_preflight_allows_small_kmeans_or_pca() -> None:
    code = _code(
        "labels = KMeans(n_clusters=2, random_state=0).fit_predict(df[['x', 'y']])\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {'n': int(len(labels))}, 'limitations': []}",
        imports="from sklearn.cluster import KMeans",
    )
    frame = pd.DataFrame({"x": range(100), "y": range(100)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "auto_run"


def test_preflight_flags_grid_search() -> None:
    code = _code(
        "search = GridSearchCV(model, {'n_estimators': [10, 100]}).fit(X, y)\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {}, 'limitations': []}",
        imports="from sklearn.model_selection import GridSearchCV",
    )

    plan = inspect_generated_code(code, _frame())

    assert plan.decision == "manual_run_recommended"
    assert any("search" in reason for reason in plan.reasons)


def test_preflight_flags_large_fit() -> None:
    code = _code(
        "model = SomeEstimator().fit(df[['x']], df['y'])\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {}, 'limitations': []}"
    )
    frame = pd.DataFrame({"x": range(100_001), "y": range(100_001)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "manual_run_recommended"
    assert any("large" in reason for reason in plan.reasons)


def test_preflight_flags_while_loop() -> None:
    code = _code(
        "while True:\n"
        "    break\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {}, 'limitations': []}"
    )

    plan = inspect_generated_code(code, _frame())

    assert plan.decision == "manual_run_recommended"
    assert any("while" in reason for reason in plan.reasons)


def test_preflight_allows_independent_comprehensions_on_non_small_dataframe() -> None:
    code = _code(
        "top_x = [str(value) for value in df['x'].head(5)]\n"
        "top_y = [str(value) for value in df['y'].head(5)]\n"
        "result = {'summary': 'ok', 'findings': top_x + top_y, 'statistics': {'n': len(df)}, 'limitations': []}"
    )
    frame = pd.DataFrame({"x": range(20_000), "y": range(20_000)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "auto_run"


def test_preflight_does_not_count_for_in_strings_or_comments() -> None:
    code = _code(
        "# for seller in sellers; for month in months\n"
        "note = 'for display only, not a loop for execution'\n"
        "result = {'summary': note, 'findings': [note], 'statistics': {'n': len(df)}, 'limitations': []}"
    )
    frame = pd.DataFrame({"x": range(20_000), "y": range(20_000)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "auto_run"


def test_preflight_flags_actual_nested_loop_on_non_small_dataframe() -> None:
    code = _code(
        "pairs = []\n"
        "for seller in df['x'].head(10):\n"
        "    for month in df['y'].head(10):\n"
        "        pairs.append((seller, month))\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {'n': len(pairs)}, 'limitations': []}"
    )
    frame = pd.DataFrame({"x": range(20_000), "y": range(20_000)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "manual_run_recommended"
    assert any("nested loop" in reason for reason in plan.reasons)


def test_preflight_flags_dataframe_row_iteration_on_non_small_dataframe() -> None:
    code = _code(
        "total = 0\n"
        "for _, row in df.iterrows():\n"
        "    total += row['x']\n"
        "result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {'total': int(total)}, 'limitations': []}"
    )
    frame = pd.DataFrame({"x": range(20_000), "y": range(20_000)})

    plan = inspect_generated_code(code, frame)

    assert plan.decision == "manual_run_recommended"
    assert any("row iteration" in reason for reason in plan.reasons)


def test_execute_generated_code_subprocess_success() -> None:
    out = execute_generated_code(
        _code("result = {'summary': 'ok', 'findings': ['ok'], 'statistics': {'n': len(df)}, 'limitations': []}"),
        _frame(),
        isolated=True,
        timeout_seconds=30,
    )

    assert out["statistics"]["n"] == len(_frame())


def test_execute_generated_code_subprocess_timeout() -> None:
    with pytest.raises(AnalysisCodeError, match="exceeded execution timeout"):
        execute_generated_code(
            _code("while True:\n    pass"),
            _frame(),
            isolated=True,
            timeout_seconds=0.5,
        )


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


def test_generate_prompt_includes_analysis_data_contract() -> None:
    context = AnalysisContext(
        user_question="analyze mart",
        goal="analyze mart",
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
    model = _CapturingCodeModel()

    generate_analysis_code(AnalysisIntent(objective="analyze mart"), context, model=model)

    prompt = model.messages[1].content
    assert "Declared upstream SQL/datamart analysis contract" in prompt
    assert "monthly_avg_delivery_days" in prompt
    assert "Group by seller_id and order_month" in prompt


def test_generate_prompt_includes_full_selected_option_as_binding_constraint() -> None:
    request = ReviewRequest.model_validate(
        {
            "question": "대표값은?",
            "proposal": "대표값 선택",
            "options": [
                {
                    "id": "mean",
                    "label": "평균",
                    "method": "산술 평균",
                    "impact": "평균을 보고합니다.",
                    "recommended": False,
                },
                {
                    "id": "median",
                    "label": "중앙값",
                    "method": "50% 분위수",
                    "assumptions": ["순서 통계량 사용 가능"],
                    "advantages": ["극단값에 강건함"],
                    "limitations": ["합계와 직접 연결되지 않음"],
                    "impact": "중앙값과 IQR을 보고합니다.",
                    "recommended": True,
                },
            ],
            "recommended_option_id": "median",
        }
    )
    context = AnalysisContext(
        user_question="매출 분석",
        goal="매출 분석",
        route_kind="simple",
        columns=["amount"],
        review_request=request,
        selection_response=AnalysisSelectionResponse(selected_option_id="median"),
    )
    model = _CapturingCodeModel()

    generate_analysis_code(AnalysisIntent(objective="매출 분석"), context, model=model)

    prompt = model.messages[1].content
    for expected in (
        "median",
        "중앙값",
        "50% 분위수",
        "순서 통계량 사용 가능",
        "극단값에 강건함",
        "합계와 직접 연결되지 않음",
        "중앙값과 IQR을 보고합니다.",
    ):
        assert expected in prompt


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


def test_generate_prompt_specifies_result_contract_shapes() -> None:
    context = AnalysisContext(
        user_question="compare price bands",
        goal="compare price bands",
        route_kind="simple",
        columns=["price_band", "review_score"],
    )
    model = _CapturingCodeModel()

    generate_analysis_code(AnalysisIntent(objective="compare price bands"), context, model=model)

    system_prompt = model.messages[0].content
    assert "title (str), columns (list[str]), rows (list[dict])" in system_prompt
    assert "do not use `name`" in system_prompt
    assert "The `n` value MUST be an integer sample" in system_prompt


def test_selection_response_requires_one_choice_or_free_text() -> None:
    with pytest.raises(ValueError):
        AnalysisSelectionResponse()
    with pytest.raises(ValueError):
        AnalysisSelectionResponse(selected_option_id="total", free_text="Use totals")
    assert AnalysisSelectionResponse(selected_option_id="total").constraint_text().endswith("total")
