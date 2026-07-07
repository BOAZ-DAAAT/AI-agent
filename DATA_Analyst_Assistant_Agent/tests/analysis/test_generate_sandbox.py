from __future__ import annotations

import pandas as pd
import pytest

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.generate import (
    AnalysisCodeError,
    execute_generated_code,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import GeneratedAnalysisCode


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


def test_file_and_builtins_escape_are_blocked() -> None:
    with pytest.raises(AnalysisCodeError):
        execute_generated_code(_code("data = open('x.txt')\nresult = {}"), _frame())
