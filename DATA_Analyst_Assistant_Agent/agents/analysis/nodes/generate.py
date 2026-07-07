"""Generate node: LLM writes analysis code, executed in an opened sandbox.

This is the branch endpoint that replaces fixed tool code. The LLM produces a
``GeneratedAnalysisCode`` (structured, so no markdown-fence parsing), and the
code runs in a sandbox that — unlike the legacy codegen path — allows a small
allowlist of statistical libraries (statsmodels/scipy/sklearn/lifelines) and
exposes the vetted heavy tools as callable primitives so the model composes
them instead of re-deriving PyMC/lifelines math.

The sandbox is a soft guard (import allowlist + builtins allowlist), not full
isolation; hard isolation is the Docker executor follow-up.
"""

from __future__ import annotations

import builtins
from typing import Any, Callable

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from DATA_Analyst_Assistant_Agent.agents.analysis.prompts.domains import domain_framing
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    GeneratedAnalysisCode,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.tools import ANALYSIS_TOOLS
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


class AnalysisCodeError(RuntimeError):
    """Raised when generated code fails to produce a valid result dict."""


# Vetted heavy methods generated code may call instead of reimplementing.
_PRIMITIVE_TOOLS: tuple[str, ...] = (
    "run_bayesian_mmm",
    "estimate_probabilistic_clv",
    "analyze_survival",
    "analyze_geospatial_hotspots",
    "optimize_business_allocation",
)

# Import roots the sandbox permits. Anything else (os, sys, subprocess, ...) is
# refused at import time.
_ALLOWED_IMPORT_ROOTS = frozenset({
    "pandas", "numpy", "math", "statistics", "datetime", "collections",
    "itertools", "functools", "re", "json",
    "statsmodels", "scipy", "sklearn", "lifelines",
})

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hasattr", "int", "isinstance", "issubclass",
    "len", "list", "map", "max", "min", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    # common exceptions generated code may raise/catch
    "Exception", "ValueError", "KeyError", "TypeError", "ZeroDivisionError",
)


REQUIRED_RESULT_KEYS = ("summary", "findings", "statistics", "limitations")


GENERATE_SYSTEM_PROMPT = """You write one small, auditable Python analysis script.

Rules:
- Operate on the pandas DataFrame named `df` (already loaded). `pd` and `np` are available.
- You MAY import from: pandas, numpy, math, statistics, scipy, statsmodels,
  sklearn, lifelines, datetime, collections, itertools, functools, re, json.
  No other imports (no os/sys/file/network access).
- For heavy specialized methods, call the provided primitives instead of
  reimplementing them: {primitives}. Each takes keyword column arguments.
- Choose the statistically valid method for the stated objective and domain.
- If the data is time-based, resample/aggregate at the given time_grain before
  fitting a trend. If there are too few periods, say so in limitations rather
  than forcing a fit.
- Set a variable `result` to a dict with keys: summary (str), findings (list of
  str), statistics (dict of computed numbers), limitations (list of str).
- Do not read/write files, print, or mutate global state.
"""


def _guarded_import(name: str, *args: Any, **kwargs: Any):
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of '{name}' is not allowed in the analysis sandbox")
    return builtins.__import__(name, *args, **kwargs)


def _safe_builtins() -> dict[str, Any]:
    allowed = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    allowed["__import__"] = _guarded_import
    return allowed


def _primitives(records: list[dict[str, Any]]) -> dict[str, Callable[..., Any]]:
    """Expose vetted tools as df-free callables bound to this run's records."""

    def _bind(tool_name: str) -> Callable[..., Any]:
        tool = ANALYSIS_TOOLS[tool_name]

        def _call(**kwargs: Any) -> Any:
            return tool.invoke({"records": records, **kwargs})

        _call.__name__ = tool_name
        return _call

    return {name: _bind(name) for name in _PRIMITIVE_TOOLS if name in ANALYSIS_TOOLS}


def _build_prompt(intent: AnalysisIntent, context: AnalysisContext) -> str:
    return (
        f"Objective: {intent.objective}\n"
        f"Analysis focus: {intent.analysis_focus}\n"
        f"Domain framing: {domain_framing(intent.domain)}\n"
        f"Metric hints: {intent.metric_hints}\n"
        f"Dimension hints: {intent.dimension_hints}\n"
        f"Entity hints: {intent.entity_hints}\n"
        f"Time column: {intent.time_column}; is_time_based: {intent.is_time_based}; "
        f"time_grain: {intent.time_grain}; time_span_days: {intent.time_span_days}\n\n"
        f"Columns: {context.columns}\n"
        f"Numeric: {context.numeric_columns}\n"
        f"Categorical: {context.categorical_columns}\n"
        f"Temporal: {context.temporal_columns}\n"
        f"Sample rows: {context.sample_rows}\n"
    )


def generate_analysis_code(
    intent: AnalysisIntent,
    context: AnalysisContext,
    *,
    model: Any | None = None,
    feedback: str | None = None,
) -> GeneratedAnalysisCode:
    """Ask CODE_GENERATOR_MODEL for structured analysis code."""

    chat_model = model or get_chat_model(model_env="CODE_GENERATOR_MODEL", temperature=0)
    structured_model = chat_model.with_structured_output(GeneratedAnalysisCode)
    system = GENERATE_SYSTEM_PROMPT.format(primitives=list(_PRIMITIVE_TOOLS))
    human = _build_prompt(intent, context)
    if feedback:
        human += (
            "\nA previous attempt was rejected. Fix these issues and regenerate:\n"
            f"{feedback}\n"
        )
    result = structured_model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=human),
    ])
    return result if isinstance(result, GeneratedAnalysisCode) else GeneratedAnalysisCode.model_validate(result)


def execute_generated_code(
    code: GeneratedAnalysisCode, dataframe: pd.DataFrame
) -> dict[str, Any]:
    """Run generated code in the opened sandbox and return its `result` dict."""

    import math
    import numpy as np

    records = dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")
    globals_dict: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "math": math,
        "primitives": _primitives(records),
    }
    locals_dict: dict[str, Any] = {"df": dataframe.copy(), "result": None}
    source = f"{code.imports}\n{code.code}" if code.imports else code.code
    try:
        exec(compile(source, "<generated_analysis>", "exec"), globals_dict, locals_dict)
    except Exception as exc:  # noqa: BLE001 - surfaced to the reflect loop
        raise AnalysisCodeError(f"generated code failed: {exc}") from exc

    result = locals_dict.get("result")
    if not isinstance(result, dict):
        raise AnalysisCodeError("generated code must set `result` to a dict.")
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise AnalysisCodeError(f"result dict is missing required keys: {missing}")
    return result
