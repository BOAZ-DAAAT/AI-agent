"""Generate node: LLM writes analysis code, executed in an opened sandbox.

This is the branch endpoint that replaces fixed tool code. The LLM produces a
``GeneratedAnalysisCode`` (structured, so no markdown-fence parsing), and the
code runs in a sandbox that, unlike the legacy codegen path, allows a small
allowlist of statistical libraries (statsmodels/scipy/sklearn/lifelines) and
exposes the vetted heavy tools as callable primitives so the model composes
them instead of re-deriving PyMC/lifelines math.

The sandbox is a soft guard (import allowlist + builtins allowlist), not full
isolation; hard isolation is the Docker executor follow-up.
"""

from __future__ import annotations

import builtins
import json
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
- Treat EDA insights and hypotheses as exploratory candidate hints, not as a
  checklist. Select only candidates that directly support the analysis plan and
  user question; ignore unrelated candidates without mentioning every omission.
- You may add new analysis hypotheses when they are needed to answer the plan
  and are supported by the available columns/data. Include them in
  `hypothesis_tests` when tested.
- When defining or evaluating a heuristic metric, proxy label, operational
  threshold, or segment (for example churn-risk groups), treat it as a
  hypothesis-driven analysis:
  - include `hypothesis_tests` in result, with items containing hypothesis,
    test_name, null_hypothesis, alternative_hypothesis, statistic, p_value,
    effect_size, n, decision (supported/inconclusive/not_supported), and caveats.
  - use statistical tests where feasible, e.g. chi-square or two-proportion
    tests for repurchase-rate differences, Kruskal-Wallis/Mann-Whitney for
    skewed numeric group differences, Spearman correlation for ordered risk
    levels, and weighted/descriptive trend models for cohorts.
  - report effect sizes with p-values; do not make a strong finding from
    p-values alone.
  - if a threshold is central to the analysis, add sensitivity checks where
    feasible and store them in statistics.
- Always include `method_decision` in result with selected_method, rationale,
  assumptions_checked, and fallbacks_considered. Choose a method automatically
  whenever the observed data and objective establish a defensible preference.
- Create `review_request` only when at least two mutually exclusive analysis
  paths are each valid for this data and their different assumptions or
  interpretations would materially change the next analysis. Do not ask merely
  because a parameter has multiple possible values.
- A `review_request` contains decision_type, question, proposal, rationale
  (list of strings), evidence, options, recommended_option_id, allow_free_text,
  free_text_prompt, impact_if_approved, and requires_followup_analysis. Each
  option contains a stable id, label, method, assumptions, advantages,
  limitations, impact, and recommended. Provide at least two options and
  exactly one recommended option.
- Do NOT create `review_request` for facts the user cannot fix by choosing an
  option, such as small sample size, group imbalance, missing values, skewed
  distributions, lack of validation set, lack of true label, short observation
  window, weak effect size, or non-causal observational data. Put those in
  `limitations` and/or `method_notes`.
- If the data is time-based, resample/aggregate at the given time_grain before
  fitting a trend. If there are too few periods, say so in limitations rather
  than forcing a fit.
- Set a variable `result` to a dict with keys: summary (str), findings (list of
  str), statistics (dict of computed numbers), limitations (list of str).
  Optional but preferred keys: hypothesis_tests, evidence_tables, interpretation,
  method_notes, review_request.
- If you include `evidence_tables`, every item MUST be a dict with exactly this
  stable shape: title (str), columns (list[str]), rows (list[dict]). Use `title`;
  do not use `name` as the table label key.
- If you include `hypothesis_tests`, every tested item should contain hypothesis,
  test_name, null_hypothesis, alternative_hypothesis, statistic, p_value,
  effect_size, n, decision, and caveats. The `n` value MUST be an integer sample
  count. Never put a mean count, weighted value, percentage, ratio, or any float
  with a fractional part in `n`.
- Example:
  result["evidence_tables"] = [{{
      "title": "group summary",
      "columns": ["group", "n", "mean_value"],
      "rows": [{{"group": "A", "n": 10, "mean_value": 3.2}}],
  }}]
- When a previous selection response is supplied, treat it as a binding
  constraint for this analysis. A free-text response is a new analysis
  constraint, not a note to append to the report.
- Phrase unsupported or weak tests as inconclusive. Never claim prediction,
  causality, or true churn labels unless those were directly measured and tested.
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


def _records_from_dataframe(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")


def _primitives(dataframe: pd.DataFrame) -> dict[str, Callable[..., Any]]:
    """Expose vetted tools as df-free callables bound to this run's records."""

    records: list[dict[str, Any]] | None = None

    def _bind(tool_name: str) -> Callable[..., Any]:
        tool = ANALYSIS_TOOLS[tool_name]

        def _call(**kwargs: Any) -> Any:
            nonlocal records
            if records is None:
                records = _records_from_dataframe(dataframe)
            return tool.invoke({"records": records, **kwargs})

        _call.__name__ = tool_name
        return _call

    return {name: _bind(name) for name in _PRIMITIVE_TOOLS if name in ANALYSIS_TOOLS}


def _build_prompt(intent: AnalysisIntent, context: AnalysisContext) -> str:
    prompt = (
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
    if context.eda_candidate_insights or context.eda_candidate_hypotheses:
        prompt += (
            "\nEDA exploratory candidates (optional context; use only if relevant to the plan):\n"
            f"Candidate insights: {context.eda_candidate_insights}\n"
            f"Candidate hypotheses: {context.eda_candidate_hypotheses}\n"
            "Do not test or report every EDA candidate by default. You may create additional "
            "hypotheses if the analysis plan requires them.\n"
        )
    if context.known_data_quality_issues:
        # Known upstream SQL source-table integrity issues (#130); reflect them in guards and limitations.
        joined = "\n".join(f"- {issue}" for issue in context.known_data_quality_issues)
        prompt += (
            "\nKnown upstream data quality issues (from SQL integrity check):\n"
            f"{joined}\n"
            "Account for these when choosing methods and stating limitations.\n"
        )
    if context.last_failure:
        prompt += (
            "\nPrevious Supervisor failure:\n"
            f"{json.dumps(context.last_failure, ensure_ascii=False, sort_keys=True)}\n"
        )
    if context.selection_response is not None:
        constraint = context.selection_response.constraint_text()
        if (
            context.selection_response.selected_option_id
            and context.review_request is not None
        ):
            selected_option = next(
                (
                    option
                    for option in context.review_request.options
                    if option.id == context.selection_response.selected_option_id
                ),
                None,
            )
            if selected_option is not None:
                constraint = json.dumps(
                    selected_option.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
        if constraint:
            prompt += (
                "\nUser response to an earlier analysis decision (binding constraint):\n"
                f"{constraint}\n"
            )
    return prompt


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

    globals_dict: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "math": math,
        "primitives": _primitives(dataframe),
        "df": dataframe.copy(),
        "result": None,
    }
    source = f"{code.imports}\n{code.code}" if code.imports else code.code
    try:
        # Use a single namespace so comprehensions/generators can resolve
        # top-level variables created by generated code.
        exec(compile(source, "<generated_analysis>", "exec"), globals_dict)
    except Exception as exc:  # noqa: BLE001 - surfaced to the reflect loop
        raise AnalysisCodeError(f"generated code failed: {exc}") from exc

    result = globals_dict.get("result")
    if not isinstance(result, dict):
        raise AnalysisCodeError("generated code must set `result` to a dict.")
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise AnalysisCodeError(f"result dict is missing required keys: {missing}")
    return result
