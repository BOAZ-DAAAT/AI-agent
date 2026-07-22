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

import ast
import builtins
import json
import multiprocessing as mp
import os
import queue
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

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


@dataclass(frozen=True)
class CodeExecutionPlan:
    decision: str
    risk_level: str
    reasons: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "risk_level": self.risk_level,
            "reasons": list(self.reasons),
            "row_count": self.row_count,
            "column_count": self.column_count,
        }


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
# "time" is included because pandas Timestamp.strftime() imports it internally
# even when the generated code never writes `import time` itself.
_ALLOWED_IMPORT_ROOTS = frozenset({
    "pandas", "numpy", "math", "statistics", "datetime", "time", "collections",
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
DEFAULT_CODE_EXEC_TIMEOUT_SECONDS = 600.0
SMALL_MODELING_ROW_LIMIT = 10_000
LARGE_MODELING_ROW_LIMIT = 100_000


_ROW_ITERATION_METHODS = frozenset({"iterrows", "itertuples"})


GENERATE_SYSTEM_PROMPT = """You write one small, auditable Python analysis script.

Rules:
- Operate on the pandas DataFrame named `df` (already loaded). `pd` and `np` are available.
- 사용자에게 노출되는 모든 결과 텍스트는 한국어로 작성한다: summary, findings,
  limitations, interpretation, method_notes, review_request의 labels/questions/options,
  evidence table titles.
- You MAY import from: pandas, numpy, math, statistics, scipy, statsmodels,
  sklearn, lifelines, datetime, time, collections, itertools, functools, re, json.
  No other imports (no os/sys/file/network access).
- For heavy specialized methods, call the provided primitives instead of
  reimplementing them: {primitives}. Each takes keyword column arguments.
- Choose the statistically valid method for the stated objective and domain.
- Treat EDA insights and hypotheses as exploratory candidate hints, not as a
  checklist. Select only candidates that directly support the analysis plan and
  user question; ignore unrelated candidates without mentioning every omission.
- Do not copy EDA insight_result, hypotheses, or final_summary into the final
  analysis as evidence. Recompute coefficients, p-values, test decisions, effect
  sizes, and interpretations from `df` in this analysis stage.
- You may add new analysis hypotheses when they are needed to answer the plan
  and are supported by the available columns/data. Include them in
  `hypothesis_tests` when tested.
- When defining or evaluating a heuristic metric, proxy label, operational
  threshold, or segment, treat it as a hypothesis-driven analysis:
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
- Do not create `review_request` just because an operational definition or
  threshold is needed. First compare defensible criteria with the data: observed
  distribution, relationship to the outcome, sample size, sensitivity, and
  limitations. Create `review_request` only when the definition or method choice
  materially affects interpretation and user confirmation would improve analysis
  quality.
- When the user asks to exclude entities below a sample-size threshold (for
  example "n<30 sellers"), compute that threshold at the requested entity grain
  from the current dataframe before filtering. For seller-level thresholds on a
  seller/order/month mart, prefer `df.groupby("seller_id")["order_id"].nunique()`
  when `order_id` exists, otherwise use row counts. Do NOT treat a pre-existing
  count-like mart column such as `seller_order_count` as the entity sample size
  unless the analysis_data_contract explicitly defines it as that exact
  entity-level total.
- Prefer analysis_data_contract.sample_size_rules over column-name guessing.
  If sample_size_rules has no implemented preferred_column, compute from its
  source_columns when available or record a limitation instead of substituting
  another count-like column by name.
- Treat analysis_data_contract.analysis_heuristics as analyst-side method
  assumptions. You may apply defensible thresholds/bins/labels, but record the
  selected value, rationale, and sensitivity/limitation in method_decision,
  method_notes, or limitations.
- If an entity filter leaves too few rows/groups for a statistic, set the
  related decision to `inconclusive`, state that the test is not estimable, and
  do not describe NaN/None statistics as a positive/negative relationship.
- Always include `method_decision` in result with selected_method, rationale,
  assumptions_checked, and fallbacks_considered. Choose a method automatically
  whenever the observed data and objective establish a defensible preference.
- Treat `method_decision` as part of the required result contract, not as an
  optional note. If you compute a correlation, trend, regression, test, or
  heuristic threshold, explicitly name that method and why it was chosen.
- Create `review_request` only when at least two mutually exclusive analysis
  paths are each valid for this data and their different assumptions or
  interpretations would materially change the next analysis. Do not ask merely
  because a parameter has multiple possible values. A review_request may ask for
  confirmation of one recommended path, or present options when valid paths have
  different interpretive tradeoffs; options must not be a bare list of numbers.
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
  result["method_decision"] = {{
      "selected_method": "spearman_correlation",
      "rationale": "왜도가 있는 집계 지표 간 단조 관계를 확인하기 위해 Pearson 대신 Spearman을 선택했다.",
      "assumptions_checked": ["판매자 단위 재집계 후 순위 기반 관계를 해석한다."],
      "fallbacks_considered": ["선형성 가정이 필요한 Pearson 상관은 보조 대안으로만 검토했다."],
  }}
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


class AnalysisGenerationError(RuntimeError):
    """Raised when the code-generator model's structured output cannot be
    parsed, even after stripping markdown fences/trailing noise."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Recover the JSON object from a response the model wrapped in markdown
    fences or trailed with stray prose after the closing brace.

    Mirrors validation_contract.py's `_normalized_sql()` for the SQL agent:
    LLM structured output is not always byte-clean, so noise around a
    genuinely valid object is tolerated before handing off to the schema
    validator, instead of treating any formatting slip as a hard failure.
    """
    stripped = text.strip()
    fence_match = _JSON_FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped[start:]


def _recover_from_validation_error(exc: ValidationError) -> GeneratedAnalysisCode | None:
    """Best-effort repair for a schema-shaped response the strict JSON parser
    rejected. Avoids burning a full model round-trip for output that was
    actually fine once the fence/trailing noise around it is stripped."""

    for error in exc.errors():
        raw = error.get("input")
        if not isinstance(raw, str):
            continue
        cleaned = _extract_json_object(raw)
        if cleaned == raw:
            continue
        try:
            return GeneratedAnalysisCode.model_validate_json(cleaned)
        except ValidationError:
            continue
    return None


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
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        result = structured_model.invoke(messages)
    except ValidationError as exc:
        recovered = _recover_from_validation_error(exc)
        if recovered is not None:
            return recovered
        raise AnalysisGenerationError(
            f"Model response was not valid JSON and could not be repaired: {exc}"
        ) from exc
    return result if isinstance(result, GeneratedAnalysisCode) else GeneratedAnalysisCode.model_validate(result)


def inspect_generated_code(
    code: GeneratedAnalysisCode,
    dataframe: pd.DataFrame,
) -> CodeExecutionPlan:
    source = _source_for_code(code)
    lowered = source.lower()
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    reasons: list[str] = []

    blocked_patterns = (
        ("file/network/process access is not allowed", r"\b(open|eval|exec|compile)\s*\("),
        (
            "filesystem/network/process module usage is not allowed",
            r"(^|\n)\s*(import|from)\s+(os|sys|subprocess|socket|pathlib|requests|urllib)\b|\b(os|sys|subprocess|socket|pathlib|requests|urllib)\s*\.",
        ),
    )
    for reason, pattern in blocked_patterns:
        if re.search(pattern, lowered):
            return CodeExecutionPlan(
                decision="blocked_unsafe",
                risk_level="high",
                reasons=[reason],
                row_count=row_count,
                column_count=column_count,
            )

    if re.search(r"\bwhile\b", lowered):
        reasons.append("while loop may run indefinitely")
    if re.search(r"\b(gridsearchcv|randomizedsearchcv|cross_val_score|cross_validate)\b", lowered):
        reasons.append("cross-validation or hyperparameter search can be long-running")
    if re.search(r"\b(randomforest|gradientboosting|xgboost|xgb|lightgbm|catboost)\b", lowered):
        reasons.append("tree ensemble training can be long-running")
    if re.search(r"\b(bootstrap|permutation|simulation|monte.?carlo)\b", lowered) and re.search(r"\bfor\b|\brange\s*\(", lowered):
        reasons.append("simulation or resampling loop detected")
    reasons.extend(_expensive_iteration_reasons(source, row_count))

    fitting_detected = bool(re.search(r"\.fit(_predict|_transform)?\s*\(", lowered))
    simple_modeling = bool(re.search(r"\b(ols|glm|logit|kmeans|pca)\b", lowered))
    if fitting_detected:
        if row_count > LARGE_MODELING_ROW_LIMIT:
            reasons.append("model fitting on a large dataframe")
        elif row_count > SMALL_MODELING_ROW_LIMIT and not simple_modeling:
            reasons.append("model fitting on a medium dataframe without an explicitly lightweight method")

    if reasons:
        return CodeExecutionPlan(
            decision="manual_run_recommended",
            risk_level="high" if any("large" in reason or "indefinitely" in reason for reason in reasons) else "medium",
            reasons=reasons,
            row_count=row_count,
            column_count=column_count,
        )
    return CodeExecutionPlan(
        decision="auto_run",
        risk_level="low",
        reasons=["code passed preflight for automatic execution"],
        row_count=row_count,
        column_count=column_count,
    )


def execute_generated_code(
    code: GeneratedAnalysisCode,
    dataframe: pd.DataFrame,
    *,
    isolated: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run generated code in the opened sandbox and return its `result` dict."""

    if isolated:
        return _execute_generated_code_isolated(
            code,
            dataframe,
            timeout_seconds=_code_exec_timeout_seconds(timeout_seconds),
        )
    return _execute_generated_code_inline(code, dataframe)


def _source_for_code(code: GeneratedAnalysisCode) -> str:
    return f"{code.imports}\n{code.code}" if code.imports else code.code


def _expensive_iteration_reasons(source: str, row_count: int) -> list[str]:
    if row_count <= SMALL_MODELING_ROW_LIMIT:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    visitor = _IterationRiskVisitor()
    visitor.visit(tree)
    reasons: list[str] = []
    if visitor.has_nested_loop:
        reasons.append("actual nested loop on a non-small dataframe may be expensive")
    if visitor.has_dataframe_row_iteration:
        reasons.append("dataframe row iteration on a non-small dataframe may be expensive")
    return reasons


class _IterationRiskVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_depth = 0
        self.has_nested_loop = False
        self.has_dataframe_row_iteration = False

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor) -> None:
        if self.loop_depth > 0:
            self.has_nested_loop = True
        if _iterates_dataframe_rows(node.iter):
            self.has_dataframe_row_iteration = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        if self.loop_depth > 0 or len(node.generators) > 1:
            self.has_nested_loop = True
        for generator in node.generators:
            if _iterates_dataframe_rows(generator.iter):
                self.has_dataframe_row_iteration = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1


def _iterates_dataframe_rows(node: ast.AST) -> bool:
    unwrapped = _unwrap_iteration_call(node)
    if _is_dataframe_row_method_call(unwrapped):
        return True
    if _is_range_over_dataframe_length(unwrapped):
        return True
    if _is_zip_over_dataframe_columns(unwrapped):
        return True
    return _is_dataframe_records_call(unwrapped)


def _unwrap_iteration_call(node: ast.AST) -> ast.AST:
    current = node
    wrappers = {"enumerate", "list", "tuple", "iter", "reversed", "sorted"}
    while (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Name)
        and current.func.id in wrappers
        and current.args
    ):
        current = current.args[0]
    return current


def _is_dataframe_row_method_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _ROW_ITERATION_METHODS
        and _is_df_name(node.func.value)
    )


def _is_range_over_dataframe_length(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and node.args
    ):
        return False
    return any(_is_dataframe_length_expr(arg) for arg in node.args)


def _is_dataframe_length_expr(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and _is_df_name(node.args[0])
    ):
        return True
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "shape"
        and _is_df_name(node.value.value)
        and _is_zero_constant(node.slice)
    )


def _is_zip_over_dataframe_columns(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and any(_is_dataframe_column_access(arg) for arg in node.args)
    )


def _is_dataframe_records_call(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_dict"
        and _is_df_name(node.func.value)
    ):
        return False
    if node.args and _is_records_literal(node.args[0]):
        return True
    return any(
        keyword.arg == "orient" and keyword.value is not None and _is_records_literal(keyword.value)
        for keyword in node.keywords
    )


def _is_dataframe_column_access(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript) and _is_df_name(node.value)


def _is_df_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "df"


def _is_zero_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def _is_records_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "records"


def _execute_generated_code_inline(
    code: GeneratedAnalysisCode,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Run generated code in-process. Tests and the subprocess worker use this."""

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
    source = _source_for_code(code)
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


def _code_exec_timeout_seconds(value: float | None) -> float:
    if value is not None:
        return float(value)
    return float(os.getenv("ANALYSIS_CODE_EXEC_TIMEOUT_SECONDS", str(DEFAULT_CODE_EXEC_TIMEOUT_SECONDS)))


def _execute_generated_code_isolated(
    code: GeneratedAnalysisCode,
    dataframe: pd.DataFrame,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_execute_generated_code_worker,
        args=(code.model_dump(mode="json"), dataframe, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise AnalysisCodeError(f"generated code exceeded execution timeout: {timeout_seconds:g} seconds")
    try:
        payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise AnalysisCodeError("generated code worker exited without returning a result.") from exc
    if not payload.get("ok"):
        raise AnalysisCodeError(str(payload.get("error") or "generated code failed in isolated worker."))
    result = payload.get("result")
    if not isinstance(result, dict):
        raise AnalysisCodeError("generated code worker returned an invalid result payload.")
    return result


def _execute_generated_code_worker(
    code_payload: dict[str, Any],
    dataframe: pd.DataFrame,
    result_queue: Any,
) -> None:
    try:
        code = GeneratedAnalysisCode.model_validate(code_payload)
        result_queue.put({"ok": True, "result": _execute_generated_code_inline(code, dataframe)})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
