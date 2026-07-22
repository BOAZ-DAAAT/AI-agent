from __future__ import annotations

import json
import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    AnswerCoverage,
    GeneratedAnalysisCode,
)


def build_answer_coverage(
    intent: AnalysisIntent,
    context: AnalysisContext,
    code: GeneratedAnalysisCode | None,
    result: dict[str, Any] | None,
) -> AnswerCoverage:
    """Summarize whether generated analysis used the requested metric/dimension/time signals.

    This is intentionally structural and conservative. The supervisor owns the
    final answer-adequacy decision, but it needs explicit evidence instead of
    rereading arbitrary generated code and prose from scratch.
    """

    requested_metrics = _dedupe([context.metric_hint, *intent.metric_hints])
    requested_dimensions = _dedupe([context.dimension_hint, *intent.dimension_hints])
    searchable = _searchable_text(code, result)
    columns = [str(column) for column in context.columns]

    used_metrics = _used_requested_items(requested_metrics, searchable)
    used_dimensions = _used_requested_items(requested_dimensions, searchable)
    used_time_column = (
        intent.time_column
        if intent.time_column and _mentions(searchable, intent.time_column)
        else _first_mentioned(context.temporal_columns, searchable)
    )

    missing: list[str] = []
    for metric in requested_metrics:
        if metric not in used_metrics:
            missing.append(f"metric:{metric}")
    for dimension in requested_dimensions:
        if dimension not in used_dimensions:
            missing.append(f"dimension:{dimension}")
    if intent.is_time_based and intent.time_column and not used_time_column:
        missing.append(f"time_column:{intent.time_column}")
    missing.extend(_missing_contract_metric_support(context, searchable))

    requested_count = len(requested_metrics) + len(requested_dimensions)
    if intent.is_time_based and intent.time_column:
        requested_count += 1

    used_count = len(used_metrics) + len(used_dimensions)
    if intent.is_time_based and used_time_column:
        used_count += 1

    if not missing:
        status = "full"
    elif used_count > 0 and used_count < max(1, requested_count):
        status = "partial"
    else:
        status = "missing"

    return AnswerCoverage(
        requested_metrics=[item for item in requested_metrics if item in columns or item],
        used_metrics=used_metrics,
        requested_dimensions=[item for item in requested_dimensions if item in columns or item],
        used_dimensions=used_dimensions,
        requested_time_column=intent.time_column if intent.is_time_based else None,
        used_time_column=used_time_column,
        requested_time_grain=intent.time_grain if intent.is_time_based else None,
        used_time_grain=intent.time_grain if intent.is_time_based and used_time_column else None,
        coverage_status=status,
        missing_requirements=missing,
    )


def _searchable_text(code: GeneratedAnalysisCode | None, result: dict[str, Any] | None) -> str:
    parts = []
    if code is not None:
        parts.extend([code.imports or "", code.code or "", code.rationale or ""])
    if result:
        try:
            parts.append(json.dumps(result, ensure_ascii=False, default=str))
        except TypeError:
            parts.append(str(result))
    return "\n".join(parts).casefold()


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _used_requested_items(requested: list[str], searchable: str) -> list[str]:
    return [item for item in requested if _mentions(searchable, item)]


def _first_mentioned(candidates: list[str], searchable: str) -> str | None:
    for candidate in candidates:
        if _mentions(searchable, candidate):
            return candidate
    return None


def _missing_contract_metric_support(context: AnalysisContext, searchable: str) -> list[str]:
    contract = context.analysis_data_contract or {}
    row_grain = {str(item).strip() for item in contract.get("grain_columns") or [] if str(item).strip()}
    if not row_grain:
        return []
    metric_support = [
        item for item in contract.get("metric_support") or []
        if isinstance(item, dict)
    ]
    missing: list[str] = []
    for item in metric_support:
        metric_name = str(item.get("metric_name") or "").strip() or "unnamed_metric"
        calculation_grain = [
            str(column).strip()
            for column in item.get("calculation_grain") or []
            if str(column).strip()
        ]
        if not calculation_grain:
            continue
        grain_set = set(calculation_grain)
        if grain_set == row_grain:
            continue
        if not all(_mentions(searchable, column) for column in calculation_grain):
            missing.append(
                f"contract_metric_support:{metric_name}:missing_calculation_grain:{','.join(calculation_grain)}"
            )
            continue
        if not _mentions_downstream_aggregation(searchable):
            missing.append(
                f"contract_metric_support:{metric_name}:missing_downstream_aggregation:{','.join(calculation_grain)}"
            )
    return missing


def _mentions_downstream_aggregation(searchable: str) -> bool:
    markers = (
        "groupby",
        "pivot_table",
        "resample",
        ".agg",
        "aggregate",
        "aggregation",
    )
    return any(marker in searchable for marker in markers)


def _mentions(searchable: str, needle: str) -> bool:
    value = str(needle or "").strip()
    if not value:
        return False
    folded = value.casefold()
    if not re.search(r"[A-Za-z0-9_가-힣]", folded):
        return folded in searchable
    return re.search(rf"(?<![A-Za-z0-9_가-힣]){re.escape(folded)}(?![A-Za-z0-9_가-힣])", searchable) is not None
