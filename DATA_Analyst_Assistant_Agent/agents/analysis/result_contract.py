"""Normalize generated analysis results before stable schema validation."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def normalize_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-friendly copy of an LLM generated result payload."""

    normalized = deepcopy(payload)
    notes: list[str] = []

    for field in ("findings", "limitations", "method_notes", "interpretation"):
        normalized[field] = _normalize_string_list(
            normalized.get(field),
            f"result.{field}",
            notes,
        )
    normalized["evidence_tables"] = _normalize_evidence_tables(
        normalized.get("evidence_tables"),
        notes,
    )
    normalized["hypothesis_tests"] = _normalize_hypothesis_tests(
        normalized.get("hypothesis_tests"),
        notes,
    )
    normalized["method_decision"] = _normalize_method_decision(
        normalized.get("method_decision"),
        notes,
    )
    normalized["method_decision"] = _fallback_method_decision(normalized, notes)
    _extend_method_notes(normalized, notes)
    return normalized


def fatal_result_contract_errors(payload: dict[str, Any]) -> list[str]:
    """Return result shape errors that should be fixed by regeneration."""

    errors: list[str] = []
    evidence_tables = payload.get("evidence_tables")
    if evidence_tables is not None:
        if not isinstance(evidence_tables, list):
            errors.append("result['evidence_tables'] must be a list of dicts.")
        else:
            for index, table in enumerate(evidence_tables):
                if not isinstance(table, dict):
                    errors.append(f"result['evidence_tables'][{index}] must be a dict.")
                    continue
                rows = table.get("rows")
                if rows is not None and not isinstance(rows, list):
                    errors.append(
                        f"result['evidence_tables'][{index}]['rows'] must be a list of row dicts."
                    )
                elif isinstance(rows, list) and any(not isinstance(row, dict) for row in rows):
                    errors.append(
                        f"result['evidence_tables'][{index}]['rows'] must contain only row dicts."
                    )

    hypothesis_tests = payload.get("hypothesis_tests")
    if hypothesis_tests is not None:
        if not isinstance(hypothesis_tests, list):
            errors.append("result['hypothesis_tests'] must be a list of dicts.")
        else:
            for index, test in enumerate(hypothesis_tests):
                if not isinstance(test, dict):
                    errors.append(f"result['hypothesis_tests'][{index}] must be a dict.")

    return errors


def _normalize_evidence_tables(value: Any, notes: list[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        notes.append("Dropped evidence_tables because it was not a list.")
        return []

    tables: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            notes.append(f"Dropped evidence_tables[{index}] because it was not a dict.")
            continue

        table = dict(item)
        title = table.get("title")
        if not str(title or "").strip():
            name = table.get("name")
            if str(name or "").strip():
                table["title"] = str(name)
                notes.append(f"Normalized evidence_tables[{index}].name to title.")
            else:
                table["title"] = "Untitled table"
                notes.append(f"Filled missing evidence_tables[{index}].title.")
        else:
            table["title"] = str(title)

        rows = table.get("rows")
        if isinstance(rows, list):
            table["rows"] = [dict(row) for row in rows if isinstance(row, dict)]
            if len(table["rows"]) != len(rows):
                notes.append(f"Dropped non-dict rows from evidence_tables[{index}].rows.")
        else:
            table["rows"] = []
            if rows is not None:
                notes.append(f"Reset evidence_tables[{index}].rows because it was not a list.")

        columns = table.get("columns")
        if isinstance(columns, list):
            table["columns"] = [str(column) for column in columns]
        else:
            table["columns"] = _columns_from_rows(table["rows"])
            if columns is not None:
                notes.append(f"Reset evidence_tables[{index}].columns because it was not a list.")
        if not table["columns"] and table["rows"]:
            table["columns"] = _columns_from_rows(table["rows"])

        tables.append({
            "title": table["title"],
            "columns": table["columns"],
            "rows": table["rows"],
        })
    return tables


def _normalize_hypothesis_tests(value: Any, notes: list[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        notes.append("Dropped hypothesis_tests because it was not a list.")
        return []

    tests: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            notes.append(f"Dropped hypothesis_tests[{index}] because it was not a dict.")
            continue

        test = dict(item)
        prefix = f"hypothesis_tests[{index}]"
        test["hypothesis"] = _text_or_default(
            test.get("hypothesis"),
            "Untitled hypothesis",
            f"{prefix}.hypothesis",
            notes,
        )
        test["test_name"] = _text_or_default(
            test.get("test_name"),
            "unspecified test",
            f"{prefix}.test_name",
            notes,
        )
        for field in ("null_hypothesis", "alternative_hypothesis"):
            if test.get(field) is not None:
                test[field] = str(test[field])
        for field in ("statistic", "p_value", "effect_size"):
            test[field] = _normalize_optional_float(test.get(field), f"{prefix}.{field}", notes)
        test["n"] = _normalize_sample_size(test.get("n"), f"{prefix}.n", notes)
        test["decision"] = _normalize_decision(test.get("decision"), f"{prefix}.decision", notes)
        test["caveats"] = _normalize_string_list(test.get("caveats"), f"{prefix}.caveats", notes)
        tests.append(test)
    return tests


def _normalize_method_decision(value: Any, notes: list[str]) -> dict[str, Any] | Any:
    if value is None or not isinstance(value, dict):
        return value

    decision = dict(value)
    for field in ("selected_method", "rationale"):
        if decision.get(field) is not None:
            decision[field] = str(decision[field])
    for field in ("assumptions_checked", "fallbacks_considered"):
        decision[field] = _normalize_string_list(
            decision.get(field),
            f"method_decision.{field}",
            notes,
        )
    return decision


def _fallback_method_decision(payload: dict[str, Any], notes: list[str]) -> dict[str, Any] | Any:
    existing = payload.get("method_decision")
    if isinstance(existing, dict):
        selected = str(existing.get("selected_method") or "").strip()
        rationale = str(existing.get("rationale") or "").strip()
        if selected and rationale:
            return existing

    selected_method = _infer_selected_method(payload)
    rationale = _infer_method_rationale(payload)
    if not selected_method and not rationale:
        return existing

    fallback = existing if isinstance(existing, dict) else {}
    fallback["selected_method"] = selected_method or "unspecified_analysis_method"
    fallback["rationale"] = rationale or "분석 결과와 method_notes를 바탕으로 후처리 단계에서 방법 선택 근거를 복원했습니다."
    fallback["assumptions_checked"] = _normalize_string_list(
        fallback.get("assumptions_checked"),
        "method_decision.assumptions_checked",
        notes,
    )
    fallback["fallbacks_considered"] = _normalize_string_list(
        fallback.get("fallbacks_considered"),
        "method_decision.fallbacks_considered",
        notes,
    )
    notes.append("Filled missing method_decision from generated analysis evidence.")
    return fallback


def _infer_selected_method(payload: dict[str, Any]) -> str:
    tests = payload.get("hypothesis_tests")
    if isinstance(tests, list):
        for test in tests:
            if not isinstance(test, dict):
                continue
            name = str(test.get("test_name") or "").strip()
            if name:
                return name

    for note in _all_method_text(payload):
        lowered = note.casefold()
        if "spearman" in lowered:
            return "spearman_correlation"
        if "pearson" in lowered:
            return "pearson_correlation"
        if "kruskal" in lowered:
            return "kruskal_wallis"
        if "mann-whitney" in lowered or "mann whitney" in lowered:
            return "mann_whitney_u"
        if "chi-square" in lowered or "chi square" in lowered:
            return "chi_square_test"
        if "regression" in lowered or "회귀" in note:
            return "regression_analysis"
        if "trend" in lowered or "추세" in note:
            return "trend_analysis"
        if "correlation" in lowered or "상관" in note:
            return "correlation_analysis"
    return ""


def _infer_method_rationale(payload: dict[str, Any]) -> str:
    for note in payload.get("method_notes") or []:
        text = str(note or "").strip()
        if text:
            return text
    for field in ("interpretation", "summary"):
        value = payload.get(field)
        if isinstance(value, list):
            for item in value:
                text = str(item or "").strip()
                if text:
                    return text
        else:
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _all_method_text(payload: dict[str, Any]) -> list[str]:
    collected: list[str] = []
    for field in ("summary",):
        text = str(payload.get(field) or "").strip()
        if text:
            collected.append(text)
    for field in ("method_notes", "interpretation", "findings", "limitations"):
        value = payload.get(field)
        if isinstance(value, list):
            collected.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
            if text:
                collected.append(text)
    return collected


def _normalize_string_list(value: Any, field: str, notes: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        notes.append(f"Normalized {field} from string to list.")
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    notes.append(f"Reset {field} because it was not a list.")
    return []


def _text_or_default(value: Any, default: str, field: str, notes: list[str]) -> str:
    text = str(value or "").strip()
    if text:
        return text
    notes.append(f"Filled missing {field}.")
    return default


def _normalize_optional_float(value: Any, field: str, notes: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        notes.append(f"Cleared {field} because it was not numeric.")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
        if match:
            notes.append(f"Parsed numeric value from {field}.")
            return float(match.group(0))
        notes.append(f"Cleared {field} because it was not numeric.")
        return None


def _normalize_sample_size(value: Any, field: str, notes: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        notes.append(f"Cleared {field} because sample size was not numeric.")
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        notes.append(f"Cleared {field} because sample size was not numeric.")
        return None
    if numeric.is_integer():
        if isinstance(value, float):
            notes.append(f"Normalized {field} from integer-valued float.")
        return int(numeric)
    notes.append(f"Cleared {field} because sample size was a non-integer float.")
    return None


def _normalize_decision(value: Any, field: str, notes: list[str]) -> str:
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "supported": "supported",
        "support": "supported",
        "accepted": "supported",
        "accept": "supported",
        "not_supported": "not_supported",
        "unsupported": "not_supported",
        "rejected": "not_supported",
        "reject": "not_supported",
        "inconclusive": "inconclusive",
        "not_conclusive": "inconclusive",
        "uncertain": "inconclusive",
    }
    decision = aliases.get(normalized)
    if decision is None:
        notes.append(f"Normalized {field} to inconclusive because it was not recognized.")
        return "inconclusive"
    if decision != value:
        notes.append(f"Normalized {field} to {decision}.")
    return decision


def _columns_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            column = str(key)
            if column not in columns:
                columns.append(column)
    return columns


def _extend_method_notes(payload: dict[str, Any], notes: list[str]) -> None:
    if not notes:
        return
    current = payload.get("method_notes")
    if not isinstance(current, list):
        current = []
    merged = [str(item) for item in current]
    for note in notes:
        if note not in merged:
            merged.append(note)
    payload["method_notes"] = merged
