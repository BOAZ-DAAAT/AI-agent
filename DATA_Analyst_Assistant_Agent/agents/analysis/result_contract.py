"""Normalize generated analysis results before stable schema validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-friendly copy of an LLM generated result payload."""

    normalized = deepcopy(payload)
    notes: list[str] = []

    normalized["evidence_tables"] = _normalize_evidence_tables(
        normalized.get("evidence_tables"),
        notes,
    )
    normalized["hypothesis_tests"] = _normalize_hypothesis_tests(
        normalized.get("hypothesis_tests"),
        notes,
    )
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
        n = test.get("n")
        if isinstance(n, float):
            if n.is_integer():
                test["n"] = int(n)
                notes.append(f"Normalized hypothesis_tests[{index}].n from integer-valued float.")
            else:
                test["n"] = None
                notes.append(
                    f"Cleared hypothesis_tests[{index}].n because sample size was a non-integer float."
                )
        caveats = test.get("caveats")
        if isinstance(caveats, str):
            test["caveats"] = [caveats]
            notes.append(f"Normalized hypothesis_tests[{index}].caveats from string to list.")
        elif isinstance(caveats, list):
            test["caveats"] = [str(caveat) for caveat in caveats]
        elif caveats is not None:
            test["caveats"] = []
            notes.append(f"Reset hypothesis_tests[{index}].caveats because it was not a list.")
        tests.append(test)
    return tests


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
