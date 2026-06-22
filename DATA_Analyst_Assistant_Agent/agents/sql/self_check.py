"""SQL safety validation for the SQL-Agent.

Returns LocalCheck results compatible with AgentEnvelope.validation.local_checks.
"""

from __future__ import annotations

import re

from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck

_BLOCKED_KEYWORDS: set[str] = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "RENAME", "MERGE",
    "EXEC", "EXECUTE", "CALL",
}

_BLOCKED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _BLOCKED_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

_MULTI_STATEMENT_PATTERN = re.compile(r";\s*\S")


def _normalize(sql: str) -> str:
    return sql.replace("```sql", "").replace("```", "").strip()


def check_read_only(sql: str) -> LocalCheck:
    normalized = _normalize(sql).lstrip()
    first_word = normalized.split()[0].upper() if normalized else ""
    passed = first_word in ("SELECT", "WITH")
    return LocalCheck(
        name="read_only_sql",
        passed=passed,
        severity="error" if not passed else "info",
        detail=f"SQL starts with '{first_word}'." if first_word else "Empty SQL.",
    )


def check_blocked_keywords(sql: str) -> LocalCheck:
    normalized = _normalize(sql)
    match = _BLOCKED_PATTERN.search(normalized)
    passed = match is None
    return LocalCheck(
        name="no_blocked_keywords",
        passed=passed,
        severity="error" if not passed else "info",
        detail=f"Blocked keyword: {match.group(0)}." if match else "No blocked keywords.",
    )


def check_single_statement(sql: str) -> LocalCheck:
    normalized = _normalize(sql)
    match = _MULTI_STATEMENT_PATTERN.search(normalized)
    passed = match is None
    return LocalCheck(
        name="single_statement",
        passed=passed,
        severity="error" if not passed else "info",
        detail="Multiple statements detected." if not passed else "Single statement.",
    )


def check_preview_has_columns(columns: list[str] | None) -> LocalCheck:
    passed = bool(columns)
    return LocalCheck(
        name="preview_has_columns",
        passed=passed,
        severity="error" if not passed else "info",
        detail=f"{len(columns)} columns." if columns else "No columns in preview.",
    )


def check_row_count_available(row_count: int | None) -> LocalCheck:
    passed = row_count is not None and row_count >= 0
    return LocalCheck(
        name="preview_row_count_available",
        passed=passed,
        severity="warning" if not passed else "info",
        detail=f"row_count={row_count}." if passed else "Row count unavailable.",
    )


def run_sql_self_check(
    sql: str,
    columns: list[str] | None = None,
    row_count: int | None = None,
) -> list[LocalCheck]:
    """Run all SQL safety checks. Pre-execution checks always run.
    Post-execution checks (columns, row_count) run when values are provided."""
    checks = [
        check_read_only(sql),
        check_blocked_keywords(sql),
        check_single_statement(sql),
    ]
    if columns is not None:
        checks.append(check_preview_has_columns(columns))
    if row_count is not None:
        checks.append(check_row_count_available(row_count))
    return checks


def is_sql_safe(sql: str) -> bool:
    """Quick boolean: is this SQL safe to send to the backend?"""
    return all(c.passed for c in run_sql_self_check(sql))
