"""SQL safety and MySQL dialect validation for the SQL-Agent."""

from __future__ import annotations

import re
from typing import Iterable

from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import split_sql_statements
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

_MYSQL_BANNED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bjulianday\s*\(", "SQLite JULIANDAY() 함수는 MySQL에서 지원되지 않습니다. DATEDIFF()/TIMESTAMPDIFF()를 사용하세요."),
    (r"\bstrftime\s*\(", "SQLite STRFTIME() 함수는 MySQL에서 지원되지 않습니다. DATE_FORMAT()을 사용하세요."),
    (r"\bdate_trunc\s*\(", "PostgreSQL DATE_TRUNC() 함수는 MySQL에서 지원되지 않습니다. DATE_FORMAT() 또는 TIMESTAMP() 조합을 사용하세요."),
    (r"\bilike\b", "PostgreSQL ILIKE 연산자는 MySQL에서 지원되지 않습니다. LOWER(col) LIKE LOWER(pattern) 형태를 사용하세요."),
    (r"::[a-z_]+", "PostgreSQL 타입 캐스팅(::type)은 MySQL에서 지원되지 않습니다. CAST(... AS ...)를 사용하세요."),
    (r"\b(count|sum|avg|min|max)\s*\(\s*\)", "집계 함수에 인자가 비어 있습니다. COUNT(*) 또는 적절한 컬럼/표현식을 사용하세요."),
    (
        r"\b(count|sum|avg|min|max)\s*\(\s*distinct\b[\s\S]*?\)\s*over\s*\(",
        "MySQL does not support DISTINCT inside window aggregate functions. Pre-aggregate in a CTE/subquery, then join it back.",
    ),
)


def _normalize(sql: str) -> str:
    return sql.replace("```sql", "").replace("```", "").strip()


def _first_match(sql: str, patterns: Iterable[tuple[str, str]]) -> tuple[str, str] | None:
    normalized = _normalize(sql)
    for pattern, message in patterns:
        matched = re.search(pattern, normalized, re.IGNORECASE)
        if matched:
            return matched.group(0), message
    return None


def check_read_only(sql: str) -> LocalCheck:
    statements = split_sql_statements(sql)
    first_words = []
    passed = True
    for statement in statements:
        normalized = _normalize(statement).lstrip()
        first_word = normalized.split()[0].upper() if normalized else ""
        first_words.append(first_word)
        if first_word not in ("SELECT", "WITH"):
            passed = False
    return LocalCheck(
        name="read_only_sql",
        passed=passed,
        severity="error" if not passed else "info",
        detail=f"Statements start with {first_words}." if first_words else "Empty SQL.",
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
    statements = split_sql_statements(normalized)
    match = _MULTI_STATEMENT_PATTERN.search(normalized)
    passed = len(statements) <= 1 and match is None
    return LocalCheck(
        name="single_statement",
        passed=passed,
        severity="error" if not passed else "info",
        detail=f"{len(statements)} statements detected." if not passed else "Single statement.",
    )


def check_mysql_dialect_compatibility(sql: str) -> LocalCheck:
    matched = _first_match(sql, _MYSQL_BANNED_PATTERNS)
    passed = matched is None
    detail = "MySQL 호환 함수/문법만 사용했습니다."
    if matched:
        token, message = matched
        detail = f"비호환 표현 '{token}' 감지: {message}"
    return LocalCheck(
        name="mysql_dialect_compatibility",
        passed=passed,
        severity="error" if not passed else "info",
        detail=detail,
    )


def mysql_dialect_error(sql: str) -> str:
    check = check_mysql_dialect_compatibility(sql)
    return "" if check.passed else check.detail


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
    *,
    allow_multi_statement: bool = False,
) -> list[LocalCheck]:
    """Run all SQL safety checks. Pre-execution checks always run.
    Post-execution checks (columns, row_count) run when values are provided."""
    checks = [
        check_read_only(sql),
        check_blocked_keywords(sql),
        check_mysql_dialect_compatibility(sql),
    ]
    if not allow_multi_statement:
        checks.append(check_single_statement(sql))
    if columns is not None:
        checks.append(check_preview_has_columns(columns))
    if row_count is not None:
        checks.append(check_row_count_available(row_count))
    return checks


def is_sql_safe(sql: str) -> bool:
    """Quick boolean: is this SQL safe to send to the backend?"""
    return all(c.passed for c in run_sql_self_check(sql))
