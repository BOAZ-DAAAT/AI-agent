from __future__ import annotations

import re


_SQL_RESERVED_WORDS = {
    "ALL", "AND", "AS", "ASC", "BY", "CASE", "CHAR", "COLLATE", "CONVERT", "CROSS",
    "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATE", "DECIMAL", "DEFAULT", "DESC",
    "DISTINCT", "DOUBLE", "ELSE", "END", "EXISTS", "FALSE", "FLOAT", "FOR", "FROM", "FULL",
    "GROUP", "HAVING", "IN", "INNER", "INT", "INTEGER", "INTERVAL", "JOIN", "LEFT", "LIKE",
    "LIMIT", "NATURAL", "NOT", "NULL", "ON", "OR", "ORDER", "OUTER", "RIGHT", "SELECT", "SIGNED",
    "THEN", "TIME", "TIMESTAMP", "TRUE", "UNION", "UNSIGNED", "USING", "VARCHAR", "WHEN", "WHERE",
    "WITH",
}


def extract_sql_aliases(sql: str) -> set[str]:
    """명시적 AS alias와 CTE 결과 컬럼 alias를 대소문자 없이 추출한다."""
    sanitized = _strip_sql_literals_and_comments(sql)
    aliases: set[str] = set()

    for match in re.finditer(
        r"\bAS\s+(`[^`]+`|[A-Za-z_][A-Za-z0-9_$]*)(?=\s|,|\)|;|$)",
        sanitized,
        flags=re.IGNORECASE,
    ):
        alias = _normalize_alias(match.group(1))
        if alias and alias.upper() not in _SQL_RESERVED_WORDS:
            aliases.add(alias.casefold())

    for match in re.finditer(
        r"(?:\bWITH|,)\s*[A-Za-z_][A-Za-z0-9_$]*\s*\(([^)]*)\)\s*AS\s*\(",
        sanitized,
        flags=re.IGNORECASE,
    ):
        for item in match.group(1).split(","):
            alias = _normalize_alias(item)
            if alias:
                aliases.add(alias.casefold())

    return aliases


def _normalize_alias(value: str) -> str:
    return value.strip().strip("`")


def _strip_sql_literals_and_comments(sql: str) -> str:
    """문자열 리터럴과 주석을 공백으로 치환해 alias 정규식의 오탐을 막는다."""
    result: list[str] = []
    index = 0
    length = len(sql)
    quote: str | None = None

    while index < length:
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < length else ""
        if quote:
            if char == "\\" and quote in {"'", '"'} and index + 1 < length:
                result.extend("  ")
                index += 2
                continue
            result.append(" ")
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            result.append(" ")
            index += 1
            continue
        if char == "-" and next_char == "-":
            end = sql.find("\n", index)
            end = length if end == -1 else end
            result.extend(" " * (end - index))
            index = end
            continue
        if char == "/" and next_char == "*":
            end = sql.find("*/", index + 2)
            end = length if end == -1 else end + 2
            result.extend(" " * (end - index))
            index = end
            continue
        result.append(char)
        index += 1

    return "".join(result)


def split_sql_statements(sql: str) -> list[str]:
    normalized = sql.replace("```sql", "").replace("```", "").strip()
    if not normalized.endswith(";"):
        normalized += ";"

    statements: list[str] = []
    buffer: list[str] = []
    in_single = False
    in_double = False
    in_backtick = False
    escape = False

    for char in normalized:
        if escape:
            buffer.append(char)
            escape = False
            continue

        if char == "\\" and (in_single or in_double):
            buffer.append(char)
            escape = True
            continue

        if char == "'" and not in_double and not in_backtick:
            in_single = not in_single
            buffer.append(char)
            continue

        if char == '"' and not in_single and not in_backtick:
            in_double = not in_double
            buffer.append(char)
            continue

        if char == "`" and not in_single and not in_double:
            in_backtick = not in_backtick
            buffer.append(char)
            continue

        if char == ";" and not in_single and not in_double and not in_backtick:
            statement = "".join(buffer).strip()
            if statement:
                if not statement.endswith(";"):
                    statement += ";"
                statements.append(statement)
            buffer = []
            continue

        buffer.append(char)

    trailing = "".join(buffer).strip()
    if trailing:
        if not trailing.endswith(";"):
            trailing += ";"
        statements.append(trailing)

    return statements
