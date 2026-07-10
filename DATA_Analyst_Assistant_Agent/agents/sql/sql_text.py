from __future__ import annotations


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
