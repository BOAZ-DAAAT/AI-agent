"""DB 실행 예외를 범용 SQL repair 전략으로 보수적으로 분류한다."""

from __future__ import annotations

import re
from typing import Any


REPAIR_STRATEGY_BY_MYSQL_CODE = {
    1052: "rewrite_identifier",  # Column ... is ambiguous
    1054: "rewrite_identifier",  # Unknown column
    1060: "rewrite_identifier",  # Duplicate column name
    1055: "rewrite_aggregation",  # GROUP BY 계약 위반
    1111: "rewrite_aggregation",  # Invalid use of group function
    1140: "rewrite_aggregation",  # 집계 컬럼과 비집계 컬럼 혼용
    1064: "rewrite_syntax",  # SQL syntax error
    1066: "rewrite_syntax",  # Not unique table/alias
    1109: "rewrite_syntax",  # Unknown table(alias 참조 오류 포함)
    1248: "rewrite_syntax",  # Every derived table must have its own alias
    1305: "rewrite_syntax",  # FUNCTION does not exist
    1582: "rewrite_syntax",  # 함수 인자 수 오류
    1265: "normalize_invalid_value",  # Data truncated
    1292: "normalize_invalid_value",  # Incorrect date/time/value
    1366: "normalize_invalid_value",  # Incorrect integer/string value
    1411: "normalize_invalid_value",  # Incorrect value for conversion function
}
MISSING_TABLE_MYSQL_CODES = {1146}
INFRASTRUCTURE_MYSQL_CODES = {
    1044, 1045, 1049,  # 권한, 인증, 데이터베이스 선택
    1142, 1143, 1227,  # 권한
    1205, 1213,  # lock timeout/deadlock
    2002, 2003, 2005, 2006, 2013,  # 연결과 네트워크
    3024,  # query timeout
}

STRATEGY_PATTERNS = {
    "rewrite_identifier": (
        r"unknown column",
        r"column .* is ambiguous",
        r"ambiguous column",
        r"duplicate column",
        r"column .* specified twice",
    ),
    "rewrite_aggregation": (
        r"invalid use of group function",
        r"mixing of group columns",
        r"isn't in group by",
        r"not in group by",
    ),
    "rewrite_syntax": (
        r"not unique table/alias",
        r"unknown table .* in (?:field|where|on|order|group)",
        r"every derived table must have its own alias",
        r"you have an error in your sql syntax",
        r"syntax error",
        r"function .* does not exist",
        r"incorrect parameter count .* function",
    ),
    "normalize_invalid_value": (
        r"data truncated for column",
        r"incorrect (?:date|datetime|time|integer|decimal|double|numeric|string|value)",
        r"invalid (?:date|datetime|time|integer|decimal|double|numeric|string) value",
        r"truncated incorrect",
    ),
}
INFRASTRUCTURE_PATTERNS = (
    r"access denied",
    r"permission denied",
    r"not authorized",
    r"authentication",
    r"unknown database",
    r"no database selected",
    r"can't connect",
    r"cannot connect",
    r"connection (?:refused|reset|lost|closed)",
    r"lost connection",
    r"server has gone away",
    r"network",
    r"timed? out",
    r"timeout",
)

_QUOTED_TOKEN = re.compile(r"['\"]([^'\"]*)['\"]")
_COLUMN_PATTERNS = (
    re.compile(r"for\s+column\s+[`'\"]?([^`'\"\s,]+)[`'\"]?", re.IGNORECASE),
    re.compile(r"column\s+[`'\"]([^`'\"]+)[`'\"]", re.IGNORECASE),
)
_INVALID_VALUE_PATTERNS = (
    re.compile(
        r"(?:incorrect|invalid|truncated)\s+(?:date|datetime|time|integer|decimal|double|numeric|string)?\s*value\s*:\s*[`'\"]([^`'\"]*)[`'\"]",
        re.IGNORECASE,
    ),
    re.compile(r"value\s*[=:]\s*[`'\"]([^`'\"]*)[`'\"]", re.IGNORECASE),
    re.compile(r"data\s+truncated\s*:\s*[`'\"]([^`'\"]*)[`'\"]", re.IGNORECASE),
)


def _mysql_error_code(error: BaseException) -> int | None:
    """SQLAlchemy/DBAPI 래퍼를 따라가며 MySQL 숫자 오류 코드를 찾는다."""
    current: Any = error
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        args = getattr(current, "args", ())
        if args and isinstance(args[0], int):
            return args[0]
        current = getattr(current, "orig", None)
    return None


def _first_match(patterns: tuple[re.Pattern[str], ...], message: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_invalid_value(message: str, column_name: str | None) -> str | None:
    for pattern in _INVALID_VALUE_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1).strip()

    # 일부 드라이버는 1265 오류의 잘못된 값을 별도 라벨 없이 인용한다.
    quoted = [item.strip() for item in _QUOTED_TOKEN.findall(message) if item.strip()]
    candidates = [item for item in quoted if item != column_name and not item.casefold().startswith("db.")]
    return candidates[0] if candidates else None


def _strategy_from_message(normalized_message: str) -> str:
    for strategy, patterns in STRATEGY_PATTERNS.items():
        if any(re.search(pattern, normalized_message) for pattern in patterns):
            return strategy
    return "none"


def classify_execution_error(error: BaseException | str) -> dict[str, Any]:
    """오류 코드 우선, 제한된 메시지 패턴 보조로 내부 repair 컨텍스트를 만든다."""
    message = str(error)
    error_code = _mysql_error_code(error) if isinstance(error, BaseException) else None
    normalized = message.casefold()

    if error_code in MISSING_TABLE_MYSQL_CODES:
        classification = "missing_table"
        repair_strategy = "none"
    elif error_code in REPAIR_STRATEGY_BY_MYSQL_CODE:
        repair_strategy = REPAIR_STRATEGY_BY_MYSQL_CODE[error_code]
        classification = "repairable_data" if repair_strategy == "normalize_invalid_value" else "repairable_sql"
    elif error_code in INFRASTRUCTURE_MYSQL_CODES:
        classification = "infrastructure"
        repair_strategy = "none"
    elif re.search(r"table .*doesn't exist", normalized):
        classification = "missing_table"
        repair_strategy = "none"
    elif any(re.search(pattern, normalized) for pattern in INFRASTRUCTURE_PATTERNS):
        classification = "infrastructure"
        repair_strategy = "none"
    else:
        repair_strategy = _strategy_from_message(normalized)
        if repair_strategy == "normalize_invalid_value":
            classification = "repairable_data"
        elif repair_strategy != "none":
            classification = "repairable_sql"
        else:
            classification = "unknown"

    column_name = _first_match(_COLUMN_PATTERNS, message) if repair_strategy == "normalize_invalid_value" else None
    invalid_value = _extract_invalid_value(message, column_name) if repair_strategy == "normalize_invalid_value" else None
    return {
        "classification": classification,
        "repair_strategy": repair_strategy,
        "retryable": repair_strategy != "none",
        "error_code": error_code,
        "component": "main",
        "column_name": column_name,
        "invalid_value": invalid_value,
        "message": message,
    }
