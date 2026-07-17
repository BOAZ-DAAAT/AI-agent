"""DB 실행 예외를 SQL repair 가능 여부에 따라 보수적으로 분류한다."""

from __future__ import annotations

import re
from typing import Any


REPAIRABLE_MYSQL_CODES = {
    1052,  # Column ... is ambiguous
    1054,  # Unknown column
    1055,  # GROUP BY 계약 위반
    1060,  # Duplicate column name
    1064,  # SQL syntax error
    1066,  # Not unique table/alias
    1109,  # Unknown table(alias 참조 오류 포함)
    1111,  # Invalid use of group function
    1140,  # 집계 컬럼과 비집계 컬럼 혼용
    1248,  # Every derived table must have its own alias
    1305,  # FUNCTION does not exist
    1582,  # 함수 인자 수 오류
}
MISSING_TABLE_MYSQL_CODES = {1146}
INFRASTRUCTURE_MYSQL_CODES = {
    1044, 1045, 1049,  # 권한, 인증, 데이터베이스 선택
    1142, 1143, 1227,  # 권한
    1205, 1213,  # lock timeout/deadlock
    2002, 2003, 2005, 2006, 2013,  # 연결과 네트워크
    3024,  # query timeout
}

REPAIRABLE_PATTERNS = (
    r"unknown column",
    r"column .* is ambiguous",
    r"ambiguous column",
    r"duplicate column",
    r"column .* specified twice",
    r"not unique table/alias",
    r"unknown table .* in (?:field|where|on|order|group)",
    r"invalid use of group function",
    r"mixing of group columns",
    r"isn't in group by",
    r"every derived table must have its own alias",
    r"you have an error in your sql syntax",
    r"syntax error",
    r"function .* does not exist",
    r"incorrect parameter count .* function",
)
INFRASTRUCTURE_PATTERNS = (
    r"access denied",
    r"permission denied",
    r"not authorized",
    r"authentication",
    r"can't connect",
    r"cannot connect",
    r"connection (?:refused|reset|lost|closed)",
    r"lost connection",
    r"server has gone away",
    r"network",
    r"timed? out",
    r"timeout",
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


def classify_execution_error(error: BaseException | str) -> dict[str, Any]:
    """오류 코드 우선, 제한된 문자열 패턴 보조로 실행 오류를 분류한다."""
    message = str(error)
    error_code = _mysql_error_code(error) if isinstance(error, BaseException) else None
    normalized = message.casefold()

    if error_code in MISSING_TABLE_MYSQL_CODES or re.search(r"table .*doesn't exist", normalized):
        classification = "missing_table"
        retryable = False
    elif error_code in REPAIRABLE_MYSQL_CODES:
        classification = "repairable_sql"
        retryable = True
    elif error_code in INFRASTRUCTURE_MYSQL_CODES:
        classification = "infrastructure"
        retryable = False
    elif any(re.search(pattern, normalized) for pattern in INFRASTRUCTURE_PATTERNS):
        classification = "infrastructure"
        retryable = False
    elif any(re.search(pattern, normalized) for pattern in REPAIRABLE_PATTERNS):
        classification = "repairable_sql"
        retryable = True
    else:
        classification = "unknown"
        retryable = False

    return {
        "error_code": error_code,
        "classification": classification,
        "retryable": retryable,
        "message": message,
    }
