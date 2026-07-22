from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

import pymysql

# 사용자에게 보여줄 필요 없는 MySQL 내부 시스템 DB들
SYSTEM_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}
INTERNAL_ROW_ID = "__daaat_row_id"
UNINDEXABLE_SORT_TYPES = {
    "blob",
    "longblob",
    "mediumblob",
    "tinyblob",
    "text",
    "longtext",
    "mediumtext",
    "tinytext",
    "json",
    "geometry",
}

# DB/테이블 이름 검증: 글자(한글 포함)/숫자/언더스코어만 허용
SAFE_IDENTIFIER = re.compile(r"^\w+$")

def quote_identifier(name: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError(f"허용되지 않는 이름입니다: {name}")
    return f"`{name}`"

def connect(host: str, port: int, user: str, password: str, database: str | None = None):
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=30,
    )

def list_databases(host: str, port: int, user: str, password: str) -> list[str]:
    conn = connect(host, port, user, password)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW DATABASES")
            names = [row[0] for row in cur.fetchall()]
        return [n for n in names if n.lower() not in SYSTEM_DATABASES]
    finally:
        conn.close()

def list_tables(host: str, port: int, user: str, password: str, database: str) -> list[str]:
    quote_identifier(database)
    conn = connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

# 선택한 DB 안의 뷰를 제외한 실제 테이블 목록
def list_base_tables(host: str, port: int, user: str, password: str, database: str) -> list[str]:
    quote_identifier(database)
    conn = connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def preview_table(
    host: str, port: int, user: str, password: str,
    database: str, table: str, limit: int = 50,
) -> tuple[list[str], list[dict]]:
    """테이블 데이터를 (컬럼 목록, 행 목록)으로 돌려준다."""
    quote_identifier(database)
    quoted_table = quote_identifier(table)  # 검증 + 백틱
    conn = connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {quoted_table} LIMIT %s", (int(limit),))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description or []]
        return columns, [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


@dataclass(frozen=True)
class TablePreviewPage:
    columns: list[str]
    rows: list[dict]
    next_cursor: str | None
    has_more: bool


@dataclass(frozen=True)
class _TableMetadata:
    columns: list[str]
    nullable: set[str]
    primary_key: list[str]
    column_types: dict[str, str]

    @property
    def has_internal_row_id(self) -> bool:
        return INTERNAL_ROW_ID in self.primary_key


@dataclass(frozen=True)
class _OrderTerm:
    expression: str
    column: str
    direction: str
    null_rank: bool = False

    def value_from(self, row: dict) -> Any:
        value = row[self.column]
        return int(value is None) if self.null_rank else value


def _encode_cursor_value(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "plain", "value": value}
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"type": "time", "value": value.isoformat()}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.urlsafe_b64encode(value).decode("ascii")}
    raise ValueError(f"cursor로 변환할 수 없는 값입니다: {type(value).__name__}")


def _decode_cursor_value(item: dict[str, Any]) -> Any:
    if not isinstance(item, dict):
        raise ValueError("유효하지 않은 cursor 값입니다.")
    kind = item.get("type")
    value = item.get("value")
    try:
        if kind == "plain":
            return value
        if kind == "decimal":
            return Decimal(value)
        if kind == "datetime":
            return datetime.fromisoformat(value)
        if kind == "date":
            return date.fromisoformat(value)
        if kind == "time":
            return time.fromisoformat(value)
        if kind == "bytes":
            return base64.urlsafe_b64decode(value.encode("ascii"))
    except (ValueError, TypeError, InvalidOperation, binascii.Error) as exc:
        raise ValueError("유효하지 않은 cursor 값입니다.") from exc
    raise ValueError("알 수 없는 cursor 값 형식입니다.")


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error) as exc:
        raise ValueError("유효하지 않은 cursor입니다.") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("유효하지 않은 cursor입니다.")
    return payload


def _rows_as_dicts(cur, rows: tuple) -> list[dict]:
    columns = [desc[0] for desc in cur.description or []]
    return [dict(zip(columns, row)) for row in rows]


def _table_metadata(cur, quoted_table: str) -> _TableMetadata:
    cur.execute(f"SHOW COLUMNS FROM {quoted_table}")
    column_rows = _rows_as_dicts(cur, cur.fetchall())
    columns = [row["Field"] for row in column_rows if row["Field"] != INTERNAL_ROW_ID]
    nullable = {
        row["Field"]
        for row in column_rows
        if row["Field"] != INTERNAL_ROW_ID and row["Null"] == "YES"
    }
    column_types = {row["Field"]: str(row["Type"]).lower() for row in column_rows}

    cur.execute(f"SHOW KEYS FROM {quoted_table} WHERE Key_name = 'PRIMARY'")
    key_rows = _rows_as_dicts(cur, cur.fetchall())
    primary_key = [row["Column_name"] for row in sorted(key_rows, key=lambda row: row["Seq_in_index"])]
    return _TableMetadata(columns, nullable, primary_key, column_types)


def ensure_internal_row_id(cur, quoted_table: str) -> _TableMetadata:
    """PK가 없는 복사 테이블에 API에서 숨겨지는 내부 PK를 추가한다."""
    metadata = _table_metadata(cur, quoted_table)
    if metadata.primary_key:
        return metadata
    if INTERNAL_ROW_ID in metadata.column_types:
        raise ValueError(f"내부 관리 컬럼과 이름이 충돌합니다: {INTERNAL_ROW_ID}")

    quoted_row_id = quote_identifier(INTERNAL_ROW_ID)
    try:
        cur.execute(
            f"ALTER TABLE {quoted_table} "
            f"ADD COLUMN {quoted_row_id} BIGINT UNSIGNED NOT NULL AUTO_INCREMENT INVISIBLE PRIMARY KEY"
        )
    except pymysql.MySQLError as exc:
        if not exc.args or exc.args[0] not in {1060, 1068}:
            raise
    return _table_metadata(cur, quoted_table)


def _sort_index_name(table: str, sort_by: str) -> str:
    digest = hashlib.sha256(f"{table}:{sort_by}".encode("utf-8")).hexdigest()[:16]
    return f"idx_daaat_sort_{digest}"


def _base_mysql_type(mysql_type: str) -> str:
    return mysql_type.split("(", 1)[0].strip()


def _effective_nullable_columns(
    cur,
    quoted_table: str,
    sort_by: str | None,
    declared_nullable: set[str],
) -> set[str]:
    """복사 데이터에 실제 NULL이 있을 때만 NULL 순서용 표현식을 사용한다."""
    effective = set(declared_nullable)
    if sort_by is None or sort_by not in effective:
        return effective

    quoted_sort = quote_identifier(sort_by)
    cur.execute(f"SELECT EXISTS(SELECT 1 FROM {quoted_table} WHERE {quoted_sort} IS NULL LIMIT 1)")
    if not bool(cur.fetchone()[0]):
        effective.remove(sort_by)
    return effective


def _ensure_sort_index(
    cur,
    quoted_table: str,
    table: str,
    sort_by: str,
    metadata: _TableMetadata,
    effective_nullable: set[str],
) -> bool:
    """정렬 컬럼과 PK 순서의 보조 인덱스를 한 번만 생성한다."""
    if _base_mysql_type(metadata.column_types[sort_by]) in UNINDEXABLE_SORT_TYPES:
        return False

    index_name = _sort_index_name(table, sort_by)
    quoted_index = quote_identifier(index_name)
    cur.execute(f"SHOW INDEX FROM {quoted_table} WHERE Key_name = %s", (index_name,))
    if cur.fetchall():
        return True

    quoted_sort = quote_identifier(sort_by)
    index_parts: list[str] = []
    if sort_by in effective_nullable:
        index_parts.append(f"(({quoted_sort} IS NULL))")
    index_parts.append(quoted_sort)
    index_parts.extend(
        quote_identifier(column)
        for column in metadata.primary_key
        if column != sort_by
    )
    add_index = f"ADD INDEX {quoted_index} ({', '.join(index_parts)})"

    try:
        cur.execute(f"ALTER TABLE {quoted_table} {add_index}, ALGORITHM=INPLACE, LOCK=NONE")
    except pymysql.MySQLError as exc:
        if exc.args and exc.args[0] == 1061:
            return True
        if exc.args and exc.args[0] == 1846:
            try:
                cur.execute(f"ALTER TABLE {quoted_table} {add_index}, ALGORITHM=INPLACE, LOCK=SHARED")
            except pymysql.MySQLError as retry_exc:
                if retry_exc.args and retry_exc.args[0] == 1061:
                    return True
                if retry_exc.args and retry_exc.args[0] == 1846:
                    cur.execute(f"ALTER TABLE {quoted_table} {add_index}, ALGORITHM=COPY")
                    return True
                raise
            return True
        # 너무 긴 문자열 키처럼 인덱싱 자체가 불가능한 컬럼은 기존 정렬로 안전하게 동작한다.
        if exc.args and exc.args[0] in {1071, 1170}:
            return False
        raise
    return True


def _build_order_terms(
    sort_by: str | None,
    sort_order: str,
    nullable: set[str],
    primary_key: list[str],
) -> list[_OrderTerm]:
    terms: list[_OrderTerm] = []
    if sort_by is not None:
        quoted_sort = quote_identifier(sort_by)
        if sort_by in nullable:
            terms.append(_OrderTerm(f"({quoted_sort} IS NULL)", sort_by, sort_order.upper(), null_rank=True))
        terms.append(_OrderTerm(quoted_sort, sort_by, sort_order.upper()))
        terms.extend(
            _OrderTerm(quote_identifier(column), column, sort_order.upper())
            for column in primary_key
            if column != sort_by
        )
    else:
        terms.extend(
            _OrderTerm(quote_identifier(column), column, sort_order.upper())
            for column in primary_key
        )
    return terms


def _build_keyset_condition(terms: list[_OrderTerm], values: list[Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for index, (term, value) in enumerate(zip(terms, values, strict=True)):
        # NULL 자체는 대소 비교가 불가능하지만, 뒤의 PK tie-breaker 비교는 계속 사용할 수 있다.
        if value is None:
            continue
        equality = [f"{previous.expression} <=> %s" for previous in terms[:index]]
        comparator = ">" if term.direction == "ASC" else "<"
        clauses.append("(" + " AND ".join([*equality, f"{term.expression} {comparator} %s"]) + ")")
        params.extend(values[:index])
        params.append(value)
    if not clauses:
        raise ValueError("cursor에서 다음 페이지 조건을 만들 수 없습니다.")
    return " OR ".join(clauses), params


def preview_table_page(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
    limit: int = 50,
    sort_by: str | None = None,
    sort_order: str = "asc",
    cursor: str | None = None,
) -> TablePreviewPage:
    """정렬을 유지하며 테이블의 다음 페이지를 반환한다."""
    quote_identifier(database)
    quoted_table = quote_identifier(table)
    normalized_order = sort_order.lower()
    if normalized_order not in {"asc", "desc"}:
        raise ValueError("sort_order는 asc 또는 desc여야 합니다.")

    conn = connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            metadata = _table_metadata(cur, quoted_table)
            if sort_by is not None and sort_by not in metadata.columns:
                raise ValueError(f"존재하지 않는 정렬 컬럼입니다: {sort_by}")

            # 새 ingest 이전에 생성된 세션 테이블도 최초 정렬 시 같은 구조로 보강한다.
            if sort_by is not None and not metadata.primary_key:
                metadata = ensure_internal_row_id(cur, quoted_table)
            effective_nullable = _effective_nullable_columns(
                cur,
                quoted_table,
                sort_by,
                metadata.nullable,
            )
            if sort_by is not None:
                _ensure_sort_index(
                    cur,
                    quoted_table,
                    table,
                    sort_by,
                    metadata,
                    effective_nullable,
                )

            terms = _build_order_terms(
                sort_by,
                normalized_order,
                effective_nullable,
                metadata.primary_key,
            )
            mode = "keyset" if metadata.primary_key else "offset"
            payload = _decode_cursor(cursor) if cursor else None
            if payload and (
                payload.get("database") != database
                or payload.get("table") != table
                or payload.get("sort_by") != sort_by
                or payload.get("sort_order") != normalized_order
                or payload.get("mode") != mode
            ):
                raise ValueError("현재 정렬 조건과 cursor가 일치하지 않습니다.")

            select_columns = "*"
            if metadata.has_internal_row_id:
                select_columns += f", {quote_identifier(INTERNAL_ROW_ID)}"
            sql = f"SELECT {select_columns} FROM {quoted_table}"
            params: list[Any] = []
            if mode == "keyset" and payload:
                encoded_values = payload.get("values")
                if not isinstance(encoded_values, list) or len(encoded_values) != len(terms):
                    raise ValueError("유효하지 않은 cursor입니다.")
                values = [_decode_cursor_value(item) for item in encoded_values]
                condition, condition_params = _build_keyset_condition(terms, values)
                sql += f" WHERE {condition}"
                params.extend(condition_params)

            if terms:
                sql += " ORDER BY " + ", ".join(f"{term.expression} {term.direction}" for term in terms)

            offset = 0
            if mode == "offset" and payload:
                offset = payload.get("offset")
                if not isinstance(offset, int) or offset < 0:
                    raise ValueError("유효하지 않은 cursor입니다.")
            sql += " LIMIT %s"
            params.append(int(limit) + 1)
            if mode == "offset":
                sql += " OFFSET %s"
                params.append(offset)

            cur.execute(sql, tuple(params))
            fetched = cur.fetchall()
            fetched_columns = [desc[0] for desc in cur.description or []]
            all_rows = [dict(zip(fetched_columns, row)) for row in fetched]

        has_more = len(all_rows) > limit
        cursor_rows = all_rows[:limit]
        next_cursor = None
        if has_more and cursor_rows:
            next_payload: dict[str, Any] = {
                "version": 1,
                "database": database,
                "table": table,
                "sort_by": sort_by,
                "sort_order": normalized_order,
                "mode": mode,
            }
            if mode == "keyset":
                next_payload["values"] = [
                    _encode_cursor_value(term.value_from(cursor_rows[-1])) for term in terms
                ]
            else:
                next_payload["offset"] = offset + limit
            next_cursor = _encode_cursor(next_payload)

        result_columns = [column for column in fetched_columns if column != INTERNAL_ROW_ID]
        rows = [
            {column: row[column] for column in result_columns}
            for row in cursor_rows
        ]
        return TablePreviewPage(result_columns, rows, next_cursor, has_more)
    finally:
        conn.close()


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def stream_table_csv(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
    sort_by: str | None = None,
    sort_order: str = "asc",
    batch_size: int = 1000,
) -> Iterator[str]:
    """테이블 전체를 메모리에 적재하지 않고 UTF-8 CSV로 스트리밍한다."""
    quote_identifier(database)
    quoted_table = quote_identifier(table)
    normalized_order = sort_order.lower()
    if normalized_order not in {"asc", "desc"}:
        raise ValueError("sort_order는 asc 또는 desc여야 합니다.")

    conn = connect(host, port, user, password, database=database)
    stream_cursor = None
    try:
        with conn.cursor() as cur:
            metadata = _table_metadata(cur, quoted_table)
            if sort_by is not None and sort_by not in metadata.columns:
                raise ValueError(f"존재하지 않는 정렬 컬럼입니다: {sort_by}")

            if sort_by is not None and not metadata.primary_key:
                metadata = ensure_internal_row_id(cur, quoted_table)
            effective_nullable = _effective_nullable_columns(
                cur,
                quoted_table,
                sort_by,
                metadata.nullable,
            )
            if sort_by is not None:
                _ensure_sort_index(
                    cur,
                    quoted_table,
                    table,
                    sort_by,
                    metadata,
                    effective_nullable,
                )

            terms = _build_order_terms(
                sort_by,
                normalized_order,
                effective_nullable,
                metadata.primary_key,
            )

        selected_columns = ", ".join(quote_identifier(column) for column in metadata.columns)
        sql = f"SELECT {selected_columns} FROM {quoted_table}"
        if terms:
            sql += " ORDER BY " + ", ".join(
                f"{term.expression} {term.direction}" for term in terms
            )

        stream_cursor = conn.cursor(pymysql.cursors.SSCursor)
        stream_cursor.execute(sql)
    except Exception:
        if stream_cursor is not None:
            stream_cursor.close()
        conn.close()
        raise

    def generate() -> Iterator[str]:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\r\n")

        def flush_buffer() -> str:
            chunk = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            return chunk

        try:
            buffer.write("\ufeff")
            writer.writerow(metadata.columns)
            yield flush_buffer()

            while rows := stream_cursor.fetchmany(batch_size):
                writer.writerows(
                    [_csv_cell(value) for value in row]
                    for row in rows
                )
                yield flush_buffer()
        finally:
            stream_cursor.close()
            conn.close()

    return generate()
