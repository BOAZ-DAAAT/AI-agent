from __future__ import annotations

from collections.abc import Callable
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.mysql import db
from backend.session import routes as session_routes


class FakeCursor:
    def __init__(
        self,
        columns: list[tuple],
        keys: list[tuple],
        select_rows: list[tuple],
        select_columns: list[str],
        indexes: set[str] | None = None,
        has_nulls: bool = False,
    ) -> None:
        self._columns = columns
        self._keys = keys
        self._select_rows = select_rows
        self._select_columns = select_columns
        self._indexes = indexes or set()
        self._has_nulls = has_nulls
        self._rows: list[tuple] = []
        self.description: list[tuple] = []
        self.executions: list[tuple[str, tuple | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executions.append((sql, params))
        if sql.startswith("SHOW COLUMNS"):
            self.description = [("Field",), ("Type",), ("Null",)]
            self._rows = self._columns
        elif sql.startswith("SHOW KEYS"):
            self.description = [("Column_name",), ("Seq_in_index",)]
            self._rows = self._keys
        elif sql.startswith("SHOW INDEX"):
            self.description = [("Key_name",)]
            requested = params[0] if params else None
            self._rows = [(requested,)] if requested in self._indexes else []
        elif sql.startswith("SELECT EXISTS"):
            self.description = [("EXISTS",)]
            self._rows = [(int(self._has_nulls),)]
        elif "ADD COLUMN `__daaat_row_id`" in sql:
            self._columns.append((db.INTERNAL_ROW_ID, "bigint unsigned", "NO"))
            self._keys = [(db.INTERNAL_ROW_ID, 1)]
            self.description = []
            self._rows = []
        elif "ADD INDEX" in sql:
            match = re.search(r"ADD INDEX `([^`]+)`", sql)
            assert match is not None
            self._indexes.add(match.group(1))
            self.description = []
            self._rows = []
        else:
            selected_columns = list(self._select_columns)
            if f"`{db.INTERNAL_ROW_ID}`" in sql and db.INTERNAL_ROW_ID not in selected_columns:
                selected_columns.append(db.INTERNAL_ROW_ID)
            self.description = [(column,) for column in selected_columns]
            self._rows = self._select_rows

    def fetchall(self) -> tuple:
        return tuple(self._rows)

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


class FakeStreamCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.offset = 0
        self.sql = ""
        self.closed = False

    def execute(self, sql: str) -> None:
        self.sql = sql

    def fetchmany(self, size: int) -> tuple:
        batch = tuple(self.rows[self.offset:self.offset + size])
        self.offset += len(batch)
        return batch

    def close(self) -> None:
        self.closed = True


class FakeCsvConnection:
    def __init__(self, metadata_cursor: FakeCursor, stream_cursor: FakeStreamCursor) -> None:
        self.metadata_cursor = metadata_cursor
        self.stream_cursor = stream_cursor
        self.closed = False

    def cursor(self, cursor_type=None):
        if cursor_type is not None:
            return self.stream_cursor
        return self.metadata_cursor

    def close(self) -> None:
        self.closed = True


def _column(name: str, nullable: str = "NO") -> tuple:
    return name, "varchar(255)", nullable


def _key(name: str, sequence: int = 1) -> tuple:
    return name, sequence


def _install_connections(
    monkeypatch: pytest.MonkeyPatch,
    cursors: list[FakeCursor],
) -> list[FakeConnection]:
    connections = [FakeConnection(cursor) for cursor in cursors]
    pending = iter(connections)
    connect: Callable[..., FakeConnection] = lambda *args, **kwargs: next(pending)
    monkeypatch.setattr(db, "connect", connect)
    return connections


def test_preview_table_page_uses_primary_key_as_sort_tie_breaker(monkeypatch) -> None:
    first_cursor = FakeCursor(
        columns=[_column("id"), _column("amount")],
        keys=[_key("id")],
        select_rows=[(1, 10), (2, 10), (3, 20)],
        select_columns=["id", "amount"],
    )
    second_cursor = FakeCursor(
        columns=[_column("id"), _column("amount")],
        keys=[_key("id")],
        select_rows=[(3, 20)],
        select_columns=["id", "amount"],
    )
    connections = _install_connections(monkeypatch, [first_cursor, second_cursor])

    first = db.preview_table_page(
        "host", 3306, "user", "password", "database", "orders", 2, "amount", "asc"
    )
    second = db.preview_table_page(
        "host",
        3306,
        "user",
        "password",
        "database",
        "orders",
        2,
        "amount",
        "asc",
        first.next_cursor,
    )

    assert first.rows == [{"id": 1, "amount": 10}, {"id": 2, "amount": 10}]
    assert first.has_more is True
    assert first.next_cursor is not None
    assert second.rows == [{"id": 3, "amount": 20}]
    assert second.has_more is False

    second_sql, second_params = second_cursor.executions[-1]
    assert "ORDER BY `amount` ASC, `id` ASC" in second_sql
    assert "(`amount` > %s) OR (`amount` <=> %s AND `id` > %s)" in second_sql
    assert second_params == (10, 10, 2, 3)
    assert all(connection.closed for connection in connections)


def test_preview_table_page_adds_hidden_primary_key_and_uses_keyset(monkeypatch) -> None:
    first_cursor = FakeCursor(
        columns=[_column("name")],
        keys=[],
        select_rows=[("c", 1), ("b", 2), ("a", 3)],
        select_columns=["name"],
    )
    second_cursor = FakeCursor(
        columns=[_column("name"), (db.INTERNAL_ROW_ID, "bigint unsigned", "NO")],
        keys=[_key(db.INTERNAL_ROW_ID)],
        select_rows=[("a", 3)],
        select_columns=["name"],
        indexes={db._sort_index_name("labels", "name")},
    )
    _install_connections(monkeypatch, [first_cursor, second_cursor])

    first = db.preview_table_page(
        "host", 3306, "user", "password", "database", "labels", 2, "name", "desc"
    )
    second = db.preview_table_page(
        "host",
        3306,
        "user",
        "password",
        "database",
        "labels",
        2,
        "name",
        "desc",
        first.next_cursor,
    )

    second_sql, second_params = second_cursor.executions[-1]
    assert "ORDER BY `name` DESC, `__daaat_row_id` DESC" in second_sql
    assert "(`name` < %s) OR (`name` <=> %s AND `__daaat_row_id` < %s)" in second_sql
    assert second_params == ("b", "b", 2, 3)
    assert db.INTERNAL_ROW_ID not in first.columns
    assert all(db.INTERNAL_ROW_ID not in row for row in first.rows)
    assert any("ADD COLUMN `__daaat_row_id`" in sql for sql, _ in first_cursor.executions)
    assert any("ADD INDEX" in sql for sql, _ in first_cursor.executions)
    assert second.next_cursor is None


def test_preview_table_page_rejects_unknown_sort_column(monkeypatch) -> None:
    cursor = FakeCursor(
        columns=[_column("id")],
        keys=[_key("id")],
        select_rows=[],
        select_columns=["id"],
    )
    connections = _install_connections(monkeypatch, [cursor])

    with pytest.raises(ValueError, match="존재하지 않는 정렬 컬럼"):
        db.preview_table_page(
            "host", 3306, "user", "password", "database", "orders", sort_by="missing"
        )

    assert connections[0].closed is True


def test_stream_table_csv_exports_all_visible_columns_in_batches(monkeypatch) -> None:
    metadata_cursor = FakeCursor(
        columns=[_column("id"), _column("name"), _column("note")],
        keys=[_key("id")],
        select_rows=[],
        select_columns=[],
        indexes={db._sort_index_name("customers", "name")},
    )
    stream_cursor = FakeStreamCursor([
        (2, "김철수", "쉼표, 포함"),
        (1, "Alice", None),
    ])
    connection = FakeCsvConnection(metadata_cursor, stream_cursor)
    monkeypatch.setattr(db, "connect", lambda *args, **kwargs: connection)

    stream = db.stream_table_csv(
        "host",
        3306,
        "user",
        "password",
        "database",
        "customers",
        sort_by="name",
        sort_order="desc",
        batch_size=1,
    )

    assert connection.closed is False
    content = "".join(stream)

    assert content == (
        '\ufeffid,name,note\r\n'
        '2,김철수,"쉼표, 포함"\r\n'
        '1,Alice,\r\n'
    )
    assert "SELECT `id`, `name`, `note` FROM `customers`" in stream_cursor.sql
    assert "ORDER BY `name` DESC, `id` DESC" in stream_cursor.sql
    assert stream_cursor.closed is True
    assert connection.closed is True


def test_declared_nullable_column_without_nulls_uses_plain_index(monkeypatch) -> None:
    cursor = FakeCursor(
        columns=[_column("id"), _column("state", nullable="YES")],
        keys=[_key("id")],
        select_rows=[(1, "SP")],
        select_columns=["id", "state"],
        has_nulls=False,
    )
    _install_connections(monkeypatch, [cursor])

    page = db.preview_table_page(
        "host", 3306, "user", "password", "database", "customers", 1, "state", "asc"
    )

    index_sql = next(sql for sql, _ in cursor.executions if "ADD INDEX" in sql)
    select_sql = cursor.executions[-1][0]
    assert "IS NULL" not in index_sql
    assert "ORDER BY `state` ASC, `id` ASC" in select_sql
    assert page.rows == [{"id": 1, "state": "SP"}]


def test_preview_table_page_rejects_cursor_from_different_sort(monkeypatch) -> None:
    first_cursor = FakeCursor(
        columns=[_column("id"), _column("amount")],
        keys=[_key("id")],
        select_rows=[(1, 10), (2, 20)],
        select_columns=["id", "amount"],
    )
    second_cursor = FakeCursor(
        columns=[_column("id"), _column("amount")],
        keys=[_key("id")],
        select_rows=[],
        select_columns=["id", "amount"],
    )
    _install_connections(monkeypatch, [first_cursor, second_cursor])
    first = db.preview_table_page(
        "host", 3306, "user", "password", "database", "orders", 1, "amount", "asc"
    )

    with pytest.raises(ValueError, match="정렬 조건과 cursor"):
        db.preview_table_page(
            "host",
            3306,
            "user",
            "password",
            "database",
            "orders",
            1,
            "amount",
            "desc",
            first.next_cursor,
        )


def test_session_preview_route_exposes_sort_and_page_contract(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_preview(
        session_id: str,
        username: str,
        table: str,
        limit: int,
        sort_by: str | None,
        sort_order: str,
        cursor: str | None,
    ) -> db.TablePreviewPage:
        seen.update(
            session_id=session_id,
            username=username,
            table=table,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            cursor=cursor,
        )
        return db.TablePreviewPage(
            columns=["id", "amount"],
            rows=[{"id": 2, "amount": 20}],
            next_cursor="next-page",
            has_more=True,
        )

    monkeypatch.setattr(session_routes, "preview_session_table", fake_preview)
    app = FastAPI()
    app.include_router(session_routes.router)
    app.dependency_overrides[session_routes.current_username] = lambda: "tester"
    client = TestClient(app)

    response = client.get(
        "/sessions/session-1/tables/orders/preview",
        params={
            "limit": 50,
            "sort_by": "amount",
            "sort_order": "desc",
            "cursor": "current-page",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "columns": ["id", "amount"],
        "rows": [{"id": 2, "amount": 20}],
        "page_info": {"next_cursor": "next-page", "has_more": True},
    }
    assert seen == {
        "session_id": "session-1",
        "username": "tester",
        "table": "orders",
        "limit": 50,
        "sort_by": "amount",
        "sort_order": "desc",
        "cursor": "current-page",
    }
