from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import extract_target_schema
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.execute import execute_sql


def test_extract_target_schema_parses_qualified_name() -> None:
    assert extract_target_schema("analytics.payment_datamart") == "analytics"
    assert extract_target_schema("`analytics`.`payment_datamart`") == "analytics"
    assert extract_target_schema("payment_datamart") is None


def test_execute_sql_creates_target_schema_before_mart_commit(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.nodes.execute.validate_mysql_sql",
        lambda sql: "",
    )
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.nodes.execute.can_use_live_db",
        lambda: True,
    )
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.nodes.execute.ensure_target_schema_exists",
        lambda target_table: calls.append(("ensure_schema", str(target_table))) or "analytics",
    )
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.nodes.execute.run_sql_commit",
        lambda sql: calls.append(("commit", sql)),
    )
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.nodes.execute.run_sql_fetchall",
        lambda sql: [(1,)],
    )

    result = execute_sql(
        {
            "sql_draft": {
                "sql": "CREATE TABLE analytics.payment_datamart AS SELECT order_payments.* FROM order_payments ;",
                "sql_type": "create_table_as",
                "target_table": "analytics.payment_datamart",
                "postcheck_sql": "SELECT COUNT(*) FROM analytics.payment_datamart;",
            }
        }
    )

    assert calls[0] == ("ensure_schema", "analytics.payment_datamart")
    assert calls[1][0] == "commit"
    assert result["error"] == ""
