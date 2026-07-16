"""SQL 국소 repair 경로와 실행 오류 분류 회귀 테스트."""

from __future__ import annotations

import json

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql.execution_errors import classify_execution_error
from DATA_Analyst_Assistant_Agent.agents.sql.graph import route_after_retry, route_after_validation
from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as execute_node
from DATA_Analyst_Assistant_Agent.agents.sql.nodes import repair as repair_node
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.prevalidate import prevalidate_sql
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.retry import increase_retry
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.validate import validate_sql_and_result
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.repair import repair_sql_prompt


class DriverError(Exception):
    """테스트에서 MySQL 드라이버의 ``(code, message)`` 예외를 흉내 낸다."""


def simple_plan(**overrides):
    plan = {
        "route_kind": "simple",
        "selected_join_tables": ["orders"],
        "required_columns": ["orders.order_id", "orders.amount"],
        "required_aggregations": ["SUM"],
        "dimensions": ["orders.order_id"],
        "target_metrics": ["매출"],
        "validation_contract": {"expected_result_shape": "table_preview"},
    }
    plan.update(overrides)
    return plan


def sql_draft(**overrides):
    draft = {
        "sql": "SELECT order_id, SUM(amount) AS revenue FROM orders GROUP BY order_id",
        "sql_type": "select",
        "target_table": None,
        "source_tables": ["orders"],
        "source_column_refs": ["orders.order_id", "orders.amount"],
        "derived_columns": ["revenue"],
        "output_columns": ["order_id", "revenue"],
        "business_grain": None,
        "precheck_sql": None,
        "postcheck_sql": None,
        "reasoning": "주문별 매출 집계",
    }
    draft.update(overrides)
    return draft


def repair_state(**overrides):
    state = {
        "user_question": "노출되면 안 되는 원문 질문",
        "plan": simple_plan(internal_plan_secret="전체 계획 비밀"),
        "mart_design": {},
        "schema_text": json.dumps({
            "orders": {"columns": [{"name": "order_id", "type": "VARCHAR"}, {"name": "amount", "type": "DECIMAL"}]},
            "unrelated_secret_table": {"columns": [{"name": "unrelated_secret_column"}]},
        }, ensure_ascii=False),
        "integrity_text": "전체 정합성 비밀",
        "sql_draft": {},
        "previous_sql_draft": sql_draft(),
        "validation": {"result": "invalid", "feedback": "컬럼 오류"},
        "validation_findings": [{"category": "execution_error", "detail": "Unknown column orders.total"}],
        "retry_hint": {"reason_code": "execution_error", "retryable": True},
        "retry_count": 1,
        "max_retries": 2,
        "feedback": "컬럼 오류",
        "error": "(1054, Unknown column 'orders.total')",
        "statement_results": [{"index": 0, "sql": "SELECT 1", "row_count": 1, "columns": ["1"], "rows": [(1,)]}],
        "failed_sql_component": "main",
        "failed_statement_index": 1,
        "failed_statement_sql": "SELECT orders.total FROM orders",
        "execution_error_info": {"error_code": 1054, "classification": "repairable_sql", "retryable": True},
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize("code", [1052, 1054, 1055, 1060, 1064, 1066, 1111, 1140, 1248, 1305, 1582])
def test_mysql_sql_errors_are_repairable(code):
    info = classify_execution_error(DriverError(code, "driver detail"))

    assert info["classification"] == "repairable_sql"
    assert info["retryable"] is True
    assert info["error_code"] == code


@pytest.mark.parametrize("error, classification", [
    (DriverError(1146, "Table 'db.missing' doesn't exist"), "missing_table"),
    (DriverError(1045, "Access denied"), "infrastructure"),
    (RuntimeError("connection timed out"), "infrastructure"),
    (RuntimeError("unclassified driver failure"), "unknown"),
])
def test_non_repairable_execution_errors_stop(error, classification):
    info = classify_execution_error(error)

    assert info["classification"] == classification
    assert info["retryable"] is False


def test_repair_prompt_contains_only_relevant_context():
    state = repair_state()

    prompt = repair_sql_prompt(state, state["previous_sql_draft"])

    assert "Unknown column orders.total" in prompt
    assert "SELECT orders.total FROM orders" in prompt
    assert '"orders"' in prompt
    assert '"amount"' in prompt
    assert "successful_statements" in prompt
    assert "SELECT 1" in prompt
    assert "required_columns" in prompt
    assert "전체 계획 비밀" not in prompt
    assert "전체 정합성 비밀" not in prompt
    assert "unrelated_secret_table" not in prompt
    assert "unrelated_secret_column" not in prompt
    assert "노출되면 안 되는 원문 질문" not in prompt


def test_repair_success_returns_valid_draft_and_clears_failure_context(monkeypatch):
    repaired = sql_draft(sql="SELECT order_id, SUM(amount) AS revenue FROM orders GROUP BY order_id")
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: json.dumps(repaired, ensure_ascii=False))

    result = repair_node.repair_sql(repair_state())

    assert result["generation_source"] == "repair"
    assert result["sql_draft"]["sql"].rstrip(";") == repaired["sql"]
    assert result["validation_findings"] == []
    assert result["execution_error_info"] == {}


@pytest.mark.parametrize("response, failure_code", [
    ("", "llm_empty_response"),
    ("not-json", "llm_json_parse_failed"),
    ("[]", "llm_json_not_object"),
    (json.dumps(sql_draft(sql=""), ensure_ascii=False), "repair_contract_invalid"),
    (json.dumps({"sql": "CREATE TABLE bad AS SELECT 1", "sql_type": "create_table_as"}), "repair_contract_invalid"),
])
def test_repair_format_failure_keeps_previous_draft_for_next_retry(monkeypatch, response, failure_code):
    state = repair_state()
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: response)

    failed = repair_node.repair_sql(state)
    retried = increase_retry({**state, **failed})

    assert failed["generation_source"] == "failed"
    assert failed["generation_failure_reason"] == failure_code
    assert failed["retry_hint"]["details"]["generation_stage"] == "repair"
    assert failed["retry_hint"]["details"]["original_retry_hint"] == state["retry_hint"]
    assert failed["error"] == state["error"]
    assert state["validation_findings"][0] in failed["validation_findings"]
    assert retried["previous_sql_draft"] == state["previous_sql_draft"]
    assert route_after_retry({**state, **failed, **retried}) == "repair"


def test_execute_records_partial_success_and_repairable_main_error(monkeypatch):
    calls = {"count": 0}

    def fetch(sql):
        calls["count"] += 1
        if calls["count"] == 1:
            return [(1,)]
        raise DriverError(1054, "Unknown column 'bad_column'")

    monkeypatch.setattr(execute_node, "can_use_live_db", lambda: True)
    monkeypatch.setattr(execute_node, "run_sql_fetchall", fetch)
    state = repair_state(
        sql_draft=sql_draft(sql="SELECT 1; SELECT bad_column FROM orders"),
        previous_sql_draft={},
        validation={},
        validation_findings=[],
        error="",
        statement_results=[],
    )

    executed = execute_node.execute_sql(state)
    validated = validate_sql_and_result({**state, **executed})

    assert len(executed["statement_results"]) == 1
    assert executed["failed_sql_component"] == "main"
    assert executed["failed_statement_index"] == 1
    assert executed["execution_error_info"]["error_code"] == 1054
    assert validated["retry_hint"]["reason_code"] == "execution_error"
    assert validated["retry_hint"]["retryable"] is True


@pytest.mark.parametrize("component, failing_sql", [
    ("precheck", "SELECT broken_precheck FROM orders"),
    ("postcheck", "SELECT broken_postcheck FROM analytics.sales_mart"),
])
def test_execute_distinguishes_precheck_and_postcheck_failures(monkeypatch, component, failing_sql):
    draft = sql_draft(
        sql="CREATE TABLE analytics.sales_mart AS SELECT order_id, amount FROM orders",
        sql_type="create_table_as",
        target_table="analytics.sales_mart",
        precheck_sql=failing_sql if component == "precheck" else None,
        postcheck_sql=failing_sql if component == "postcheck" else None,
    )

    def fetch(sql):
        if sql == failing_sql:
            raise DriverError(1064, "You have an error in your SQL syntax")
        return [(1,)]

    monkeypatch.setattr(execute_node, "can_use_live_db", lambda: True)
    monkeypatch.setattr(execute_node, "run_sql_fetchall", fetch)
    monkeypatch.setattr(execute_node, "is_safe_mart_sql", lambda *_: (True, ""))
    monkeypatch.setattr(execute_node, "ensure_target_schema_exists", lambda *_: None)
    monkeypatch.setattr(execute_node, "run_sql_commit", lambda *_: None)

    result = execute_node.execute_sql(repair_state(sql_draft=draft))

    assert result["failed_sql_component"] == component
    assert result["failed_statement_sql"] == failing_sql
    assert result["execution_error_info"]["classification"] == "repairable_sql"


def test_prevalidation_missing_identifier_is_terminal_without_repair():
    state = repair_state(
        schema_text=json.dumps({"orders": {"columns": [{"name": "order_id"}]}}),
        sql_draft=sql_draft(source_column_refs=["orders.missing_column"]),
        previous_sql_draft={},
        validation={},
        validation_findings=[],
        retry_hint={},
        error="",
    )

    result = prevalidate_sql(state)
    routed_state = {**state, **result}

    assert result["retry_hint"]["reason_code"] == "missing_column"
    assert result["retry_hint"]["retryable"] is False
    assert route_after_validation(routed_state) == "finalize"


@pytest.mark.parametrize("reason_code", ["empty_result", "postcheck_failed"])
def test_result_data_failures_are_terminal(reason_code):
    state = repair_state(
        validation={"result": "invalid"},
        retry_hint={"reason_code": reason_code, "retryable": False},
    )

    assert route_after_validation(state) == "finalize"


def test_database_missing_table_and_infrastructure_errors_are_terminal():
    for error in (
        DriverError(1146, "Table 'db.missing' doesn't exist"),
        DriverError(2003, "Can't connect to MySQL server"),
    ):
        info = classify_execution_error(error)
        state = repair_state(
            error=str(error),
            execution_error_info=info,
            validation={},
            validation_findings=[],
        )
        result = validate_sql_and_result(state)

        assert result["retry_hint"]["retryable"] is False
        assert route_after_validation({**state, **result}) == "finalize"


def test_comprehensive_ctas_repair_reenters_full_validation_and_execution(monkeypatch):
    plan = {
        "route_kind": "comprehensive",
        "selected_join_tables": ["orders"],
        "required_columns": ["orders.order_id", "orders.amount"],
        "required_aggregations": [],
        "dimensions": ["orders.order_id"],
        "target_metrics": ["매출 원천값"],
        "validation_contract": {},
    }
    mart_design = {
        "grain": "order_id",
        "grain_columns": ["order_id"],
        "column_plan": [
            {"output_column": "order_id", "role": "dimension", "aggregation_method": "none"},
            {"output_column": "amount", "role": "measure", "aggregation_method": "none"},
        ],
        "aggregation_policy": "preserve_common_grain",
    }
    broken = sql_draft(
        sql="CREATE TABLE analytics.sales_mart AS SELECT order_id, amount, amount FROM orders",
        sql_type="create_table_as",
        target_table="analytics.sales_mart",
        output_columns=["order_id", "amount", "amount"],
        business_grain="order_id",
    )
    repaired = sql_draft(
        sql="CREATE TABLE analytics.sales_mart AS SELECT order_id, amount FROM orders",
        sql_type="create_table_as",
        target_table="analytics.sales_mart",
        output_columns=["order_id", "amount"],
        business_grain="order_id",
    )
    state = repair_state(
        plan=plan,
        mart_design=mart_design,
        previous_sql_draft=broken,
        validation_findings=[{"category": "execution_error", "detail": "Duplicate column name 'amount'"}],
        error="(1060, Duplicate column name 'amount')",
        failed_statement_sql=broken["sql"],
        execution_error_info={"error_code": 1060, "classification": "repairable_sql", "retryable": True},
    )
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: json.dumps(repaired, ensure_ascii=False))
    monkeypatch.setattr(execute_node, "can_use_live_db", lambda: False)

    repair_result = repair_node.repair_sql(state)
    prevalidation = prevalidate_sql({**state, **repair_result})
    executed = execute_node.execute_sql({**state, **repair_result, **prevalidation})
    validated = validate_sql_and_result({**state, **repair_result, **prevalidation, **executed})

    assert prevalidation["validation"]["result"] == "valid"
    assert executed["error"] == ""
    assert validated["validation"]["result"] == "valid"
    assert repair_result["generation_source"] == "repair"
