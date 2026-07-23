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
        "feedback": "컬럼 오류",
        "error": "(1054, Unknown column 'orders.total')",
        "statement_results": [{"index": 0, "sql": "SELECT 1", "row_count": 1, "columns": ["1"], "rows": [(1,)]}],
        "failed_sql_component": "main",
        "failed_statement_index": 1,
        "failed_statement_sql": "SELECT orders.total FROM orders",
        "execution_error_info": {
            "error_code": 1054,
            "classification": "repairable_sql",
            "repair_strategy": "rewrite_identifier",
            "retryable": True,
            "component": "main",
        },
        "classification": "repairable_sql",
        "repair_strategy": "rewrite_identifier",
        "repair_attempted": False,
        "repair_validation_result": {"result": "not_attempted"},
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize("code, strategy", [
    (1052, "rewrite_identifier"),
    (1054, "rewrite_identifier"),
    (1060, "rewrite_identifier"),
    (1055, "rewrite_aggregation"),
    (1111, "rewrite_aggregation"),
    (1140, "rewrite_aggregation"),
    (1064, "rewrite_syntax"),
    (1066, "rewrite_syntax"),
    (1109, "rewrite_syntax"),
    (1248, "rewrite_syntax"),
    (1305, "rewrite_syntax"),
    (1582, "rewrite_syntax"),
])
def test_mysql_sql_errors_are_classified_by_repair_strategy(code, strategy):
    info = classify_execution_error(DriverError(code, "driver detail"))

    assert info["classification"] == "repairable_sql"
    assert info["repair_strategy"] == strategy
    assert info["retryable"] is True
    assert info["error_code"] == code


@pytest.mark.parametrize("code, message, column_name, invalid_value", [
    (1265, "Data truncated: '12x' for column 'amount' at row 1", "amount", "12x"),
    (1292, "Incorrect datetime value: '0000-00-00 00:00:00' for column 'review_creation_date' at row 1", "review_creation_date", "0000-00-00 00:00:00"),
    (1366, "Incorrect integer value: 'abc' for column 'quantity' at row 1", "quantity", "abc"),
    (1411, "Incorrect datetime value: 'bad-date' for column 'created_at' in function str_to_date", "created_at", "bad-date"),
])
def test_mysql_data_value_errors_extract_repair_context(code, message, column_name, invalid_value):
    info = classify_execution_error(DriverError(code, message))

    assert info["classification"] == "repairable_data"
    assert info["repair_strategy"] == "normalize_invalid_value"
    assert info["retryable"] is True
    assert info["column_name"] == column_name
    assert info["invalid_value"] == invalid_value


@pytest.mark.parametrize("error, classification", [
    (DriverError(1146, "Table 'db.missing' doesn't exist"), "missing_table"),
    (DriverError(1045, "Access denied"), "infrastructure"),
    (RuntimeError("connection timed out"), "infrastructure"),
    (RuntimeError("unclassified driver failure"), "unknown"),
])
def test_non_repairable_execution_errors_stop(error, classification):
    info = classify_execution_error(error)

    assert info["classification"] == classification
    assert info["repair_strategy"] == "none"
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


@pytest.mark.parametrize("strategy, required_rules", [
    ("rewrite_syntax", ["문법, 함수 호출 또는 alias 오류만", "SQL의 의미, grain, 출력 컬럼"]),
    ("rewrite_identifier", ["실제로 존재하는 컬럼과 alias만", "없는 컬럼을 새로 만들거나"]),
    ("rewrite_aggregation", ["grain과 aggregation contract", "GROUP BY와 집계 표현식만"]),
    ("normalize_invalid_value", ["원본 행 삭제 없이", "CASE + 명시적 CAST + NULL", "precheck_sql"]),
])
def test_repair_prompt_contains_strategy_specific_local_rules(strategy, required_rules):
    state = repair_state(
        repair_strategy=strategy,
        execution_error_info={
            "error_code": 1292 if strategy == "normalize_invalid_value" else 1064,
            "classification": "repairable_data" if strategy == "normalize_invalid_value" else "repairable_sql",
            "repair_strategy": strategy,
            "retryable": True,
            "component": "main",
            "column_name": "review_creation_date" if strategy == "normalize_invalid_value" else None,
            "invalid_value": "0000-00-00 00:00:00" if strategy == "normalize_invalid_value" else None,
        },
    )

    prompt = repair_sql_prompt(state, state["previous_sql_draft"])

    assert f"선택된 repair 전략: {strategy}" in prompt
    for rule in required_rules:
        assert rule in prompt


def test_repair_success_returns_valid_draft_and_clears_failure_context(monkeypatch):
    state = repair_state(
        previous_sql_draft=sql_draft(sql="SELECT order_id, SUM(total) AS revenue FROM orders GROUP BY order_id")
    )
    repaired = sql_draft(sql="SELECT order_id, SUM(amount) AS revenue FROM orders GROUP BY order_id")
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: json.dumps(repaired, ensure_ascii=False))

    result = repair_node.repair_sql(state)

    assert result["generation_source"] == "semantic_llm"
    assert result["sql_draft"]["sql"].rstrip(";") == repaired["sql"]
    assert result["validation_findings"] == []
    assert result["execution_error_info"] == {}
    assert result["repair_attempted"] is True
    assert result["repair_validation_result"]["result"] == "passed"


def test_unchanged_sql_is_blocked_before_execution(monkeypatch):
    state = repair_state()
    monkeypatch.setattr(
        repair_node,
        "try_llm_json",
        lambda _: json.dumps(state["previous_sql_draft"], ensure_ascii=False),
    )

    result = repair_node.repair_sql(state)

    assert result["generation_source"] == "failed"
    assert result["generation_failure_reason"] == "repair_no_effect"
    assert result["repair_validation_result"]["result"] == "failed"
    assert result["retry_hint"]["retryable"] is False


def _datetime_repair_state(**overrides):
    invalid_value = "0000-00-00 00:00:00"
    broken = sql_draft(
        sql="SELECT CAST(review_creation_date AS DATETIME) AS review_creation_date FROM orders",
        source_column_refs=["orders.review_creation_date"],
        derived_columns=["review_creation_date"],
        output_columns=["review_creation_date"],
        reasoning="리뷰 생성일 조회",
    )
    state = repair_state(
        plan=simple_plan(
            required_columns=["orders.review_creation_date"],
            required_aggregations=[],
            dimensions=[],
            target_metrics=["리뷰 생성일"],
        ),
        schema_text=json.dumps({
            "orders": {"columns": [{"name": "review_creation_date", "type": "VARCHAR"}]},
        }),
        previous_sql_draft=broken,
        error=f"(1292, Incorrect datetime value: '{invalid_value}')",
        failed_statement_sql=broken["sql"],
        execution_error_info={
            "error_code": 1292,
            "classification": "repairable_data",
            "repair_strategy": "normalize_invalid_value",
            "retryable": True,
            "component": "main",
            "column_name": "review_creation_date",
            "invalid_value": invalid_value,
            "message": f"Incorrect datetime value: '{invalid_value}' for column 'review_creation_date'",
        },
        classification="repairable_data",
        repair_strategy="normalize_invalid_value",
    )
    state.update(overrides)
    return state


def _valid_datetime_repair():
    invalid_value = "0000-00-00 00:00:00"
    return sql_draft(
        sql=f"""WITH cleaned AS (
SELECT CASE
    WHEN review_creation_date = '{invalid_value}' THEN NULL
    ELSE CAST(review_creation_date AS DATETIME)
END AS review_creation_date_clean
FROM orders
)
SELECT review_creation_date_clean
FROM cleaned
ORDER BY review_creation_date_clean""",
        source_column_refs=["orders.review_creation_date"],
        derived_columns=["review_creation_date_clean"],
        output_columns=["review_creation_date_clean"],
        precheck_sql=(
            "SELECT COUNT(*) AS invalid_value_count FROM orders "
            f"WHERE review_creation_date = '{invalid_value}'"
        ),
        reasoning="비정상 리뷰 생성일을 NULL로 정규화",
    )


def test_datetime_value_repair_requires_normalization_and_precheck(monkeypatch):
    repaired = _valid_datetime_repair()
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: json.dumps(repaired, ensure_ascii=False))

    result = repair_node.repair_sql(_datetime_repair_state())

    assert result["generation_source"] == "semantic_llm"
    assert result["repair_validation_result"]["result"] == "passed"
    evidence = result["repair_validation_result"]["details"]["evidence"]
    assert "정규화 표현식과 정제 alias 사용" in evidence
    assert "오류 컬럼·값의 비정상 건수 precheck" in evidence


@pytest.mark.parametrize("mutation", ["missing_precheck", "missing_null", "truncation", "row_filter"])
def test_invalid_data_value_repair_is_blocked_before_execution(monkeypatch, mutation):
    repaired = _valid_datetime_repair()
    if mutation == "missing_precheck":
        repaired["precheck_sql"] = None
    elif mutation == "missing_null":
        repaired["sql"] = repaired["sql"].replace("THEN NULL", "THEN CAST('1970-01-01' AS DATETIME)")
    elif mutation == "truncation":
        repaired["sql"] = repaired["sql"].replace(
            "CAST(review_creation_date AS DATETIME)",
            "CAST(LEFT(review_creation_date, 10) AS DATETIME)",
        )
    else:
        repaired["sql"] = repaired["sql"].replace(
            "FROM orders",
            "FROM orders WHERE review_creation_date <> '0000-00-00 00:00:00'",
        )
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: json.dumps(repaired, ensure_ascii=False))

    result = repair_node.repair_sql(_datetime_repair_state())

    assert result["generation_source"] == "failed"
    assert result["generation_failure_reason"] == "repair_strategy_validation_failed"
    assert result["repair_validation_result"]["result"] == "failed"
    assert result["retry_hint"]["retryable"] is False


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


@pytest.mark.parametrize("strategy, error, broken_sql, repaired_factory", [
    (
        "rewrite_syntax",
        DriverError(1064, "You have an error in your SQL syntax"),
        "SELECT order_id SUM(amount) AS revenue FROM orders GROUP BY order_id",
        sql_draft,
    ),
    (
        "rewrite_identifier",
        DriverError(1054, "Unknown column 'total'"),
        "SELECT order_id, SUM(total) AS revenue FROM orders GROUP BY order_id",
        sql_draft,
    ),
    (
        "rewrite_aggregation",
        DriverError(1055, "Expression isn't in GROUP BY"),
        "SELECT order_id, SUM(amount) AS revenue FROM orders",
        sql_draft,
    ),
    (
        "normalize_invalid_value",
        DriverError(1292, "Incorrect datetime value: '0000-00-00 00:00:00' for column 'review_creation_date'"),
        "SELECT CAST(review_creation_date AS DATETIME) AS review_creation_date FROM orders",
        _valid_datetime_repair,
    ),
])
def test_execution_failure_repair_prevalidation_and_reexecution_flow(
    monkeypatch,
    strategy,
    error,
    broken_sql,
    repaired_factory,
):
    repaired = repaired_factory()
    if strategy == "normalize_invalid_value":
        state = _datetime_repair_state(
            sql_draft=sql_draft(
                sql=broken_sql,
                source_column_refs=["orders.review_creation_date"],
                derived_columns=["review_creation_date"],
                output_columns=["review_creation_date"],
            ),
            previous_sql_draft={},
            validation={},
            validation_findings=[],
            retry_hint={},
            retry_count=0,
            error="",
            execution_error_info={},
            classification="none",
            repair_strategy="none",
        )
    else:
        state = repair_state(
            sql_draft=sql_draft(sql=broken_sql),
            previous_sql_draft={},
            validation={},
            validation_findings=[],
            retry_hint={},
            retry_count=0,
            error="",
            execution_error_info={},
            classification="none",
            repair_strategy="none",
        )

    failed_once = {"value": False}

    def fetch(sql):
        if _normalize_for_test(sql) == _normalize_for_test(broken_sql) and not failed_once["value"]:
            failed_once["value"] = True
            raise error
        return [("ok",)]

    monkeypatch.setattr(execute_node, "can_use_live_db", lambda: True)
    monkeypatch.setattr(execute_node, "validate_mysql_sql", lambda _: None)
    monkeypatch.setattr(execute_node, "run_sql_fetchall", fetch)
    monkeypatch.setattr(repair_node, "try_llm_json", lambda _: json.dumps(repaired, ensure_ascii=False))

    first_execution = execute_node.execute_sql(state)
    first_validation = validate_sql_and_result({**state, **first_execution})
    retry_update = increase_retry({**state, **first_execution, **first_validation})
    retry_state = {**state, **first_execution, **first_validation, **retry_update}
    repair_result = repair_node.repair_sql(retry_state)
    prevalidation = prevalidate_sql({**retry_state, **repair_result})
    second_execution = execute_node.execute_sql({**retry_state, **repair_result, **prevalidation})
    final_validation = validate_sql_and_result(
        {**retry_state, **repair_result, **prevalidation, **second_execution}
    )

    assert first_execution["repair_strategy"] == strategy
    assert first_validation["retry_hint"]["retryable"] is True
    assert repair_result["repair_validation_result"]["result"] == "passed"
    assert prevalidation["validation"]["result"] == "valid"
    assert second_execution["error"] == ""
    assert final_validation["validation"]["result"] == "valid"


def _normalize_for_test(sql):
    return " ".join(str(sql).split()).casefold().rstrip(";")


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


def test_retryable_flag_without_repair_strategy_does_not_enter_repair():
    state = repair_state(
        error="분류 계약이 없는 실행 오류",
        execution_error_info={
            "classification": "repairable_sql",
            "repair_strategy": "none",
            "retryable": True,
            "message": "분류 계약이 없는 실행 오류",
        },
        validation={},
        validation_findings=[],
    )

    result = validate_sql_and_result(state)

    assert result["retry_hint"]["retryable"] is False
    assert route_after_validation({**state, **result}) == "finalize"


def test_same_error_after_first_repair_allows_one_more_repair_then_stops():
    info = classify_execution_error(DriverError(1064, "You have an error in your SQL syntax"))
    state = repair_state(
        retry_count=1,
        repair_retry_count=1,
        max_repair_retries=2,
        error=info["message"],
        execution_error_info=info,
        validation={},
        validation_findings=[],
        repair_attempted=True,
        repair_validation_result={"result": "passed"},
    )

    result = validate_sql_and_result(state)

    assert result["retry_hint"]["retryable"] is True
    assert route_after_validation({**state, **result}) == "retry"
    assert route_after_validation({**state, **result, "repair_retry_count": 2}) == "finalize"


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
    assert repair_result["generation_source"] == "semantic_llm"
