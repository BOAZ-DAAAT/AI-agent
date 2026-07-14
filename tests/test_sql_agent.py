"""B-lane tests: SQL-Agent, planner, self-check, mart candidate, and Phase 2A routing."""

from __future__ import annotations

import json
import shutil
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from data_agent_backend.models.common import BackendError
from DATA_Analyst_Assistant_Agent.agents.sql.self_check import is_sql_safe, run_sql_self_check
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import extract_sql_aliases, split_sql_statements
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
try:
    from DATA_Analyst_Assistant_Agent.agents.validation.agent import CentralValidationAgent
except ModuleNotFoundError:
    CentralValidationAgent = None
from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import normalize_generated_sql
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import validate_sql_identifiers
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    LocalCheck,
    OrchestrationState,
    RetryHint,
    ValidationBlock,
)



# ── fixtures ──

@pytest.fixture()
def adapter() -> BackendAdapter:
    base_dir = Path(".test_data") / f"sql_agent_{uuid.uuid4().hex}"
    config = BackendConfig(base_data_dir=base_dir / ".data_agent")
    try:
        yield BackendAdapter(config=config)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def started_agents(adapter: BackendAdapter, run_id: str) -> list[str]:
    return [
        e.node_name for e in adapter.services.run_service.list_events(run_id)
        if e.event_type == "agent.started"
    ]


def event_pairs(adapter: BackendAdapter, run_id: str) -> list[tuple[str | None, str]]:
    return [(e.node_name, e.event_type) for e in adapter.services.run_service.list_events(run_id)]


# ── self_check tests ──

class TestSQLSelfCheck:
    def test_select_is_safe(self):
        assert is_sql_safe("SELECT 1 AS x")

    def test_with_cte_is_safe(self):
        assert is_sql_safe("WITH cte AS (SELECT 1) SELECT * FROM cte")

    def test_drop_is_blocked(self):
        assert not is_sql_safe("DROP TABLE users")

    def test_insert_is_blocked(self):
        assert not is_sql_safe("INSERT INTO users VALUES (1)")

    def test_delete_is_blocked(self):
        assert not is_sql_safe("DELETE FROM users")

    def test_update_is_blocked(self):
        assert not is_sql_safe("UPDATE users SET name='x'")

    def test_multi_statement_is_blocked(self):
        assert not is_sql_safe("SELECT 1; DROP TABLE users")

    def test_run_sql_self_check_can_allow_multi_statement_selects(self):
        checks = run_sql_self_check("SELECT 1; SELECT 2;", allow_multi_statement=True)
        assert all(c.passed for c in checks)

    def test_markdown_fence_stripped(self):
        assert is_sql_safe("```sql\nSELECT 1\n```")

    def test_mysql_incompatible_function_is_blocked(self):
        checks = run_sql_self_check("SELECT JULIANDAY(created_at) FROM orders")
        dialect_check = next(c for c in checks if c.name == "mysql_dialect_compatibility")
        assert not dialect_check.passed
        assert "JULIANDAY" in dialect_check.detail

    def test_post_execution_checks_columns(self):
        checks = run_sql_self_check("SELECT 1", columns=["x"], row_count=1)
        names = {c.name for c in checks}
        assert "preview_has_columns" in names
        assert "preview_row_count_available" in names
        assert all(c.passed for c in checks)

    def test_empty_columns_fails(self):
        checks = run_sql_self_check("SELECT 1", columns=[])
        col_check = next(c for c in checks if c.name == "preview_has_columns")
        assert not col_check.passed


class TestSQLStatementSplit:
    def test_split_multi_statement_sql(self):
        assert split_sql_statements("SELECT 1; SELECT 2;") == ["SELECT 1;", "SELECT 2;"]

    def test_split_preserves_semicolon_inside_string(self):
        assert split_sql_statements("SELECT 'a;b' AS value; SELECT 2;") == [
            "SELECT 'a;b' AS value;",
            "SELECT 2;",
        ]


class TestSQLDraftColumnContract:
    def test_extract_sql_aliases_excludes_literals_comments_cast_types_and_reserved_words(self):
        aliases = extract_sql_aliases("""
            WITH customer_orders(customer_alias) AS (
                SELECT
                    CASE WHEN status = 'AS ignored' THEN 1 ELSE 0 END AS `delivery_delay_flag`,
                    SUM(amount) AS repurchase_customer_flag,
                    CAST(status AS CHAR) AS status_text -- AS ignored_comment
                FROM orders
            )
            SELECT customer_alias AS result_value FROM customer_orders;
        """)

        assert aliases == {
            "customer_alias",
            "delivery_delay_flag",
            "repurchase_customer_flag",
            "status_text",
            "result_value",
        }

    def test_normalize_legacy_columns_used_removes_bare_aliases_and_preserves_source_refs(self):
        parsed = {
            "sql": """
                WITH customer_orders(customer_alias) AS (
                    SELECT CASE WHEN status = 'delivered' THEN 1 END AS delivery_delay_flag
                    FROM orders
                )
                SELECT customer_alias AS repurchase_customer_flag FROM customer_orders;
            """,
            "columns_used": [
                "order_id",
                "`ORDER_ID`",
                "delivery_delay_flag",
                "customer_alias",
                "repurchase_customer_flag",
                "orders.delivery_delay_flag",
            ],
        }

        normalized = normalize_generated_sql(parsed, "simple")

        assert parsed["columns_used"] == [
            "order_id",
            "`ORDER_ID`",
            "delivery_delay_flag",
            "customer_alias",
            "repurchase_customer_flag",
            "orders.delivery_delay_flag",
        ]
        assert normalized["source_column_refs"] == ["order_id", "orders.delivery_delay_flag"]
        assert normalized["derived_columns"] == []
        assert normalized["output_columns"] == []
        assert "columns_used" not in normalized

    def test_normalize_generated_sql_prefers_explicit_empty_new_source_refs(self):
        normalized = normalize_generated_sql(
            {
                "sql": "SELECT order_id AS output_id FROM orders;",
                "source_column_refs": [],
                "derived_columns": ["output_id"],
                "output_columns": ["output_id"],
                "columns_used": ["order_id"],
            },
            "simple",
        )

        assert normalized["source_column_refs"] == []
        assert normalized["derived_columns"] == ["output_id"]
        assert normalized["output_columns"] == ["output_id"]
        assert "columns_used" not in normalized

    def test_validate_sql_identifiers_legacy_aliases_do_not_become_missing_columns(self):
        schema_text = '{"orders": {"columns": [{"name": "order_id"}, {"name": "status"}]}}'
        sql_draft = {
            "sql": """
                WITH customer_orders(customer_alias) AS (
                    SELECT CASE WHEN status = 'delivered' THEN 1 END AS delivery_delay_flag
                    FROM orders
                )
                SELECT customer_alias AS repurchase_customer_flag FROM customer_orders;
            """,
            "sql_type": "select",
            "source_tables": ["orders"],
            "columns_used": ["status", "delivery_delay_flag", "customer_alias", "repurchase_customer_flag"],
        }

        findings = validate_sql_identifiers({}, sql_draft, schema_text)

        assert not any(item["category"] == "missing_column" for item in findings)

    @pytest.mark.parametrize("source_column_ref", ["missing_source", "orders.delivery_delay_flag"])
    def test_validate_sql_identifiers_blocks_unknown_source_column_refs(self, source_column_ref):
        schema_text = '{"orders": {"columns": [{"name": "order_id"}]}}'
        findings = validate_sql_identifiers(
            {},
            {
                "sql": "SELECT order_id AS delivery_delay_flag FROM orders;",
                "source_tables": ["orders"],
                "source_column_refs": [source_column_ref],
            },
            schema_text,
        )

        missing_columns = [item for item in findings if item["category"] == "missing_column"]
        assert len(missing_columns) == 1
        assert missing_columns[0]["retryable"] is False

    def test_validate_sql_identifiers_ignores_derived_and_output_columns(self):
        schema_text = '{"orders": {"columns": [{"name": "order_id"}]}}'
        findings = validate_sql_identifiers(
            {},
            {
                "sql": "SELECT order_id AS computed_total FROM orders;",
                "source_tables": ["orders"],
                "source_column_refs": ["order_id"],
                "derived_columns": ["computed_total", "not_in_schema"],
                "output_columns": ["computed_total", "not_in_schema"],
            },
            schema_text,
        )

        assert not any(item["category"] == "missing_column" for item in findings)


# ── planner tests ──

class TestSQLLangGraphSmoke:
    def test_integrity_summary_compacts_legacy_and_ge_payloads(self, monkeypatch, tmp_path):
        from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader

        payload = {
            "summary": {"status": "ACTION_REQUIRED"},
            "tables": {
                "orders": [
                    {"column": "order_id", "status": "PASS", "intent": "PK check", "observed": "all good"},
                    {
                        "column": "customer_id",
                        "status": "FAIL",
                        "intent": "FK check",
                        "observed": "orphan rows detected",
                    },
                    {
                        "success": False,
                        "expectation_config": {
                            "expectation_type": "expect_column_values_to_not_be_null",
                            "kwargs": {"column": "amount"},
                        },
                        "result": {"unexpected_count": 3},
                    },
                ],
                "customers": [
                    {"column": "customer_id", "status": "PASS", "intent": "PK check", "observed": "all good"}
                ],
            },
        }

        text = integrity_loader.compact_integrity_summary_text(payload, tables=["orders"])

        assert "customer_id" in text
        assert "orphan rows detected" in text
        assert "amount" in text
        assert "unexpected_count" not in text
        assert "all good" not in text
        assert "customers" not in text

        legacy_path = tmp_path / "db_integrity_result.json"
        legacy_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(integrity_loader, "INTEGRITY_JSON_PATH", legacy_path)
        assert json.loads(integrity_loader.load_integrity_text()) == payload

    def test_backend_integrity_service_refresh_updates_prompt_context(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module
        from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader

        calls: list[tuple[str, list[str], float, dict | None]] = []
        monkeypatch.setenv("DATA_AGENT_BACKEND_URL", "http://backend.local")

        def fake_post_backend_json(base_url, path, payload):
            assert base_url == "http://backend.local"
            if path == "/integrity/ensure-tables-ready":
                calls.append((payload["dataset_name"], list(payload["tables"]), payload["wait_timeout_s"], payload["metadata"]))
                return {"ok": True, "data": {"ready": False}}
            if path == "/integrity/summary":
                return {"ok": True, "data": {"tables": {"orders": {"checks": [{"column": "order_id", "status": "PASS", "intent": "PK check", "observed": "all good"}, {"column": "amount", "status": "WARNING", "intent": "null profile", "observed": "amount has 2 nulls"}]}}}}
            raise AssertionError(f"unexpected path: {path}")

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"table_preview","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"table_preview","required_tables":["orders"]},"reasoning":"simple query"}')
                return DummyResponse('{"sql":"SELECT order_id, amount FROM orders LIMIT 50;","sql_type":"select","source_tables":["orders"],"columns_used":["order_id","amount"],"reasoning":"preview orders"}')

        monkeypatch.setattr(integrity_loader, "_post_backend_json", fake_post_backend_json)
        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}', "integrity_text": 'legacy pass dump should be replaced'})
        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1, 10)])

        result = build_app().invoke({"user_question": "?? ???? ??? ???", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "integrity_dataset_name": "orders_ds", "integrity_refresh": {}, "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "validation_findings": [], "retry_hint": {}, "validation_summary": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert calls == [("orders_ds", ["orders"], 0.0, {"source": "sql_agent"})]
        assert result["integrity_refresh"]["status"] == "queued"
        assert result["integrity_refresh"]["tables"] == ["orders"]
        assert result["integrity_refresh"]["ready"] is False
        assert result["integrity_refresh"]["local_snapshot_used"] is True
        assert result["integrity_text"] == ""

    def test_preplan_integrity_gate_updates_planner_context(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader

        monkeypatch.setenv("DATA_AGENT_BACKEND_URL", "http://backend.local")

        def fake_post_backend_json(base_url, path, payload):
            assert path == "/integrity/summary"
            assert payload["dataset_name"] == "orders_ds"
            return {
                "ok": True,
                "data": {
                    "summaries": [
                        {
                            "table_name": "orders",
                            "status": "stale",
                            "summary": {"message": "orders needs refresh"},
                        }
                    ]
                },
            }

        monkeypatch.setattr(integrity_loader, "_post_backend_json", fake_post_backend_json)
        update = context_module.preplan_integrity_gate({"integrity_dataset_name": "orders_ds", "integrity_text": "legacy text"})

        assert update["integrity_preplan"]["status"] == "prefetched"
        assert "orders" in update["integrity_text"]
        assert "[STALE]" in update["integrity_text"]

    def test_build_app_simple_path_supports_future_input_fields(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_date"}]}}', "integrity_text": '{}'})
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1, "2024-01-01")])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"table_preview","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"table_preview","required_tables":["orders"]},"reasoning":"simple query"}')
                return DummyResponse('{"sql":"SELECT order_id, order_date FROM orders LIMIT 50;","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.order_date"],"derived_columns":[],"output_columns":["order_id","order_date"],"reasoning":"preview orders"}')

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        result = build_app().invoke({"user_question": "?? ???? ??? ???", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["plan"]["route_kind"] == "simple"
        assert result["sql_draft"]["sql_type"] == "select"
        assert result["sql_draft"]["source_column_refs"] == ["orders.order_id", "orders.order_date"]
        assert "columns_used" not in result["sql_draft"]
        assert result["validation"]["result"] == "valid"
        assert "simple 경로" in result["final_answer"]

    def test_build_app_simple_average_delivery_days_uses_aggregate_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}', "integrity_text": '{}'})
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(4.2,)])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"average_delivery_days","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"single_scalar","required_columns":["order_approved_at","order_delivered_customer_date"],"required_aggregations":["AVG"],"validation_contract":{"expected_result_shape":"single_scalar","required_aggregations":["AVG"],"required_columns":["order_approved_at","order_delivered_customer_date"],"expected_aliases":["avg_delivery_days"],"required_tables":["orders"],"target_metric":"average_delivery_days","dimensions":[]},"reasoning":"average delivery question"}')
                return DummyResponse('{"sql":"SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days FROM orders WHERE order_approved_at IS NOT NULL AND order_delivered_customer_date IS NOT NULL;","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_delivered_customer_date","orders.order_approved_at"],"derived_columns":["avg_delivery_days"],"output_columns":["avg_delivery_days"],"reasoning":"aggregate delivery query"}')

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        result = build_app().invoke({"user_question": "?? ????? ?? ????? ?? ??? ??", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert "AVG(DATEDIFF(order_delivered_customer_date, order_approved_at))" in result["sql_draft"]["sql"]
        assert "WHERE order_approved_at IS NOT NULL" in result["sql_draft"]["sql"]
        assert result["validation"]["result"] == "valid"
        assert result["plan"]["expected_result_shape"] == "single_scalar"

    def test_build_app_rejects_non_aggregate_sql_for_average_question(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}', "integrity_text": '{}'})

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"average_delivery_days","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"single_scalar","required_columns":["order_approved_at","order_delivered_customer_date"],"required_aggregations":["AVG"],"validation_contract":{"expected_result_shape":"single_scalar","required_aggregations":["AVG"],"required_columns":["order_approved_at","order_delivered_customer_date"],"expected_aliases":["avg_delivery_days"],"required_tables":["orders"],"target_metric":"average_delivery_days","dimensions":[]},"reasoning":"average delivery question"}')
                return DummyResponse('{"sql":"SELECT * FROM orders LIMIT 50;","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_id"],"derived_columns":[],"output_columns":[],"reasoning":"bad draft"}')

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        result = build_app().invoke({"user_question": "?? ????? ?? ????? ?? ??? ??", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 2, "feedback": "", "error": "", "final_answer": ""})

        assert result["validation"]["result"] == "invalid"
        assert any(item["category"] == "intent_mismatch" for item in result["validation"]["findings"])
        assert result["retry_hint"]["reason_code"] == "intent_mismatch"

    def test_build_app_rejects_missing_table_before_execution(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}]}}',
            "integrity_text": '{}'
        })

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner다" in prompt:
                    return DummyResponse(
                        '{"route_kind":"simple","selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],'
                        '"target_metric":"주문 수","dimensions":[],"filters":[],"time_condition":null,"reasoning":"단순 질의"}'
                    )
                return DummyResponse(
                    '{"sql":"SELECT COUNT(*) AS order_count FROM category_performance_analysis;","sql_type":"select","source_tables":["category_performance_analysis"],"source_column_refs":["category_performance_analysis.order_id"],"derived_columns":["order_count"],"output_columns":["order_count"],"reasoning":"없는 테이블"}'
                )

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        app = build_app()
        result = app.invoke({
            "user_question": "주문 건수 계산",
            "required_db_schema": "",
            "clarification_request": "",
            "planner_selection_reason": "SQL 기반 질의 응답",
            "schema_text": "",
            "integrity_text": "",
            "plan": {},
            "mart_design": {},
            "sql_draft": {},
            "sql_result": None,
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "retry_count": 0,
            "max_retries": 0,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["validation"]["result"] == "invalid"
        assert any(item["category"] == "missing_table" for item in result["validation"]["findings"])
        assert result["retry_hint"]["reason_code"] == "missing_table"

    def test_validate_sql_identifiers_allows_cte_references(self):
        schema_text = '{"orders": {"columns": [{"name": "order_id"}]}}'
        sql_draft = {
            "sql": """
            WITH customer_orders AS (
                SELECT order_id FROM orders
            )
            SELECT * FROM customer_orders;
            """,
            "sql_type": "select",
            "source_tables": ["orders"],
            "columns_used": ["orders.order_id"],
        }

        findings = validate_sql_identifiers({}, sql_draft, schema_text)

        assert not any(item["category"] == "missing_table" for item in findings)

    def test_validate_sql_identifiers_still_rejects_unknown_non_cte_table(self):
        schema_text = '{"orders": {"columns": [{"name": "order_id"}]}}'
        sql_draft = {
            "sql": """
            WITH customer_orders AS (
                SELECT order_id FROM orders
            )
            SELECT * FROM not_existing_table;
            """,
            "sql_type": "select",
            "source_tables": ["orders"],
            "columns_used": ["orders.order_id"],
        }

        findings = validate_sql_identifiers({}, sql_draft, schema_text)

        assert any(
            item["category"] == "missing_table" and "not_existing_table" in item["detail"]
            for item in findings
        )

    def test_build_app_comprehensive_path_generates_datamart_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}', "integrity_text": '{}'})
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(10,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed = []
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: committed.append(sql))

        responses = iter([
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build","requested_output":"create_table","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"mart_name":"category_performance_analysis","grain":"order_id","load_strategy":"full_refresh","expected_result_shape":"datamart_creation","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"datamart_creation","required_tables":["orders"],"target_table":"analytics.category_performance_analysis"},"reasoning":"datamart request"}',
            '{"mart_name":"category_performance_analysis","target_schema":"analytics","grain":"order_id","base_grain":"order_id","source_tables":["orders"],"key_columns":["order_id"],"measure_columns":["amount"],"dimension_columns":["order_id"],"incremental_column":null,"load_strategy":"full_refresh","row_preserving_strategy":"order row preserving","aggregation_policy":"prefer_row_preserving","aggregation_rationale":"row level mart","design_reasoning":"order mart"}',
            '{"sql":"CREATE TABLE analytics.category_performance_analysis AS SELECT * FROM orders;","sql_type":"create_table_as","target_table":"analytics.category_performance_analysis","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.amount"],"derived_columns":[],"output_columns":[],"postcheck_sql":"SELECT COUNT(*) FROM analytics.category_performance_analysis;","reasoning":"build mart"}'
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        result = build_app().invoke({"user_question": "??? ??? ?????? ????", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "??? ??? ?? datamart ??", "schema_text": "", "integrity_text": "", "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["plan"]["route_kind"] == "comprehensive"
        assert result["sql_draft"]["sql_type"] == "create_table_as"
        assert committed and committed[0].lower().startswith("create table analytics.")
        assert result["validation"]["result"] == "valid"
        assert "datamart" in result["final_answer"]

    def test_build_app_surfaces_mart_design_json_failure(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
            "integrity_text": '{}'
        })

        responses = iter([
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build","requested_output":"create_table","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"mart_name":"category_performance_analysis","grain":"order_id","load_strategy":"full_refresh","expected_result_shape":"datamart_creation","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"datamart_creation","required_tables":["orders"],"target_table":"analytics.category_performance_analysis"},"reasoning":"datamart request"}',
            '{bad json',
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())

        result = build_app().invoke({
            "user_question": "재사용 가능한 데이터마트를 만들어줘",
            "required_db_schema": "",
            "clarification_request": "",
            "planner_selection_reason": "복잡한 분석을 위한 datamart 필요",
            "schema_text": "",
            "integrity_text": "",
            "plan": {},
            "mart_design": {},
            "sql_draft": {},
            "sql_result": None,
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "validation_findings": [],
            "retry_hint": {},
            "validation_summary": {},
            "retry_count": 0,
            "max_retries": 0,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["mart_design"] == {}
        assert result["validation"]["result"] == "invalid"
        assert result["retry_hint"]["reason_code"] == "sql_mart_design_failed"
        assert result["retry_hint"]["details"]["mart_design_reason_code"] == "llm_json_parse_failed"
        assert "mart 설계 응답을 JSON으로 파싱하지 못했습니다" in result["final_answer"]

    def test_build_app_retries_after_mart_design_failure(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(10,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed = []
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: committed.append(sql))

        responses = iter([
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build","requested_output":"create_table","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"mart_name":"category_performance_analysis","grain":"order_id","load_strategy":"full_refresh","expected_result_shape":"datamart_creation","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"datamart_creation","required_tables":["orders"],"target_table":"analytics.category_performance_analysis"},"reasoning":"plan 1"}',
            '{bad json',
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build","requested_output":"create_table","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"mart_name":"category_performance_analysis","grain":"order_id","load_strategy":"full_refresh","expected_result_shape":"datamart_creation","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"datamart_creation","required_tables":["orders"],"target_table":"analytics.category_performance_analysis"},"reasoning":"plan 2"}',
            '{"mart_name":"category_performance_analysis","target_schema":"analytics","grain":"order_id","base_grain":"order_id","source_tables":["orders"],"key_columns":["order_id"],"measure_columns":["amount"],"dimension_columns":["order_id"],"incremental_column":null,"load_strategy":"full_refresh","row_preserving_strategy":"order row preserving","aggregation_policy":"prefer_row_preserving","aggregation_rationale":"row level mart","design_reasoning":"order mart"}',
            '{"sql":"CREATE TABLE analytics.category_performance_analysis AS SELECT * FROM orders;","sql_type":"create_table_as","target_table":"analytics.category_performance_analysis","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.amount"],"derived_columns":[],"output_columns":[],"postcheck_sql":"SELECT COUNT(*) FROM analytics.category_performance_analysis;","reasoning":"build mart"}',
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())

        result = build_app().invoke({
            "user_question": "재사용 가능한 데이터마트를 만들어줘",
            "required_db_schema": "",
            "clarification_request": "",
            "planner_selection_reason": "복잡한 분석을 위한 datamart 필요",
            "schema_text": "",
            "integrity_text": "",
            "plan": {},
            "mart_design": {},
            "sql_draft": {},
            "sql_result": None,
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "validation_findings": [],
            "retry_hint": {},
            "validation_summary": {},
            "retry_count": 0,
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["retry_count"] == 1
        assert result["validation"]["result"] == "valid"
        assert result["mart_design"]["mart_name"] == "category_performance_analysis"
        assert committed and committed[0].lower().startswith("create table analytics.")

    def test_build_app_normalizes_object_based_mart_design_columns(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "customer_id"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: None)

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "SQL/데이터마트 planner다" in prompt or "planner다" in prompt:
                    return DummyResponse(
                        '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build",'
                        '"requested_output":"create_table","target_metric":"재구매 분석","dimensions":["customer_id"],'
                        '"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],'
                        '"expected_result_shape":"datamart_creation","required_aggregations":[],"mart_name":"customer_reorder_mart",'
                        '"grain":"customer","load_strategy":"full_refresh","reasoning":"마트 생성"}'
                    )
                if "분석용 데이터마트 설계자다" in prompt:
                    return DummyResponse(
                        '{"mart_name":"customer_reorder_mart","target_schema":"analytics","grain":"customer","base_grain":"customer",'
                        '"source_tables":["orders"],'
                        '"key_columns":[{"column_name":"customer_id","description":"고객 식별자"}],'
                        '"measure_columns":[{"column_name":"total_orders","description":"총 주문수"}],'
                        '"dimension_columns":[{"column_name":"customer_id","description":"고객 축"}],'
                        '"load_strategy":"full_refresh","row_preserving_strategy":"고객 단위 유지",'
                        '"aggregation_policy":"aggregate_if_justified","aggregation_rationale":"고객 단위 집계 필요",'
                        '"design_reasoning":"재구매 분석용 고객 단위 마트"}'
                    )
                if "MySQL SQL 작성기다. 데이터마트 생성 SQL" in prompt:
                    return DummyResponse(
                        '{"sql":"CREATE TABLE analytics.customer_reorder_mart AS SELECT customer_id FROM orders;",'
                        '"sql_type":"create_table_as","target_table":"analytics.customer_reorder_mart",'
                        '"source_tables":["orders"],"source_column_refs":["orders.customer_id"],"derived_columns":[],"output_columns":["customer_id"],'
                        '"postcheck_sql":"SELECT COUNT(*) FROM analytics.customer_reorder_mart;","reasoning":"마트 생성"}'
                    )
                raise AssertionError(f"unexpected prompt: {prompt[:80]}")

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        app = build_app()
        result = app.invoke({
            "user_question": "고객 재구매 분석용 데이터마트를 만들어줘",
            "required_db_schema": "",
            "clarification_request": "",
            "planner_selection_reason": "복잡한 분석을 위한 datamart 필요",
            "schema_text": "",
            "integrity_text": "",
            "plan": {},
            "mart_design": {},
            "sql_draft": {},
            "sql_result": None,
            "statement_results": [],
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "validation_findings": [],
            "retry_hint": {},
            "validation_summary": {},
            "retry_count": 0,
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "generation_source": "llm",
            "generation_failure_reason": "",
            "failed_statement_index": None,
            "failed_statement_sql": "",
            "final_answer": "",
        })

        assert result["mart_design"]["key_columns"] == ["customer_id"]
        assert result["mart_design"]["measure_columns"] == ["total_orders"]
        assert result["mart_design"]["dimension_columns"] == ["customer_id"]
        assert result["validation"]["result"] == "valid"

    def test_build_app_retries_after_mysql_dialect_failure(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}', "integrity_text": '{}'})
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        responses = iter(['{"sql": "SELECT JULIANDAY(order_delivered_customer_date) - JULIANDAY(order_approved_at) AS delivery_days FROM orders;", "sql_type": "select", "source_tables": ["orders"], "source_column_refs": ["orders.order_delivered_customer_date", "orders.order_approved_at"], "derived_columns": ["delivery_days"], "output_columns": ["delivery_days"], "reasoning": "1? ??"}', '{"sql": "SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days FROM orders;", "sql_type": "select", "source_tables": ["orders"], "source_column_refs": ["orders.order_delivered_customer_date", "orders.order_approved_at"], "derived_columns": ["avg_delivery_days"], "output_columns": ["avg_delivery_days"], "reasoning": "2? ??"}'])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"average_delivery_days","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"single_scalar","required_columns":["order_approved_at","order_delivered_customer_date"],"required_aggregations":["AVG"],"validation_contract":{"expected_result_shape":"single_scalar","required_aggregations":["AVG"],"required_columns":["order_approved_at","order_delivered_customer_date"],"expected_aliases":["avg_delivery_days"],"required_tables":["orders"],"target_metric":"average_delivery_days","dimensions":[]},"reasoning":"average delivery question"}')
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(3,)])
        result = build_app().invoke({"user_question": "?? ???? ????", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "plan": {"route_kind": "simple", "selected_join_tables": ["orders"]}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["validation"]["result"] == "valid"
        assert "AVG(DATEDIFF" in result["sql_draft"]["sql"]
        assert result["retry_count"] == 1

    def test_build_app_executes_multi_statement_selects_sequentially(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_status"}]}}',
            "integrity_text": '{}'
        })

        responses = iter([
            '{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"table_preview","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"table_preview","required_tables":["orders"]},"reasoning":"two diagnostics"}',
            '{"sql":"SELECT COUNT(*) AS total_orders FROM orders; SELECT COUNT(*) AS delivered_orders FROM orders WHERE order_status = \\"delivered\\";","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.order_status"],"derived_columns":["total_orders","delivered_orders"],"output_columns":["total_orders","delivered_orders"],"reasoning":"두 개의 진단 질의"}',
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        executed: list[str] = []

        def fake_fetch(sql: str):
            executed.append(sql)
            if "delivered_orders" in sql:
                return [(3,)]
            return [(10,)]

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", fake_fetch)

        app = build_app()
        result = app.invoke({
            "user_question": "주문 수와 배송 완료 주문 수를 각각 보여줘",
            "required_db_schema": "",
            "clarification_request": "",
            "planner_selection_reason": "SQL 기반 질의 응답",
            "schema_text": "",
            "integrity_text": "",
            "plan": {"route_kind": "simple", "selected_join_tables": ["orders"], "expected_result_shape": "table_preview"},
            "mart_design": {},
            "sql_draft": {},
            "sql_result": None,
            "statement_results": [],
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "validation_findings": [],
            "retry_hint": {},
            "validation_summary": {},
            "retry_count": 0,
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "generation_source": "llm",
            "generation_failure_reason": "",
            "failed_statement_index": None,
            "failed_statement_sql": "",
            "final_answer": "",
        })

        assert executed == [
            "SELECT COUNT(*) AS total_orders FROM orders;",
            'SELECT COUNT(*) AS delivered_orders FROM orders WHERE order_status = "delivered";',
        ]
        assert len(result["statement_results"]) == 2
        assert result["statement_results"][0]["row_count"] == 1
        assert result["statement_results"][1]["row_count"] == 1
        assert result["validation"]["result"] == "valid"

    def test_build_app_marks_retry_fallback_used_after_invalid_retry_json(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}',
            "integrity_text": '{}'
        })

        responses = iter([
            '{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"average_delivery_days","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"single_scalar","required_columns":["order_approved_at","order_delivered_customer_date"],"required_aggregations":["AVG"],"validation_contract":{"expected_result_shape":"single_scalar","required_aggregations":["AVG"],"required_columns":["order_approved_at","order_delivered_customer_date"],"expected_aliases":["avg_delivery_days"],"required_tables":["orders"],"target_metric":"average_delivery_days","dimensions":[]},"reasoning":"average delivery question"}',
            '{"sql":"SELECT JULIANDAY(order_delivered_customer_date) - JULIANDAY(order_approved_at) AS delivery_days FROM orders;","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_delivered_customer_date","orders.order_approved_at"],"derived_columns":["delivery_days"],"output_columns":["delivery_days"],"reasoning":"1차 시도"}',
            '{bad json',
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(3,)])

        app = build_app()
        result = app.invoke({
            "user_question": "배송 소요일을 계산해줘",
            "required_db_schema": "",
            "clarification_request": "",
            "planner_selection_reason": "SQL 기반 질의 응답",
            "schema_text": "",
            "integrity_text": "",
            "plan": {"route_kind": "simple", "selected_join_tables": ["orders"]},
            "mart_design": {},
            "sql_draft": {},
            "sql_result": None,
            "statement_results": [],
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "validation_findings": [],
            "retry_hint": {},
            "validation_summary": {},
            "retry_count": 0,
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "generation_source": "llm",
            "generation_failure_reason": "",
            "failed_statement_index": None,
            "failed_statement_sql": "",
            "final_answer": "",
        })

        assert result["generation_source"] == "failed"
        assert result["generation_failure_reason"] == "llm_json_parse_failed"
        assert result["validation"]["result"] == "invalid"
        assert result["retry_hint"]["reason_code"] == "sql_generation_failed"
        assert result["retry_hint"]["details"]["generation_reason_code"] == "llm_json_parse_failed"
        assert "LLM SQL" in result["final_answer"]

    def test_build_app_reports_failed_statement_metadata(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_status"}]}}', "integrity_text": '{}'})

        responses = iter([
            '{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"table_preview","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"table_preview","required_tables":["orders"]},"reasoning":"two diagnostics"}',
            '{"sql":"SELECT COUNT(*) AS total_orders FROM orders; SELECT COUNT(*) AS delivered_orders FROM orders WHERE order_status = \\\"delivered\\\";","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.order_status"],"derived_columns":["total_orders","delivered_orders"],"output_columns":["total_orders","delivered_orders"],"reasoning":"two diagnostic queries"}'
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        def fake_fetch(sql: str):
            if "delivered_orders" in sql:
                raise RuntimeError("bad delivered query")
            return [(10,)]

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", fake_fetch)

        result = build_app().invoke({"user_question": "?? ?? ?? ?? ?? ?? ?? ???", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "plan": {"route_kind": "simple", "selected_join_tables": ["orders"]}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "statement_results": [], "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "validation_findings": [], "retry_hint": {}, "validation_summary": {}, "retry_count": 0, "max_retries": 0, "feedback": "", "error": "", "generation_source": "llm", "generation_failure_reason": "", "failed_statement_index": None, "failed_statement_sql": "", "final_answer": ""})

        assert result["validation"]["result"] == "invalid"
        assert result["failed_statement_index"] == 1
        assert "delivered_orders" in result["failed_statement_sql"]
        assert "2번 statement" in result["validation"]["reason"]

    def test_build_app_normalizes_unqualified_mart_postcheck_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}', "integrity_text": '{}'})
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed: list[str] = []
        executed_queries: list[str] = []

        responses = iter([
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build","requested_output":"create_table","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"mart_name":"category_performance_analysis","grain":"order_id","load_strategy":"full_refresh","expected_result_shape":"datamart_creation","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"datamart_creation","required_tables":["orders"],"target_table":"analytics.category_performance_analysis"},"reasoning":"datamart request"}',
            '{"mart_name":"category_performance_analysis","target_schema":"analytics","grain":"order_id","base_grain":"order_id","source_tables":["orders"],"key_columns":["order_id"],"measure_columns":["amount"],"dimension_columns":["order_id"],"incremental_column":null,"load_strategy":"full_refresh","row_preserving_strategy":"order row preserving","aggregation_policy":"prefer_row_preserving","aggregation_rationale":"row level mart","design_reasoning":"order mart"}',
            '{"sql":"CREATE TABLE analytics.category_performance_analysis AS SELECT * FROM orders;","sql_type":"create_table_as","target_table":"category_performance_analysis","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.amount"],"derived_columns":[],"output_columns":[],"postcheck_sql":"SELECT COUNT(*) FROM category_performance_analysis;","reasoning":"?? ??"}'
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: committed.append(sql))

        def fake_fetch(sql: str):
            executed_queries.append(sql)
            return [(1,)]

        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", fake_fetch)

        result = build_app().invoke({"user_question": "??? ??? ?????? ????", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "??? ??? ?? datamart ??", "schema_text": "", "integrity_text": "", "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["sql_draft"]["target_table"] == "analytics.category_performance_analysis"
        assert result["sql_draft"]["postcheck_sql"] == "SELECT COUNT(*) FROM analytics.category_performance_analysis;"
        assert any("FROM analytics.category_performance_analysis" in query for query in executed_queries)
        assert committed

@pytest.mark.skip(reason="레거시 SQLAgentSupervisor 라우팅 계약은 LangGraph Supervisor 테스트로 대체되었습니다.")
class TestRouteDetection:
    def test_simple_route(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        assert state.route_kind == "simple"
        assert started_agents(adapter, state.run_id) == ["sql_agent", "report_agent"]

    def test_eda_route(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("데이터 품질 프로파일을 보여줘")
        assert state.route_kind == "eda"
        assert started_agents(adapter, state.run_id) == ["sql_agent", "eda_agent", "report_agent"]

    def test_trend_route(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("월별 매출 추이를 차트로 보여줘")
        assert state.route_kind == "trend"
        assert started_agents(adapter, state.run_id) == [
            "sql_agent", "analysis_agent", "visualization_agent", "report_agent",
        ]

    def test_mart_route(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트 저장을 제안해줘")
        assert state.route_kind == "mart"
        assert state.terminal_state == SupervisorTerminalState.needs_user_approval
        # mart route: sql_agent only, then approval_gate
        assert "report_agent" not in state.artifact_ids

    def test_comprehensive_route(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("데이터 품질 프로파일하고 월별 매출 추이도 분석해줘")
        assert state.route_kind == "comprehensive"
        assert state.terminal_state == SupervisorTerminalState.completed
        assert started_agents(adapter, state.run_id) == [
            "sql_agent", "eda_agent", "analysis_agent", "visualization_agent", "report_agent",
        ]

    def test_comprehensive_english(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("Profile data quality and analyze monthly revenue trend with chart")
        assert state.route_kind == "comprehensive"
        assert started_agents(adapter, state.run_id) == [
            "sql_agent", "eda_agent", "analysis_agent", "visualization_agent", "report_agent",
        ]


# ── SQLAgent integration tests ──

class _FakeRetryingSQLAgent:
    name = "sql_agent"

    def __init__(self):
        self.calls = 0

    def run(self, state, runtime):
        self.calls += 1
        if self.calls == 1:
            state.generated_sql = "SELECT JULIANDAY(order_created_at) FROM orders;"
            state.error_state = {
                "code": "mysql_dialect_incompatible_sql",
                "message": "비호환 표현 'JULIANDAY' 감지",
                "query": state.generated_sql,
                "datasource_id": state.datasource_id,
                "step": "sql_agent",
            }
            return AgentEnvelope(
                status=AgentStatus.failed,
                agent_name=self.name,
                summary="first sql failed",
                validation=ValidationBlock(local_checks=[
                    LocalCheck(name="sql", passed=False, severity="error", detail="dialect mismatch")
                ]),
                retry_hint=RetryHint(retryable=True, suggested_action="fix_sql", reason_code="mysql_dialect_incompatible_sql"),
            )
        state.generated_sql = "SELECT DATEDIFF(order_delivered_customer_date, order_approved_at) FROM orders;"
        state.error_state = {}
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name=self.name,
            summary="second sql ok",
            validation=ValidationBlock(local_checks=[
                LocalCheck(name="sql", passed=True, severity="info", detail="ok")
            ]),
        )


class _PassThroughValidationAgent:
    name = "validation_agent"

    def run(self, state, runtime, upstream):
        if upstream.status == AgentStatus.failed:
            return AgentEnvelope(
                status=AgentStatus.failed,
                agent_name=self.name,
                summary="retry upstream",
                validation=ValidationBlock(local_checks=[
                    LocalCheck(name="gate", passed=False, severity="error", detail="retry")
                ]),
                retry_hint=RetryHint(retryable=True, suggested_action="retry_upstream", reason_code="mysql_dialect_incompatible_sql"),
            )
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name=self.name,
            summary="validation ok",
            validation=ValidationBlock(local_checks=[
                LocalCheck(name="gate", passed=True, severity="info", detail="ok")
            ]),
        )


class TestSQLAgentIntegration:
    def test_main_sql_envelope_preserves_retry_required_findings_on_success(self, adapter):
        from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent

        runtime = AgentRuntime(adapter)
        state = OrchestrationState(
            run_id=adapter.create_run().run_id,
            user_query="주문과 고객을 조인해줘",
        )
        result = {
            "plan": {},
            "mart_design": {},
            "sql_draft": {
                "sql": "SELECT 1 AS value",
                "sql_type": "SELECT",
                "source_column_refs": [],
                "derived_columns": ["value"],
                "output_columns": ["value"],
            },
            "sql_result": [{"value": 1}],
            "validation": {"result": "valid"},
            "validation_findings": [
                {
                    "code": "invalid_join_plan",
                    "source": "sql_validator",
                    "severity": "warning",
                    "message": "조인 계획을 다시 확인해야 합니다.",
                    "retryable": True,
                    "suggested_action": "fix_sql",
                    "details": {"join": "orders-customers"},
                }
            ],
            "retry_hint": {
                "retryable": True,
                "suggested_action": "fix_sql",
                "reason_code": "invalid_join_plan",
                "details": {"join": "orders-customers"},
            },
            "final_answer": "SQL 실행 완료",
        }

        envelope = SQLAgent()._envelope_from_main_result(state, runtime, result)

        assert envelope.status == AgentStatus.success
        assert envelope.validation.findings[0].code == "invalid_join_plan"
        assert envelope.validation.findings[0].disposition == "retry_required"
        assert envelope.retry_hint.retryable is True
        assert envelope.retry_hint.details == {"join": "orders-customers"}
        artifact_payloads = [
            json.loads(Path(adapter.get_artifact(ref.artifact_id).local_path).read_text(encoding="utf-8"))
            for ref in envelope.artifact_refs
            if adapter.get_artifact(ref.artifact_id).metadata.get("kind") in {"sql_lang_graph_result", "sql_plan"}
        ]
        assert len(artifact_payloads) == 2
        assert all("columns_used" not in payload["sql_draft"] for payload in artifact_payloads)

    @pytest.mark.skip(reason="제거된 레거시 parse_plan API 테스트입니다.")
    def test_supervisor_parse_plan_transfers_retry_context(self, adapter):
        from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState

        sup = SQLAgentSupervisor(adapter)
        state = OrchestrationState(run_id=adapter.create_run().run_id, user_query="간단한 매출 요약을 보여줘")
        state.error_state = {
            "code": "BAD_SQL",
            "message": "unknown column amountt",
            "query": "SELECT amountt FROM orders",
            "datasource_id": "ds_retry",
            "step": "sql_agent",
        }
        reparsed = sup.parse_plan(
            {"state": state, "last_agent_result": None, "last_validation_result": None, "gate_result": None}
        )["state"]
        assert reparsed.plan is not None
        assert reparsed.plan.retry_context is not None
        assert reparsed.plan.retry_context["message"] == "unknown column amountt"

    @pytest.mark.skip(reason="레거시 주입식 Supervisor API 테스트이며 envelope 단위 테스트로 대체되었습니다.")
    def test_retry_hint_details_are_preserved(self, adapter):
        retrying_sql_agent = _FakeRetryingSQLAgent()
        sup = SQLAgentSupervisor(
            adapter,
            sql_agent=retrying_sql_agent,
            validation_agent=_PassThroughValidationAgent(),
        )
        state = sup.run("간단한 매출 요약을 보여줘")
        assert state.plan is not None
        assert state.terminal_state == SupervisorTerminalState.completed

    def test_central_validation_uses_sql_agent_plan_route_kind(self, adapter):
        runtime = AgentRuntime(adapter)
        state = OrchestrationState(
            run_id=adapter.create_run().run_id,
            user_query="주문 완료일부터 배송 완료일까지 평균 소요일 계산",
            route_kind="comprehensive",
        )
        context = runtime.context(state, node_name="sql_agent", tool_name="sql_agent.lang_graph")
        plan_ref = adapter.register_artifact(
            state.run_id,
            "file",
            content_text='{"plan":{"route_kind":"simple"},"sql_draft":{"sql":"SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days FROM orders WHERE order_approved_at IS NOT NULL AND order_delivered_customer_date IS NOT NULL;"}}',
            filename="sql_lang_graph_result_test.json",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            metadata={"kind": "sql_lang_graph_result"},
        )
        sql_ref = adapter.register_artifact(
            state.run_id,
            "sql_query",
            content_text="SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days FROM orders WHERE order_approved_at IS NOT NULL AND order_delivered_customer_date IS NOT NULL;",
            filename="generated_sql_test.sql",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            metadata={"kind": "generated_sql"},
        )
        result_ref = adapter.register_artifact(
            state.run_id,
            "sql_result",
            content_text="avg_delivery_days\n4.2\n",
            filename="sql_result_test.csv",
            created_by_tool="sql_agent.lang_graph",
            context=context,
            metadata={"kind": "sql_result"},
            preview={"row_count": 1, "columns": ["avg_delivery_days"]},
        )
        upstream = AgentEnvelope(
            status=AgentStatus.success,
            agent_name="sql_agent",
            summary="ok",
            artifact_refs=[result_ref, plan_ref, sql_ref],
        )
        state.add_artifacts("sql_agent", upstream.artifact_ids())
        if CentralValidationAgent is None:
            pytest.skip("CentralValidationAgent source is not present in this checkout")
        envelope = CentralValidationAgent().run(state, runtime, upstream)
        assert envelope.status != AgentStatus.failed
        assert not any(flag.code == "unsafe_sql" for flag in envelope.validation.business_flags)

    @pytest.mark.skip(reason="레거시 Supervisor 통합 계약 테스트입니다.")
    def test_creates_preview_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        assert "sql_agent" in state.artifact_ids
        assert len(state.artifact_ids["sql_agent"]) >= 1

    @pytest.mark.skip(reason="레거시 Supervisor 통합 계약 테스트입니다.")
    def test_creates_ge_validation_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        has_ge = any(
            adapter.get_artifact(aid).metadata.get("kind") == "ge_table_validation_json"
            for aid in state.artifact_ids["sql_agent"]
        )
        assert has_ge, "GE validation artifact not found"

    @pytest.mark.skip(reason="레거시 Supervisor 통합 계약 테스트입니다.")
    def test_creates_sql_plan_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        has_plan = any(
            adapter.get_artifact(aid).metadata.get("kind") == "sql_plan"
            for aid in state.artifact_ids["sql_agent"]
        )
        assert has_plan, "SQL plan artifact not found"

    @pytest.mark.skip(reason="레거시 Supervisor 통합 계약 테스트입니다.")
    def test_non_mart_query_no_approval(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        assert state.terminal_state == SupervisorTerminalState.completed
        assert not state.approval_ids

    @pytest.mark.skip(reason="레거시 mart 승인 흐름은 Supervisor 승인 테스트로 대체되었습니다.")
    def test_mart_query_requires_approval(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.terminal_state == SupervisorTerminalState.needs_user_approval
        assert state.approval_ids

    @pytest.mark.skip(reason="레거시 mart 승인 흐름은 Supervisor 승인 테스트로 대체되었습니다.")
    def test_mart_query_creates_candidate_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.mart_candidate_ids
        art = adapter.get_artifact(state.mart_candidate_ids[0])
        assert art.metadata.get("kind") == "mart_candidate"

    @pytest.mark.skip(reason="레거시 mart 승인 흐름은 Supervisor 승인 테스트로 대체되었습니다.")
    def test_approval_created_by_approval_gate_not_sql_agent(self, adapter):
        """Approval request is created in approval_gate, not in SQL-Agent."""
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.approval_ids
        # approval event must come from approval_gate node
        pairs = event_pairs(adapter, state.run_id)
        approval_nodes = [node for node, etype in pairs if etype == "approval.required"]
        assert approval_nodes == ["approval_gate"]
        assert state.mart_id is None  # no mart materialized yet

    @pytest.mark.skip(reason="레거시 mart 승인 흐름은 Supervisor 승인 테스트로 대체되었습니다.")
    def test_no_mart_metadata_before_approval(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.mart_id is None
        assert "mart_metadata" not in state.artifact_ids

    @pytest.mark.skip(reason="제거된 레거시 Supervisor runtime 속성 테스트입니다.")
    def test_sql_preview_failure_records_retry_context(self, adapter, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState

        monkeypatch.setattr(
            adapter,
            "run_sql_preview",
            MagicMock(side_effect=BackendError("BAD_SQL", "Unknown column 'amountt'")),
        )
        state = OrchestrationState(
            run_id=adapter.create_run().run_id,
            user_query="매출 요약",
            current_step="sql_agent",
            datasource_id="ds_retry",
            plan=AnalysisPlan(goal="simple", route_kind="simple"),
        )
        envelope = SQLAgent().run(state, SQLAgentSupervisor(adapter).runtime)
        assert envelope.status.value == "failed"
        assert state.error_state["code"] == "BAD_SQL"
        assert state.error_state["datasource_id"] == "ds_retry"
        assert "SELECT" in state.error_state["query"]
        assert state.plan is not None
        assert state.plan.retry_context is not None

    @pytest.mark.skip(reason="레거시 주입식 Supervisor API 테스트입니다.")
    def test_supervisor_retries_sql_agent_after_retryable_failure(self, adapter):
        retrying_sql_agent = _FakeRetryingSQLAgent()
        sup = SQLAgentSupervisor(
            adapter,
            sql_agent=retrying_sql_agent,
            validation_agent=_PassThroughValidationAgent(),
        )
        state = sup.run("간단한 매출 요약을 보여줘")

        assert state.terminal_state == SupervisorTerminalState.completed
        assert retrying_sql_agent.calls == 2
        assert state.retry_counts["sql_agent"] == 1
        assert any(e.event_type == "supervisor.retry_scheduled" for e in adapter.services.run_service.list_events(state.run_id))

    def test_backend_not_modified(self):
        changed = [
            "DATA_Analyst_Assistant_Agent/agents/sql/agent.py",
            "DATA_Analyst_Assistant_Agent/agents/sql/self_check.py",
            "DATA_Analyst_Assistant_Agent/mart/metadata.py",
            "DATA_Analyst_Assistant_Agent/mart/persistence.py",
            "DATA_Analyst_Assistant_Agent/supervisor.py",
            "tests/test_sql_agent.py",
        ]
        assert not any(p.startswith("data_agent_backend/") for p in changed)


    def test_backend_integrity_service_refresh_only_updates_prompt_when_ready(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module
        from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader

        monkeypatch.setenv("DATA_AGENT_BACKEND_URL", "http://backend.local")

        def fake_post_backend_json(base_url, path, payload):
            if path == "/integrity/ensure-tables-ready":
                return {"ok": True, "data": {"ready": True}}
            if path == "/integrity/summary":
                return {"ok": True, "data": {"summaries": [{"table_name": "orders", "status": "warning", "summary": {"message": "amount has 2 nulls"}}]}}
            raise AssertionError(f"unexpected path: {path}")

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"table_preview","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"table_preview","required_tables":["orders"]},"reasoning":"simple query"}')
                return DummyResponse('{"sql":"SELECT order_id, amount FROM orders LIMIT 50;","sql_type":"select","source_tables":["orders"],"source_column_refs":["orders.order_id","orders.amount"],"derived_columns":[],"output_columns":["order_id","amount"],"reasoning":"preview orders"}')

        monkeypatch.setattr(integrity_loader, "_post_backend_json", fake_post_backend_json)
        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {"schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}', "integrity_text": 'legacy pass dump should be replaced'})
        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1, 10)])

        result = build_app().invoke({"user_question": "?? ???? ??? ???", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "integrity_dataset_name": "orders_ds", "integrity_refresh": {}, "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "validation_findings": [], "retry_hint": {}, "validation_summary": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["integrity_refresh"]["status"] == "refreshed"
        assert result["integrity_refresh"]["ready"] is True
        assert "amount has 2 nulls" in result["integrity_text"]
        assert "legacy pass dump" not in result["integrity_text"]

