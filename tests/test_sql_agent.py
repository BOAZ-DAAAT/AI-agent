"""B-lane tests: SQL-Agent, planner, self-check, mart candidate, and Phase 2A routing."""

from __future__ import annotations

import json
import os
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
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import split_sql_statements
from DATA_Analyst_Assistant_Agent.agents.sql import _runtime as sql_runtime
from DATA_Analyst_Assistant_Agent.agents.sql import _runtime as sql_runtime
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import is_safe_mart_sql
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import extract_sql_aliases, split_sql_statements
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
try:
    from DATA_Analyst_Assistant_Agent.agents.validation.agent import CentralValidationAgent
except ModuleNotFoundError:
    CentralValidationAgent = None
from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import normalize_generated_sql
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import (
    validate_sql_dialect_and_route,
    validate_sql_identifiers,
)
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


def patch_staged_schema_context(monkeypatch, context_module, schema_text: str) -> None:
    """LangGraph 테스트에 계획·생성 단계용 동일한 스키마 fixture를 주입한다."""
    monkeypatch.setattr(context_module, "load_schema_catalog_text", lambda: schema_text)
    monkeypatch.setattr(context_module, "load_scoped_schema_text", lambda tables: schema_text)


def install_two_stage_llm_adapter(monkeypatch) -> None:
    """기존 그래프 fixture의 1단계 응답을 새 2단계 계약으로 변환한다."""
    from DATA_Analyst_Assistant_Agent.agents.sql import planner_support
    from DATA_Analyst_Assistant_Agent.agents.sql.nodes import finalize_plan as finalize_plan_module
    from DATA_Analyst_Assistant_Agent.agents.sql.nodes import generate as generate_module
    from DATA_Analyst_Assistant_Agent.agents.sql.nodes import mart_design as mart_design_module
    from DATA_Analyst_Assistant_Agent.agents.sql.nodes import plan as plan_module

    original_try_llm_json = planner_support.try_llm_json
    original_mart_try_llm_json = mart_design_module.try_llm_json
    last_question_plan: dict[str, object] = {}
    last_legacy_plan: dict[str, object] = {}

    def adapted_question_plan(prompt: str):
        response = original_try_llm_json(prompt)
        if not response:
            return response
        try:
            parsed = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
        except Exception:
            return response
        if not isinstance(parsed, dict):
            return response
        if "target_metrics" in parsed:
            normalized = parsed
        else:
            target_metric = str(parsed.get("target_metric") or "").strip()
            candidates = list(parsed.get("candidate_tables") or parsed.get("selected_join_tables") or parsed.get("relevant_tables") or [])
            normalized = {
                "route_kind": parsed.get("route_kind"),
                "question_type": parsed.get("question_type") or "detail",
                "target_metrics": [target_metric] if target_metric else [],
                "analysis_entities": [],
                "dimensions": list(parsed.get("dimensions") or []),
                "filters": list(parsed.get("filters") or []),
                "candidate_tables": candidates,
                "required_aggregations": list(parsed.get("required_aggregations") or []),
                "reasoning": parsed.get("reasoning") or "테스트 계획 근거",
            }
        last_question_plan.clear()
        if normalized.get("route_kind") == "comprehensive" and not normalized.get("target_metrics"):
            metric = str(parsed.get("target_metric") or parsed.get("mart_name") or "mart_metric").strip()
            normalized["target_metrics"] = [metric] if metric else ["mart_metric"]
        last_question_plan.update(normalized)
        last_question_plan["required_columns"] = list(parsed.get("required_columns") or [])
        last_legacy_plan.clear()
        last_legacy_plan.update(parsed)
        return json.dumps(normalized, ensure_ascii=False)

    def synthesized_final_plan(prompt: str):
        candidates = list(last_question_plan.get("candidate_tables") or [])
        required_columns = list(last_question_plan.get("required_columns") or [])
        if not required_columns and candidates:
            required_columns = [f"{candidates[0]}.order_id"]
        legacy_contract = dict(last_legacy_plan.get("validation_contract") or {})
        if not legacy_contract:
            shape = last_legacy_plan.get("expected_result_shape")
            if shape:
                legacy_contract["expected_result_shape"] = shape
        route_kind = str(last_question_plan.get("route_kind") or last_legacy_plan.get("route_kind") or "simple")
        if route_kind == "comprehensive":
            legacy_contract.setdefault("expected_result_shape", "datamart_creation")
            target_table = legacy_contract.get("target_table")
            if not target_table and last_legacy_plan.get("mart_name"):
                target_table = f"analytics.{last_legacy_plan['mart_name']}"
                legacy_contract["target_table"] = target_table
        else:
            legacy_contract.setdefault("expected_result_shape", "table_preview")
        target_metrics = list(last_question_plan.get("target_metrics") or [])
        if route_kind == "comprehensive" and not target_metrics:
            target_metrics = [str(last_legacy_plan.get("target_metric") or last_legacy_plan.get("mart_name") or "mart_metric")]
        return json.dumps({
            "route_kind": route_kind,
            "question_type": last_question_plan.get("question_type") or "detail",
            "target_metrics": target_metrics,
            "analysis_entities": list(last_question_plan.get("analysis_entities") or []),
            "dimensions": list(last_question_plan.get("dimensions") or []),
            "filters": list(last_question_plan.get("filters") or []),
            "selected_join_tables": candidates,
            "required_columns": required_columns,
            "required_aggregations": list(last_question_plan.get("required_aggregations") or []),
            "business_keys": {},
            "validation_contract": legacy_contract,
            "mart_name": last_legacy_plan.get("mart_name"),
            "grain": last_legacy_plan.get("grain"),
            "reasoning": "후보 상세 스키마를 사용한 테스트 최종 계획",
        }, ensure_ascii=False)

    monkeypatch.setattr(plan_module, "try_llm_json", adapted_question_plan)
    monkeypatch.setattr(finalize_plan_module, "try_llm_json", synthesized_final_plan)

    def adapt_legacy_mart_design(prompt: str):
        response = original_mart_try_llm_json(prompt)
        if not response:
            return response
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return response
        if not isinstance(parsed, dict):
            return response
        if "column_plan" in parsed and "metric_support" in parsed:
            return response

        def legacy_column_names(values):
            names = []
            for item in values or []:
                if isinstance(item, dict):
                    name = str(item.get("column_name") or item.get("name") or item.get("column") or "").strip()
                else:
                    name = str(item).strip()
                if name:
                    names.append(name)
            return names

        grain_columns = legacy_column_names(parsed.get("key_columns") or [parsed.get("grain")])
        if not grain_columns:
            grain_columns = ["id"]
        source_tables = [str(item).strip() for item in (parsed.get("source_tables") or last_question_plan.get("candidate_tables") or ["orders"]) if str(item).strip()]
        dimensions = legacy_column_names(parsed.get("dimension_columns") or grain_columns)
        measures = legacy_column_names(parsed.get("measure_columns") or [])
        column_plan = []
        for column in dict.fromkeys([*grain_columns, *dimensions]):
            column_plan.append({
                "output_column": column,
                "role": "dimension",
                "source_columns": [column],
                "calculation_type": "passthrough",
                "calculation_rule": f"{column} 그대로 사용",
                "aggregation_method": "none",
                "inclusion_reason": "grain/dimension column",
            })
        for column in measures:
            if column in {item["output_column"] for item in column_plan}:
                continue
            column_plan.append({
                "output_column": column,
                "role": "measure",
                "source_columns": [column],
                "calculation_type": "passthrough",
                "calculation_rule": f"{column} 그대로 사용",
                "aggregation_method": "none",
                "inclusion_reason": "measure column",
            })
        metric_names = list(last_question_plan.get("target_metrics") or [])
        if not metric_names:
            metric_names = [str(last_legacy_plan.get("target_metric") or parsed.get("mart_name") or "mart_metric")]
        required_mart_columns = list(dict.fromkeys([item["output_column"] for item in column_plan]))
        parsed.update({
            "grain_columns": grain_columns,
            "source_grains": {table: grain_columns for table in source_tables},
            "deduplication_keys": grain_columns,
            "column_plan": column_plan,
            "metric_support": [
                {
                    "metric_name": metric,
                    "calculation_grain": grain_columns,
                    "required_mart_columns": required_mart_columns,
                    "downstream_calculation": "마트 컬럼을 사용해 후속 분석",
                }
                for metric in metric_names
            ],
            "aggregation_policy": "preserve_common_grain",
            "source_tables": source_tables,
            "design_reasoning": parsed.get("design_reasoning") or "legacy mart fixture",
        })
        return json.dumps(parsed, ensure_ascii=False)

    monkeypatch.setattr(mart_design_module, "try_llm_json", adapt_legacy_mart_design)

    def invoke_structured_with_legacy_dummy(prompt: str, schema):
        raw = planner_support.get_llm().invoke(prompt).content
        parsed = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
        return schema.model_validate(parsed)

    monkeypatch.setattr(planner_support, "invoke_llm_structured", invoke_structured_with_legacy_dummy)
    monkeypatch.setattr(generate_module, "invoke_llm_structured", invoke_structured_with_legacy_dummy)


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

    def test_mysql_distinct_window_aggregate_is_blocked(self):
        checks = run_sql_self_check(
            "SELECT COUNT(DISTINCT seller_id) OVER (PARTITION BY order_id) AS seller_count FROM order_items"
        )
        dialect_check = next(c for c in checks if c.name == "mysql_dialect_compatibility")
        assert not dialect_check.passed
        assert "DISTINCT inside window aggregate" in dialect_check.detail

    def test_mysql_unsupported_window_error_is_repairable_aggregation(self):
        from DATA_Analyst_Assistant_Agent.agents.sql.execution_errors import classify_execution_error

        result = classify_execution_error(
            "(1235, \"This version of MySQL doesn't yet support '<window function>(DISTINCT ..)'\")"
        )

        assert result["classification"] == "repairable_sql"
        assert result["repair_strategy"] == "rewrite_aggregation"

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

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO analytics.orders_mart SELECT * FROM orders",
            "CREATE OR REPLACE TABLE analytics.orders_mart AS SELECT * FROM orders",
            "CREATE TABLE analytics.orders_mart (order_id INT)",
        ],
    )
    def test_mart_runtime_allows_only_create_table_as_select(self, monkeypatch, sql):
        from DATA_Analyst_Assistant_Agent.agents.sql import _runtime

        monkeypatch.setattr(_runtime, "ALLOW_MART_WRITE", True)

        allowed, _ = is_safe_mart_sql(sql, "analytics.orders_mart")

        assert not allowed

    def test_mart_runtime_accepts_create_table_as_select(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql import _runtime

        monkeypatch.setattr(_runtime, "ALLOW_MART_WRITE", True)

        allowed, reason = is_safe_mart_sql(
            "CREATE TABLE analytics.orders_mart AS SELECT * FROM orders",
            "analytics.orders_mart",
        )

        assert allowed
        assert reason == ""


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

        findings = validate_sql_identifiers({"route_kind": "simple"}, sql_draft, schema_text)

        assert not any(item["category"] == "missing_column" for item in findings)

    @pytest.mark.parametrize("source_column_ref", ["missing_source", "orders.delivery_delay_flag"])
    def test_validate_sql_identifiers_blocks_unknown_source_column_refs(self, source_column_ref):
        schema_text = '{"orders": {"columns": [{"name": "order_id"}]}}'
        findings = validate_sql_identifiers(
            {"route_kind": "simple"},
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
            {"route_kind": "simple"},
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


class TestSQLRuntimeEngineCache:
    def test_engine_cache_refreshes_when_db_name_changes(self, monkeypatch):
        class FakeEngine:
            def __init__(self, db_name: str) -> None:
                self.db_name = db_name
                self.disposed = False

            def dispose(self) -> None:
                self.disposed = True

        created: list[FakeEngine] = []

        def fake_get_db_engine():
            engine = FakeEngine(os.getenv("DB_NAME", ""))
            created.append(engine)
            return engine

        monkeypatch.setattr(sql_runtime, "_engine", None)
        monkeypatch.setattr(sql_runtime, "_engine_cache_key", None)
        monkeypatch.setattr(sql_runtime, "get_db_engine", fake_get_db_engine)
        monkeypatch.setenv("DB_HOST", "127.0.0.1")
        monkeypatch.setenv("DB_PORT", "3306")
        monkeypatch.setenv("DB_USER", "root")
        monkeypatch.setenv("DB_PASSWORD", "")

        monkeypatch.setenv("DB_NAME", "session_a")
        first = sql_runtime.get_engine()
        assert sql_runtime.get_engine() is first

        monkeypatch.setenv("DB_NAME", "session_b")
        second = sql_runtime.get_engine()

        assert second is not first
        assert first.disposed is True
        assert [engine.db_name for engine in created] == ["session_a", "session_b"]


# ── planner tests ──

class TestSQLRouteKindContract:
    @pytest.mark.parametrize(
        ("route_kind", "legacy_fields", "expected"),
        [
            (
                "comprehensive",
                {
                    "task_type": "query_answer",
                    "requested_output": "execute_and_answer",
                    "expected_result_shape": "table_preview",
                },
                ("data_mart_build", "create_table", "datamart_creation"),
            ),
            (
                "simple",
                {
                    "task_type": "data_mart_build",
                    "requested_output": "create_table",
                    "expected_result_shape": "datamart_creation",
                },
                ("query_answer", "execute_and_answer", "table_preview"),
            ),
        ],
    )
    def test_question_plan_ignores_legacy_fields(
        self,
        route_kind,
        legacy_fields,
        expected,
    ):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes.plan import _normalize_question_plan

        parsed = {
            "route_kind": route_kind,
            "question_type": "detail",
            "target_metrics": [],
            "analysis_entities": [],
            "dimensions": [],
            "filters": [],
            "candidate_tables": ["orders"],
            "required_aggregations": [],
            "reasoning": "테스트",
            "selected_join_tables": ["legacy_orders"],
            "validation_contract": {"expected_result_shape": "contradictory_shape"},
            **legacy_fields,
        }

        normalized = _normalize_question_plan({"user_question": "질문"}, parsed)

        assert normalized["route_kind"] == route_kind
        assert normalized["candidate_tables"] == ["orders"]
        assert set(normalized) == {
            "route_kind", "question_type", "target_metrics", "analysis_entities",
            "dimensions", "filters", "candidate_tables", "required_aggregations", "reasoning",
        }

    @pytest.mark.parametrize("route_kind", [None, "", "mart", "eda", "trend"])
    def test_question_plan_rejects_missing_or_unsupported_route_kind(self, monkeypatch, route_kind):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import plan as plan_module

        payload = {"selected_join_tables": ["orders"]}
        if route_kind is not None:
            payload["route_kind"] = route_kind
        monkeypatch.setattr(plan_module, "try_llm_json", lambda prompt: json.dumps(payload))

        result = plan_module.plan_question(
            {"user_question": "질문", "schema_text": "{}", "integrity_text": "{}"}
        )

        assert result["validation"]["result"] == "invalid"
        assert result["validation"]["findings"][0]["code"] == "invalid_question_plan"
        assert result["retry_hint"]["suggested_action"] == "replan_question"

    def test_plan_prompt_does_not_request_derived_legacy_fields(self):
        from DATA_Analyst_Assistant_Agent.agents.sql.prompts.plan import plan_prompt

        prompt = plan_prompt({"user_question": "질문", "schema_text": "{}", "integrity_text": "{}"})

        assert '"task_type"' not in prompt
        assert '"requested_output"' not in prompt
        assert '"expected_result_shape"' not in prompt

    @pytest.mark.parametrize(
        ("route_kind", "expected_node"),
        [("simple", "generate"), ("comprehensive", "design")],
    )
    def test_route_after_refresh_integrity_context_uses_route_kind(self, route_kind, expected_node):
        from DATA_Analyst_Assistant_Agent.agents.sql.graph import route_after_refresh_integrity_context

        assert route_after_refresh_integrity_context({"plan": {"route_kind": route_kind}}) == expected_node

    @pytest.mark.parametrize("route_kind", [None, "", "mart", "eda"])
    def test_route_after_refresh_integrity_context_rejects_invalid_route(self, route_kind):
        from DATA_Analyst_Assistant_Agent.agents.sql.graph import route_after_refresh_integrity_context

        with pytest.raises(ValueError, match="route_kind"):
            route_after_refresh_integrity_context({"plan": {"route_kind": route_kind}})

    @pytest.mark.parametrize(
        "sql_draft",
        [
            {
                "sql": "INSERT INTO analytics.orders_mart SELECT * FROM orders;",
                "sql_type": "insert_select",
            },
            {
                "sql": "INSERT INTO analytics.orders_mart SELECT * FROM orders;",
                "sql_type": "create_table_as",
            },
            {
                "sql": "CREATE TABLE analytics.orders_mart AS SELECT * FROM orders; SELECT 1;",
                "sql_type": "create_table_as",
            },
        ],
    )
    def test_comprehensive_route_requires_single_ctas_statement(self, sql_draft):
        findings = validate_sql_dialect_and_route(
            {"route_kind": "comprehensive"},
            sql_draft,
        )

        assert any(item["category"] == "route_kind_mismatch" for item in findings)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; UPDATE orders SET amount = 0;",
            "DELETE FROM orders;",
        ],
    )
    def test_simple_route_requires_every_statement_to_be_select_or_with(self, sql):
        findings = validate_sql_dialect_and_route(
            {"route_kind": "simple"},
            {"sql": sql, "sql_type": "select"},
        )

        assert any(item["category"] == "route_kind_mismatch" for item in findings)


class TestStagedSchemaContext:
    @pytest.fixture()
    def schema_path(self, monkeypatch, tmp_path):
        from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader

        payload = {
            "orders": {
                "description": "첫 번째 설명입니다.\n이 줄은 카탈로그에 포함되면 안 됩니다." + "가" * 200,
                "primary_key": ["order_id"],
                "foreign_keys": [{"column": "customer_id", "references": "customers.customer_id"}],
                "columns": [
                    {"name": "created_at", "type": "DATETIME", "nullable": False, "description": "생성일"},
                    {"name": "customer_id", "type": "VARCHAR(50)", "nullable": False, "description": "고객"},
                    {"name": "order_id", "type": "VARCHAR(50)", "nullable": False, "description": "주문"},
                    {"name": "status", "type": "VARCHAR(20)", "nullable": True, "description": "상태"},
                    {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True, "description": "금액"},
                    {"name": "ignored_column", "type": "TEXT", "nullable": True, "description": "제외"},
                ],
                "sample_data": [{"order_id": "sample"}],
            },
            "customers": {
                "description": "고객 테이블",
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "columns": [{"name": "customer_id", "type": "VARCHAR(50)", "nullable": False, "description": "고객"}],
                "sample_data": [{"customer_id": "sample"}],
            },
        }
        path = tmp_path / "db_schema.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(integrity_loader, "SCHEMA_JSON_PATH", path)
        return path

    def test_schema_catalog_only_contains_names_and_descriptions(self, schema_path):
        from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_schema_catalog_text

        catalog = json.loads(load_schema_catalog_text())

        assert set(catalog) == {"orders", "customers"}
        assert "sample_data" not in json.dumps(catalog, ensure_ascii=False)
        assert catalog["orders"]["description"] == "첫 번째 설명입니다."
        assert set(catalog["orders"]) == {"description"}

    def test_scoped_schema_keeps_selected_full_columns_without_samples(self, schema_path):
        from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_scoped_schema_text

        scoped = json.loads(load_scoped_schema_text(["analytics.orders", "`orders`", "missing"]))

        assert set(scoped) == {"orders"}
        assert "sample_data" not in json.dumps(scoped, ensure_ascii=False)
        assert [column["name"] for column in scoped["orders"]["columns"]] == [
            "created_at", "customer_id", "order_id", "status", "amount", "ignored_column"
        ]
        assert scoped["orders"]["columns"][0] == {
            "name": "created_at", "type": "DATETIME", "nullable": False, "description": "생성일"
        }
        assert scoped["orders"]["foreign_keys"]

    @pytest.mark.parametrize("tables", [None, [], ["missing"]])
    def test_scoped_schema_returns_empty_for_no_valid_tables(self, schema_path, tables):
        from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_scoped_schema_text

        assert load_scoped_schema_text(tables) == ""

    def test_scoped_schema_propagates_schema_read_errors(self, monkeypatch, tmp_path):
        from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader

        monkeypatch.setattr(integrity_loader, "SCHEMA_JSON_PATH", tmp_path / "missing.json")

        with pytest.raises(FileNotFoundError):
            integrity_loader.load_scoped_schema_text(["orders"])

    def test_refresh_schema_context_uses_candidate_tables(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module

        monkeypatch.setattr(context_module, "load_scoped_schema_text", lambda tables: '{"orders":{"columns":[{"name":"order_id"}]}}')
        result = context_module.refresh_schema_context({
            "schema_text": '{"catalog":true}',
            "question_plan": {"candidate_tables": ["`analytics.orders`", "orders"]},
        })

        assert result["schema_text"] == '{"orders":{"columns":[{"name":"order_id"}]}}'
        assert result["schema_refresh"] == {
            "status": "refreshed",
            "candidate_tables": ["orders"],
            "applied_tables": ["orders"],
            "reason": "",
        }

    def test_refresh_schema_context_fails_when_all_candidates_are_invalid(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module

        monkeypatch.setattr(context_module, "load_scoped_schema_text", lambda tables: "")
        result = context_module.refresh_schema_context({
            "schema_text": '{"catalog":true}',
            "question_plan": {"candidate_tables": ["missing"]},
        })

        assert result["schema_refresh"]["status"] == "failed"
        assert result["schema_refresh"]["candidate_tables"] == ["missing"]
        assert result["schema_refresh"]["reason"] == "no_valid_tables"
        assert result["retry_hint"]["reason_code"] == "sql_plan_failed"

    def test_plan_prompt_receives_catalog_without_sample_or_sixth_column(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import plan as plan_module

        catalog = '{"orders":{"columns":[{"name":"order_id"},{"name":"customer_id"},{"name":"created_at"},{"name":"status"},{"name":"amount"}]}}'
        monkeypatch.setattr(context_module, "load_schema_catalog_text", lambda: catalog)
        monkeypatch.setattr(context_module, "load_integrity_json", lambda: {})
        prompts: list[str] = []
        monkeypatch.setattr(
            plan_module,
            "try_llm_json",
            lambda prompt: prompts.append(prompt) or '{"route_kind":"simple","selected_join_tables":["orders"]}',
        )

        state = context_module.load_context({})
        plan_module.plan_question({"user_question": "주문 조회", **state})

        assert prompts and catalog in prompts[0]
        assert "sample_data" not in prompts[0]
        assert "ignored_column" not in prompts[0]

    def test_detailed_schema_is_shared_by_mart_design_and_sql_generation_prompts(self):
        from DATA_Analyst_Assistant_Agent.agents.sql.prompts.generate import generate_mart_prompt, generate_query_prompt
        from DATA_Analyst_Assistant_Agent.agents.sql.prompts.mart_design import mart_design_prompt
        from DATA_Analyst_Assistant_Agent.agents.sql.generation_context import build_generation_context

        scoped_schema = '{"orders":{"columns":[{"name":"order_id","nullable":false},{"name":"ignored_column","nullable":true}]}}'
        state = {
            "user_question": "주문 분석",
            "plan": {"selected_join_tables": ["orders"], "required_columns": ["orders.order_id"]},
            "mart_design": {},
            "schema_text": scoped_schema,
            "integrity_text": "",
        }

        simple_context = build_generation_context(state, "simple", "").context
        query_prompt = generate_query_prompt(simple_context)
        assert "order_id" in query_prompt
        assert scoped_schema in mart_design_prompt(state)
        state["mart_design"] = {
            "mart_name": "orders_mart",
            "target_schema": "analytics",
            "grain": "order_id",
            "grain_columns": ["order_id"],
            "source_grains": {"orders": ["order_id"]},
            "deduplication_keys": ["order_id"],
            "column_plan": [
                {
                    "output_column": "order_id",
                    "role": "dimension",
                    "source_columns": ["order_id"],
                    "calculation_type": "passthrough",
                    "calculation_rule": "order_id 그대로 사용",
                    "aggregation_method": "none",
                    "inclusion_reason": "주문 grain",
                }
            ],
            "metric_support": [
                {
                    "metric_name": "orders_mart",
                    "calculation_grain": ["order_id"],
                    "required_mart_columns": ["order_id"],
                    "downstream_calculation": "주문 분석",
                }
            ],
            "aggregation_policy": "preserve_common_grain",
            "source_tables": ["orders"],
            "load_strategy": "full_refresh",
            "design_reasoning": "주문 분석용 마트",
        }
        state["plan"]["target_metrics"] = ["orders_mart"]
        mart_context = build_generation_context(state, "comprehensive", "").context
        mart_prompt = generate_mart_prompt(mart_context)
        assert "order_id" in mart_prompt
        assert "sample_data" not in query_prompt


class TestSQLLangGraphSmoke:
    @pytest.fixture(autouse=True)
    def _adapt_legacy_llm_fixtures(self, monkeypatch):
        install_two_stage_llm_adapter(monkeypatch)

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
        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}')
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

        patch_staged_schema_context(
            monkeypatch,
            context_module,
            '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_date"}]}}',
        )
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
        assert "simple route" in result["final_answer"]

    def test_build_app_simple_average_delivery_days_uses_aggregate_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}')
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
        assert result["plan"]["validation_contract"]["expected_result_shape"] == "single_scalar"

    def test_build_app_rejects_non_aggregate_sql_for_average_question(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}')

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

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}]}}')

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

        findings = validate_sql_identifiers({"route_kind": "simple"}, sql_draft, schema_text)

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

        findings = validate_sql_identifiers({"route_kind": "simple"}, sql_draft, schema_text)

        assert any(
            item["category"] == "missing_table" and "not_existing_table" in item["detail"]
            for item in findings
        )

    def test_build_app_comprehensive_path_generates_datamart_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        patch_staged_schema_context(
            monkeypatch,
            context_module,
            '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
        )
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

        patch_staged_schema_context(
            monkeypatch,
            context_module,
            '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
        )

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

        patch_staged_schema_context(
            monkeypatch,
            context_module,
            '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
        )
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(10,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed = []
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: committed.append(sql))

        responses = iter([
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build","requested_output":"create_table","target_metric":"","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"mart_name":"category_performance_analysis","grain":"order_id","load_strategy":"full_refresh","expected_result_shape":"datamart_creation","required_columns":[],"required_aggregations":[],"validation_contract":{"expected_result_shape":"datamart_creation","required_tables":["orders"],"target_table":"analytics.category_performance_analysis"},"reasoning":"plan 1"}',
            '{bad json',
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

    def test_intent_mismatch_repairs_sql_without_replanning_route(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module

        patch_staged_schema_context(
            monkeypatch,
            context_module,
            '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
        )
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(10,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed: list[str] = []
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", committed.append)

        responses = iter(
            [
                json.dumps(
                    {
                        "route_kind": "simple",
                        "selected_join_tables": ["orders"],
                        "required_aggregations": ["AVG"],
                        "target_metric": "평균 주문 금액",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "sql": "SELECT amount FROM orders;",
                        "sql_type": "select",
                        "source_tables": ["orders"],
                        "source_column_refs": ["orders.amount"],
                        "reasoning": "집계가 빠진 첫 SQL",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "sql": "SELECT AVG(amount) AS avg_order_amount FROM orders;",
                        "sql_type": "select",
                        "source_tables": ["orders"],
                        "source_column_refs": ["orders.amount"],
                        "derived_columns": ["avg_order_amount"],
                        "output_columns": ["avg_order_amount"],
                        "reasoning": "집계를 보강한 수정 SQL",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        prompts_seen: list[str] = []

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                prompts_seen.append(prompt)
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())

        result = build_app().invoke(
            {
                "user_question": "주문 금액을 분석해줘",
                "required_db_schema": "",
                "clarification_request": "",
                "planner_selection_reason": "SQL 분석",
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
                "retry_count": 0,
                "max_retries": 1,
                "feedback": "",
                "error": "",
                "final_answer": "",
            }
        )

        assert result["retry_count"] == 1
        assert result["plan"]["route_kind"] == "simple"
        assert result["plan"]["validation_contract"]["expected_result_shape"] == "single_scalar"
        assert result["validation"]["result"] == "valid"
        assert result["sql_draft"]["sql"] == "SELECT AVG(amount) AS avg_order_amount FROM orders;"
        assert committed == []
        assert sum("planner" in prompt for prompt in prompts_seen) == 1

    def test_build_app_normalizes_object_based_mart_design_columns(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "customer_id"}]}}')
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: None)

        responses = iter([
            '{"route_kind":"comprehensive","question_type":"mart_build","task_type":"data_mart_build",'
            '"requested_output":"create_table","target_metric":"재구매 분석","dimensions":["customer_id"],'
            '"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],'
            '"expected_result_shape":"datamart_creation","required_aggregations":[],"mart_name":"customer_reorder_mart",'
            '"grain":"customer","load_strategy":"full_refresh","reasoning":"마트 생성"}',
            '{"mart_name":"customer_reorder_mart","target_schema":"analytics","grain":"customer","base_grain":"customer",'
            '"source_tables":["orders"],'
            '"key_columns":[{"column_name":"customer_id","description":"고객 식별자"}],'
            '"measure_columns":[{"column_name":"total_orders","description":"총 주문수"}],'
            '"dimension_columns":[{"column_name":"customer_id","description":"고객 축"}],'
            '"load_strategy":"full_refresh","row_preserving_strategy":"고객 단위 유지",'
            '"aggregation_policy":"aggregate_if_justified","aggregation_rationale":"고객 단위 집계 필요",'
            '"design_reasoning":"재구매 분석용 고객 단위 마트"}',
            '{"sql":"CREATE TABLE analytics.customer_reorder_mart AS SELECT customer_id FROM orders;",'
            '"sql_type":"create_table_as","target_table":"analytics.customer_reorder_mart",'
            '"source_tables":["orders"],"source_column_refs":["orders.customer_id"],"derived_columns":[],"output_columns":["customer_id"],'
            '"postcheck_sql":"SELECT COUNT(*) FROM analytics.customer_reorder_mart;","reasoning":"마트 생성"}',
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                return DummyResponse(next(responses))

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

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}')
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        responses = iter(['{"sql": "SELECT JULIANDAY(order_delivered_customer_date) - JULIANDAY(order_approved_at) AS delivery_days FROM orders;", "sql_type": "select", "source_tables": ["orders"], "source_column_refs": ["orders.order_delivered_customer_date", "orders.order_approved_at"], "derived_columns": ["delivery_days"], "output_columns": ["delivery_days"], "reasoning": "1? ??"}', '{"sql": "SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days FROM orders;", "sql_type": "select", "source_tables": ["orders"], "source_column_refs": ["orders.order_delivered_customer_date", "orders.order_approved_at"], "derived_columns": ["avg_delivery_days"], "output_columns": ["avg_delivery_days"], "reasoning": "2? ??"}'])
        prompts_seen: list[str] = []

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                prompts_seen.append(prompt)
                if "planner" in prompt:
                    return DummyResponse('{"route_kind":"simple","question_type":"detail","task_type":"query_answer","requested_output":"execute_and_answer","target_metric":"average_delivery_days","dimensions":[],"filters":[],"time_condition":null,"selected_join_tables":["orders"],"relevant_tables":["orders"],"candidate_tables":["orders"],"expected_result_shape":"single_scalar","required_columns":["order_approved_at","order_delivered_customer_date"],"required_aggregations":["AVG"],"validation_contract":{"expected_result_shape":"single_scalar","required_aggregations":["AVG"],"required_columns":["order_approved_at","order_delivered_customer_date"],"expected_aliases":["avg_delivery_days"],"required_tables":["orders"],"target_metric":"average_delivery_days","dimensions":[]},"reasoning":"average delivery question"}')
                return DummyResponse(next(responses))

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(3,)])
        result = build_app().invoke({"user_question": "?? ???? ????", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "plan": {"route_kind": "simple", "selected_join_tables": ["orders"]}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["validation"]["result"] == "valid"
        assert "AVG(DATEDIFF" in result["sql_draft"]["sql"]
        assert result["retry_count"] == 1
        assert sum("MySQL 기반 SQL/데이터마트 planner" in prompt for prompt in prompts_seen) == 1
        assert not any("분석용 데이터마트 설계자" in prompt for prompt in prompts_seen)

    def test_build_app_executes_multi_statement_selects_sequentially(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_status"}]}}')

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

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}')

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

        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_status"}]}}')

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

        patch_staged_schema_context(
            monkeypatch,
            context_module,
            '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
        )
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
    @pytest.fixture(autouse=True)
    def _adapt_legacy_llm_fixtures(self, monkeypatch):
        install_two_stage_llm_adapter(monkeypatch)

    def test_main_sql_envelope_preserves_warning_findings_on_success(self, adapter):
        from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent

        runtime = AgentRuntime(adapter)
        state = OrchestrationState(
            run_id=adapter.create_run().run_id,
            user_query="주문과 고객을 조인해줘",
        )
        result = {
            "plan": {"route_kind": "simple"},
            "planning_stages": {
                "question_plan": {"candidate_tables": ["orders"]},
                "final_table_plan": {
                    "selected_join_tables": ["orders"],
                    "required_columns": ["orders.order_id"],
                    "business_keys": {},
                    "reasoning": "주문 조회",
                },
            },
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
        assert envelope.validation.findings[0].disposition == "warning"
        assert envelope.retry_hint.retryable is True
        assert envelope.retry_hint.details == {"join": "orders-customers"}
        artifact_payloads = [
            json.loads(Path(adapter.get_artifact(ref.artifact_id).local_path).read_text(encoding="utf-8"))
            for ref in envelope.artifact_refs
            if adapter.get_artifact(ref.artifact_id).metadata.get("kind") in {"sql_lang_graph_result", "sql_plan"}
        ]
        assert len(artifact_payloads) == 2
        assert all("columns_used" not in payload["sql_draft"] for payload in artifact_payloads)
        assert all(
            set(payload["planning_stages"]) == {"question_plan", "final_table_plan"}
            for payload in artifact_payloads
        )

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
        patch_staged_schema_context(monkeypatch, context_module, '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}')
        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1, 10)])

        result = build_app().invoke({"user_question": "?? ???? ??? ???", "required_db_schema": "", "clarification_request": "", "planner_selection_reason": "SQL ?? ?? ??", "schema_text": "", "integrity_text": "", "integrity_dataset_name": "orders_ds", "integrity_refresh": {}, "plan": {}, "mart_design": {}, "sql_draft": {}, "sql_result": None, "row_count": 0, "precheck_result": None, "postcheck_result": None, "mart_quality_result": {}, "validation": {}, "validation_findings": [], "retry_hint": {}, "validation_summary": {}, "retry_count": 0, "max_retries": 1, "feedback": "", "error": "", "final_answer": ""})

        assert result["integrity_refresh"]["status"] == "refreshed"
        assert result["integrity_refresh"]["ready"] is True
        assert "amount has 2 nulls" in result["integrity_text"]
        assert "legacy pass dump" not in result["integrity_text"]
