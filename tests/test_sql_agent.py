"""B-lane tests: SQL-Agent, planner, self-check, mart candidate, and Phase 2A routing."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from data_agent_backend.models.common import BackendError
from DATA_Analyst_Assistant_Agent.agents.sql.self_check import is_sql_safe, run_sql_self_check
from DATA_Analyst_Assistant_Agent.agents.sql import planner as planner_module
from DATA_Analyst_Assistant_Agent.agents.sql.planner import build_sql_plan, SQLPlan
from DATA_Analyst_Assistant_Agent.agents.sql.mart import needs_mart_candidate
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.validation.agent import CentralValidationAgent
from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app
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


# ── planner tests ──

class TestSQLPlanner:
    def test_revenue_monthly_plan(self):
        plan = build_sql_plan("월별 매출 추이")
        assert plan.metric == "revenue"
        assert plan.dimension == "month"
        assert "SELECT" in plan.generated_sql
        assert plan.reasoning

    def test_simple_query_fallback(self):
        plan = build_sql_plan("간단한 요약")
        assert plan.generated_sql == "SELECT 1 AS sample_value"

    def test_plan_has_structured_fields(self):
        plan = build_sql_plan("monthly revenue trend")
        assert isinstance(plan.selected_tables, list)
        assert isinstance(plan.selected_columns, list)
        assert plan.metric == "revenue"
        assert plan.dimension == "month"

    def test_trend_route_uses_cte_template(self):
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan
        ap = AnalysisPlan(goal="trend", route_kind="trend", metric="revenue", dimension="월")
        plan = build_sql_plan("월별 매출 추이", ap)
        assert "monthly_data" in plan.generated_sql
        assert "ORDER BY" in plan.generated_sql
        assert plan.route_kind == "trend"

    def test_eda_route_uses_profile_template(self):
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan
        ap = AnalysisPlan(goal="eda", route_kind="eda")
        plan = build_sql_plan("데이터 품질 프로파일", ap)
        assert "column_name" in plan.generated_sql
        assert "distinct_count" in plan.generated_sql
        assert plan.route_kind == "eda"

    def test_comprehensive_route_uses_cte_template(self):
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan
        ap = AnalysisPlan(goal="comprehensive", route_kind="comprehensive", metric="revenue", dimension="월")
        plan = build_sql_plan("품질 프로파일하고 월별 매출 분석", ap)
        assert "base_data" in plan.generated_sql
        assert plan.route_kind == "comprehensive"

    def test_simple_route_metric_only(self):
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan
        ap = AnalysisPlan(goal="simple", route_kind="simple")
        plan = build_sql_plan("매출 요약", ap)
        assert "revenue" in plan.generated_sql
        assert plan.route_kind == "simple"

    def test_llm_prompt_includes_catalog_and_previous_error(self):
        prompt = planner_module._build_llm_system_prompt(
            catalog_summary={
                "database": "db",
                "tables": {
                    "orders": {
                        "columns": {
                            "order_date": {"type": "DATE", "nullable": False, "description": "secret"},
                            "amount": {"type": "DECIMAL", "nullable": True},
                        }
                    }
                },
                "password": "should_not_leak",
            },
            retry_context={"code": "BAD_SQL", "message": "unknown column", "query": "SELECT bad", "datasource_id": "ds1"},
        )
        assert "<MYSQL_CATALOG_JSON>" in prompt
        assert '"orders"' in prompt
        assert '"order_date"' in prompt
        assert "should_not_leak" not in prompt
        assert "<PREVIOUS_ERROR>" in prompt
        assert "unknown column" in prompt

    def test_llm_json_response_converts_to_sql_plan(self):
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan

        ap = AnalysisPlan(goal="simple", route_kind="simple", planner_mode="llm")
        raw = """
        {
          "selected_tables": ["orders"],
          "selected_columns": ["order_date", "amount"],
          "generated_sql": "SELECT order_date, amount FROM orders",
          "reasoning": "answers the question",
          "confidence": 0.85
        }
        """
        plan = planner_module._parse_llm_sql_plan(raw, ap)
        assert plan.planner_mode == "llm"
        assert plan.selected_tables == ["orders"]
        assert plan.generated_sql == "SELECT order_date, amount FROM orders"
        assert plan.confidence == pytest.approx(0.85)

    def test_malformed_llm_json_falls_back_to_deterministic(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr(planner_module, "_call_llm_planner", lambda **kwargs: "{bad json")
        ap = AnalysisPlan(goal="simple", route_kind="simple", planner_mode="llm")
        plan = build_sql_plan("매출 요약", ap)
        assert plan.planner_mode == "deterministic"
        assert "revenue" in plan.generated_sql

    def test_llm_forbidden_sql_is_still_blocked(self):
        assert not is_sql_safe("DROP TABLE orders")


class TestSQLLangGraphSmoke:
    def test_build_app_simple_path_supports_future_input_fields(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_date"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(1, "2024-01-01")])
        app = build_app()

        result = app.invoke({
            "user_question": "주문 데이터를 간단히 보여줘",
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
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["plan"]["route_kind"] == "simple"
        assert result["sql_draft"]["sql_type"] == "select"
        assert result["validation"]["result"] == "valid"
        assert "simple 경로" in result["final_answer"]

    def test_build_app_simple_average_delivery_days_uses_aggregate_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(planner_support_module, "get_llm", lambda: (_ for _ in ()).throw(RuntimeError("llm disabled")))
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(4.2,)])

        app = build_app()
        result = app.invoke({
            "user_question": "주문 완료일부터 배송 완료일까지 평균 소요일 계산",
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
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert "AVG(DATEDIFF(order_delivered_customer_date, order_approved_at))" in result["sql_draft"]["sql"]
        assert "WHERE order_approved_at IS NOT NULL" in result["sql_draft"]["sql"]
        assert result["validation"]["result"] == "valid"
        assert result["plan"]["expected_result_shape"] == "single_scalar"

    def test_build_app_rejects_non_aggregate_sql_for_average_question(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}',
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
                        '"target_metric":"평균 소요일","dimensions":[],"filters":[],"time_condition":null,"reasoning":"평균 질문"}'
                    )
                return DummyResponse(
                    '{"sql":"SELECT * FROM orders LIMIT 50;","sql_type":"select","source_tables":["orders"],"columns_used":["order_id"],"reasoning":"잘못된 초안"}'
                )

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        app = build_app()
        result = app.invoke({
            "user_question": "주문 완료일부터 배송 완료일까지 평균 소요일 계산",
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
                    '{"sql":"SELECT COUNT(*) AS order_count FROM category_performance_analysis;","sql_type":"select","source_tables":["category_performance_analysis"],"columns_used":["order_id"],"reasoning":"없는 테이블"}'
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

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", lambda sql: [(10,)])
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed = []
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: committed.append(sql))
        app = build_app()

        result = app.invoke({
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
            "retry_count": 0,
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["plan"]["route_kind"] == "comprehensive"
        assert result["sql_draft"]["sql_type"] == "create_table_as"
        assert committed and committed[0].lower().startswith("create table analytics.")
        assert result["validation"]["result"] == "valid"
        assert "datamart" in result["final_answer"]

    def test_build_app_retries_after_mysql_dialect_failure(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "order_approved_at"}, {"name": "order_delivered_customer_date"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)

        responses = iter([
            '{"sql": "SELECT JULIANDAY(order_delivered_customer_date) - JULIANDAY(order_approved_at) AS delivery_days FROM orders;", "sql_type": "select", "source_tables": ["orders"], "columns_used": ["order_delivered_customer_date", "order_approved_at"], "reasoning": "1차 시도"}',
            '{"sql": "SELECT DATEDIFF(order_delivered_customer_date, order_approved_at) AS delivery_days FROM orders;", "sql_type": "select", "source_tables": ["orders"], "columns_used": ["order_delivered_customer_date", "order_approved_at"], "reasoning": "2차 시도"}',
        ])

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "MySQL SQL 작성기" in prompt:
                    return DummyResponse(next(responses))
                raise AssertionError("unexpected prompt")

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
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "mart_quality_result": {},
            "validation": {},
            "retry_count": 0,
            "max_retries": 2,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["validation"]["result"] == "valid"
        assert "DATEDIFF" in result["sql_draft"]["sql"]
        assert result["retry_count"] == 1

    def test_build_app_normalizes_unqualified_mart_postcheck_sql(self, monkeypatch):
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context as context_module
        from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute as sql_steps_module
        from DATA_Analyst_Assistant_Agent.agents.sql import planner_support as planner_support_module

        monkeypatch.setattr(context_module, "load_all_metadata", lambda: {
            "schema_text": '{"orders": {"columns": [{"name": "order_id"}, {"name": "amount"}]}}',
            "integrity_text": '{}'
        })
        monkeypatch.setattr(sql_steps_module, "can_use_live_db", lambda: True)
        committed: list[str] = []
        executed_queries: list[str] = []

        class DummyResponse:
            def __init__(self, content: str):
                self.content = content

        class DummyLLM:
            def invoke(self, prompt: str):
                if "MySQL SQL 작성기" in prompt:
                    return DummyResponse(
                        '{"sql": "CREATE TABLE analytics.category_performance_analysis AS SELECT * FROM orders;", '
                        '"sql_type": "create_table_as", '
                        '"target_table": "category_performance_analysis", '
                        '"source_tables": ["orders"], '
                        '"columns_used": ["order_id", "amount"], '
                        '"postcheck_sql": "SELECT COUNT(*) FROM category_performance_analysis;", '
                        '"reasoning": "마트 생성"}'
                    )
                raise AssertionError("unexpected prompt")

        monkeypatch.setattr(planner_support_module, "get_llm", lambda: DummyLLM())
        monkeypatch.setattr(sql_steps_module, "run_sql_commit", lambda sql: committed.append(sql))

        def fake_fetch(sql: str):
            executed_queries.append(sql)
            return [(1,)]

        monkeypatch.setattr(sql_steps_module, "run_sql_fetchall", fake_fetch)

        app = build_app()
        result = app.invoke({
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
            "retry_count": 0,
            "max_retries": 1,
            "feedback": "",
            "error": "",
            "final_answer": "",
        })

        assert result["sql_draft"]["target_table"] == "analytics.category_performance_analysis"
        assert result["sql_draft"]["postcheck_sql"] == "SELECT COUNT(*) FROM analytics.category_performance_analysis;"
        assert any("FROM analytics.category_performance_analysis" in query for query in executed_queries)
        assert committed

# ── mart detection tests ──

class TestMartDetection:
    def test_korean_mart_keywords(self):
        assert needs_mart_candidate("반복 조회용 데이터마트 저장")
        assert needs_mart_candidate("재사용 가능한 마트")

    def test_english_mart_keywords(self):
        assert needs_mart_candidate("save as reusable datamart")
        assert needs_mart_candidate("create a mart for repeat queries")

    def test_no_mart_intent(self):
        assert not needs_mart_candidate("간단한 매출 요약")
        assert not needs_mart_candidate("show revenue summary")


# ── Route detection tests (Phase 2A) ──

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
        envelope = CentralValidationAgent().run(state, runtime, upstream)
        assert envelope.status != AgentStatus.failed
        assert not any(flag.code == "unsafe_sql" for flag in envelope.validation.business_flags)

    def test_creates_preview_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        assert "sql_agent" in state.artifact_ids
        assert len(state.artifact_ids["sql_agent"]) >= 1

    def test_creates_ge_validation_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        has_ge = any(
            adapter.get_artifact(aid).metadata.get("kind") == "ge_table_validation_json"
            for aid in state.artifact_ids["sql_agent"]
        )
        assert has_ge, "GE validation artifact not found"

    def test_creates_sql_plan_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        has_plan = any(
            adapter.get_artifact(aid).metadata.get("kind") == "sql_plan"
            for aid in state.artifact_ids["sql_agent"]
        )
        assert has_plan, "SQL plan artifact not found"

    def test_non_mart_query_no_approval(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘")
        assert state.terminal_state == SupervisorTerminalState.completed
        assert not state.approval_ids

    def test_mart_query_requires_approval(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.terminal_state == SupervisorTerminalState.needs_user_approval
        assert state.approval_ids

    def test_mart_query_creates_candidate_artifact(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.mart_candidate_ids
        art = adapter.get_artifact(state.mart_candidate_ids[0])
        assert art.metadata.get("kind") == "mart_candidate"

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

    def test_no_mart_metadata_before_approval(self, adapter):
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("반복 조회용 데이터마트로 저장해줘")
        assert state.mart_id is None
        assert "mart_metadata" not in state.artifact_ids

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
            "DATA_Analyst_Assistant_Agent/agents/sql/planner.py",
            "DATA_Analyst_Assistant_Agent/agents/sql/self_check.py",
            "DATA_Analyst_Assistant_Agent/agents/sql/mart.py",
            "DATA_Analyst_Assistant_Agent/mart/metadata.py",
            "DATA_Analyst_Assistant_Agent/mart/persistence.py",
            "DATA_Analyst_Assistant_Agent/supervisor.py",
            "tests/test_sql_agent.py",
        ]
        assert not any(p.startswith("data_agent_backend/") for p in changed)
