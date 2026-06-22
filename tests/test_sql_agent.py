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
