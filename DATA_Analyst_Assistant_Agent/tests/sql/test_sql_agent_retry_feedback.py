from __future__ import annotations

from typing import Any

import json
from dataclasses import dataclass

import pytest

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType
from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AnalysisPlan,
    OlistTemplateId,
    OlistTemplateKind,
    OrchestrationState,
)
from DATA_Analyst_Assistant_Agent.agents.sql.olist_templates import (
    build_olist_mart_design,
    build_olist_sql_draft,
    build_olist_validation_plan,
)


class _FakeApp:
    def __init__(self) -> None:
        self.invoked_with: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.invoked_with = payload
        return {}


@pytest.fixture(autouse=True)
def _enable_olist_template_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLIST_TEMPLATE_ROUTING_ENABLED", "true")


def _patch_build_app(monkeypatch, fake_app: _FakeApp) -> None:
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.graph.build_app",
        lambda: fake_app,
    )


class _ExplodingApp:
    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("SQL LangGraph must not run in contract-only mode")


@dataclass
class _FakeAdapter:
    def register_artifact(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        return ArtifactRef(artifact_id="artifact_contract", type=ArtifactType.file)


@dataclass
class _CapturingAdapter:
    calls: list[dict[str, Any]]

    def register_artifact(self, *args: Any, **kwargs: Any) -> ArtifactRef:
        self.calls.append(kwargs)
        return ArtifactRef(artifact_id=f"artifact_{len(self.calls)}", type=ArtifactType.file)


def test_sql_agent_uses_agent_feedback_when_present(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="고객 단위 RFM 세그먼트를 만들어줘",
        plan=AnalysisPlan(
            goal="고객 단위 RFM",
            retry_context={
                "agent_feedback": {
                    "sql_agent": {
                        "reason": "고객 단위가 아니라 주문 단위로 생성됨",
                        "missing_evidence": ["customer_unique_id 기준 grain"],
                        "source": "semantic",
                    }
                },
                # 죽어있던 옛 키들도 같이 와도 agent_feedback을 우선해야 한다
                "message": "",
                "query": "",
            },
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    clarification = fake_app.invoked_with["clarification_request"]
    assert "고객 단위가 아니라 주문 단위로 생성됨" in clarification
    assert "customer_unique_id 기준 grain" in clarification


def test_sql_agent_falls_back_to_legacy_retry_context_keys(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="월별 매출 추이를 보여줘",
        plan=AnalysisPlan(
            goal="월별 매출 추이",
            retry_context={"message": "syntax error", "query": "SELECT *", "step": "call_sql_agent"},
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    clarification = fake_app.invoked_with["clarification_request"]
    assert "syntax error" in clarification
    assert "SELECT *" in clarification


def test_sql_agent_clarification_empty_without_retry_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(run_id="run_1", user_query="월별 매출 추이를 보여줘")

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    assert fake_app.invoked_with["clarification_request"] == ""


def test_sql_agent_ignores_stale_template_plan_when_routing_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("OLIST_TEMPLATE_ROUTING_ENABLED", "false")
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_disabled_template",
        user_query="월별 매출과 주문 수를 보여줘",
        plan=AnalysisPlan(
            goal="월별 매출과 주문 수",
            planner_mode="deterministic",
            sql_generation_source="olist_template",
            sql_template_id=OlistTemplateId.monthly_sales_orders,
            sql_template_kind=OlistTemplateKind.query,
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    assert fake_app.invoked_with["sql_template_id"] is None
    assert fake_app.invoked_with["generation_source"] == "semantic_llm"
    assert state.plan is not None
    assert state.plan.planner_mode == "llm"
    assert state.plan.sql_template_id is None
    assert state.plan.sql_template_kind is None


def test_sql_agent_includes_query_rules_in_existing_supervisor_plan_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="자주 구매하는 고객 특징을 분석해줘",
        plan=AnalysisPlan(
            goal="구매 빈도 상위 고객 분석",
            query_rules={
                "document_id": "purchase_frequency",
                "default_metrics": ["distinct order_id 구매 횟수"],
                "entity_grain": ["customer_unique_id 기준"],
            },
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    reason = fake_app.invoked_with["planner_selection_reason"]
    payload = json.loads(reason.split("Supervisor analysis_plan:\n", 1)[1])
    assert payload["query_rules"] == state.plan.query_rules


def test_sql_agent_passes_structured_contracts_outside_planner_reason(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_contract",
        user_query="배송 지연을 분석해줘",
        plan=AnalysisPlan(
            goal="배송 지연 분석",
            route_kind="simple",
            sql_generation_source="olist_template",
            sql_template_id=OlistTemplateId.delivery_delay_summary,
            sql_template_kind=OlistTemplateKind.query,
            required_derivations=[
                {
                    "name": "배송 지연 일수",
                    "preferred_name": "delivery_delay_days",
                    "source_columns": ["delivered_at", "estimated_at"],
                }
            ],
            analysis_heuristics=[{"name": "이상치 민감도 기록"}],
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    assert fake_app.invoked_with["required_derivations"][0]["preferred_name"] == "delivery_delay_days"
    assert fake_app.invoked_with["analysis_heuristics"][0]["must_record"] is True
    assert "required_derivations" not in fake_app.invoked_with["planner_selection_reason"]
    assert "analysis_heuristics" not in fake_app.invoked_with["planner_selection_reason"]
    assert fake_app.invoked_with["sql_template_id"] is None
    assert fake_app.invoked_with["generation_source"] == "semantic_llm"
    assert state.plan is not None
    assert state.plan.route_kind == "comprehensive"
    assert state.plan.requires_mart_review is True


def test_analysis_data_contract_records_derivation_implementation_status() -> None:
    required = [
        {"name": "배송 지연 일수", "preferred_name": "delivery_delay_days"},
        {"name": "지역 거리", "preferred_name": "geo_distance_km"},
    ]
    mart_design = {
        "grain": "order_id",
        "grain_columns": ["order_id"],
        "column_plan": [
            {
                "output_column": "delivery_delay_days",
                "role": "measure",
                "source_columns": ["delivered_at", "estimated_at"],
                "calculation_type": "derived",
                "calculation_rule": "날짜 차이",
                "aggregation_method": "none",
            }
        ],
        "unimplemented_derivations": [
            {
                "name": "geo_distance_km",
                "reason": "좌표 컬럼 없음",
                "required_columns": ["latitude", "longitude"],
            }
        ],
    }

    contract = SQLAgent._analysis_data_contract(
        mart_design,
        {"target_table": "analytics.delivery_mart"},
        "CREATE TABLE analytics.delivery_mart AS SELECT ...",
        required_derivations=required,
        analysis_heuristics=[{"name": "이상치 민감도 기록", "must_record": True}],
    )

    assert contract["required_derivations"] == required
    assert contract["analysis_heuristics"][0]["name"] == "이상치 민감도 기록"
    assert contract["implemented_derivations"][0]["output_column"] == "delivery_delay_days"
    assert contract["unimplemented_derivations"][0]["reason"] == "좌표 컬럼 없음"
    assert contract["unimplemented_derivations"][0]["output_column"] == "geo_distance_km"


def test_sql_agent_passes_template_source_and_retry_limit_without_llm_planning(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_olist",
        user_query="월별 매출과 주문 수를 보여줘",
        catalog_summary={"orders": {"columns": []}},
        plan=AnalysisPlan(
            goal="월별 매출과 주문 수",
            planner_mode="deterministic",
            sql_generation_source="olist_template",
            sql_template_id=OlistTemplateId.monthly_sales_orders,
            sql_template_kind=OlistTemplateKind.query,
            sql_template_parameters={"start_date": "2017-01-01"},
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    assert fake_app.invoked_with["sql_template_id"] == "monthly_sales_orders"
    assert fake_app.invoked_with["generation_source"] == "olist_template"
    assert fake_app.invoked_with["sql_template_kind"] == "query"
    assert fake_app.invoked_with["sql_template_parameters"]["start_date"] == "2017-01-01"
    assert fake_app.invoked_with["max_retries"] == 1
    assert fake_app.invoked_with["max_repair_retries"] == 2


def test_template_source_is_written_to_plan_result_artifact_metadata_and_preview() -> None:
    template_id = OlistTemplateId.monthly_sales_orders
    draft = build_olist_sql_draft(template_id).model_dump()
    result = {
        "plan": build_olist_validation_plan(template_id),
        "sql_draft": draft,
        "sql_result": [],
        "row_count": 0,
        "validation": {"result": "valid", "reason": "통과"},
        "validation_findings": [],
        "retry_hint": {"retryable": False, "reason_code": "none"},
        "generation_source": "olist_template",
        "sql_generation_source": "olist_template",
        "sql_template_id": template_id.value,
        "final_answer": "완료",
    }
    state = OrchestrationState(
        run_id="run_template_artifacts",
        user_query="월별 매출과 주문 수",
        plan=AnalysisPlan(
            goal="월별 매출과 주문 수",
            planner_mode="deterministic",
            sql_generation_source="olist_template",
            sql_template_id=template_id,
        ),
    )
    adapter = _CapturingAdapter(calls=[])

    envelope = SQLAgent()._envelope_from_main_result(state, AgentRuntime(adapter=adapter), result)  # type: ignore[arg-type]

    assert envelope.status.value == "success"
    assert state.plan is not None
    assert state.plan.sql_generation_source == "olist_template"
    assert state.plan.sql_template_id == template_id
    for call in adapter.calls:
        assert call["metadata"]["sql_generation_source"] == "olist_template"
        assert call["metadata"]["sql_template_id"] == template_id.value
        assert call["preview"]["sql_generation_source"] == "olist_template"
        assert call["preview"]["sql_template_id"] == template_id.value


def test_mart_template_reference_and_kind_are_preserved_in_plan_and_artifacts() -> None:
    template_id = OlistTemplateId.customer_rfm
    draft = build_olist_sql_draft(template_id).model_dump()
    result = {
        "plan": build_olist_validation_plan(template_id),
        "mart_design": build_olist_mart_design(template_id),
        "sql_draft": draft,
        "sql_result": [{"customer_unique_id": "c1", "frequency": 2}],
        "row_count": 1,
        "validation": {"result": "valid", "reason": "통과"},
        "validation_findings": [],
        "retry_hint": {"retryable": False, "reason_code": "none"},
        "generation_source": "olist_template",
        "sql_generation_source": "olist_template",
        "sql_template_id": template_id.value,
        "sql_template_kind": "mart",
        "sql_template_parameters": {},
        "final_answer": "완료",
    }
    state = OrchestrationState(
        run_id="run_mart_template_artifacts",
        user_query="RFM 데이터마트",
        plan=AnalysisPlan(
            goal="RFM 데이터마트",
            route_kind="comprehensive",
            planner_mode="deterministic",
            sql_generation_source="olist_template",
            sql_template_id=template_id,
            sql_template_kind=OlistTemplateKind.mart,
        ),
    )
    adapter = _CapturingAdapter(calls=[])

    envelope = SQLAgent()._envelope_from_main_result(state, AgentRuntime(adapter=adapter), result)  # type: ignore[arg-type]

    assert envelope.status.value == "success"
    assert state.plan is not None
    assert state.plan.target_table == "analytics.olist_customer_rfm"
    assert state.plan.sql_template_kind == OlistTemplateKind.mart
    template_artifacts = [call for call in adapter.calls if call["metadata"].get("sql_template_id")]
    assert template_artifacts
    assert all(call["metadata"]["sql_template_kind"] == "mart" for call in template_artifacts)
    assert any(call["metadata"].get("target_table") == state.plan.target_table for call in template_artifacts)


def test_sql_agent_contract_only_repairs_metadata_without_regenerating_sql(monkeypatch) -> None:
    _patch_build_app(monkeypatch, _ExplodingApp())
    state = OrchestrationState(
        run_id="run_1",
        user_query="판매자별 월별 배송 추세를 분석해줘",
        generated_sql="SELECT seller_id, month, AVG(delivery_days) AS avg_delivery_days FROM mart GROUP BY seller_id, month",
        plan=AnalysisPlan(
            goal="판매자별 월별 배송 추세",
            generated_sql="SELECT seller_id, month, AVG(delivery_days) AS avg_delivery_days FROM mart GROUP BY seller_id, month",
            source_sql="SELECT seller_id, month, AVG(delivery_days) AS avg_delivery_days FROM mart GROUP BY seller_id, month",
            target_table="analytics.seller_month_delivery",
            retry_context={
                "mode": "contract_only",
                "suggested_action": "repair_analysis_data_contract",
                "last_failure": {
                    "reason_code": "analysis_contract_invalid",
                    "suggested_action": "repair_analysis_data_contract",
                },
            },
        ),
    )

    envelope = SQLAgent().run(state, AgentRuntime(adapter=_FakeAdapter()))  # type: ignore[arg-type]

    assert envelope.status.value == "success"
    assert envelope.retry_hint.reason_code == "analysis_data_contract_repaired"
    assert state.plan is not None
    assert state.plan.generated_sql == state.generated_sql
    assert state.plan.analysis_data_contract["row_grain"] == "seller_id, month"
    assert state.plan.analysis_data_contract["generated_sql"] == state.generated_sql


def test_retryable_warning_finding_uses_warning_disposition_not_error() -> None:
    finding = SQLAgent._validation_finding(
        {
            "category": "intent_mismatch",
            "severity": "warning",
            "retryable": True,
            "detail": "Required aggregation hint NTILE was not explicitly detected in SQL.",
        }
    )

    assert finding.disposition == "warning"
    assert finding.retryable is True
