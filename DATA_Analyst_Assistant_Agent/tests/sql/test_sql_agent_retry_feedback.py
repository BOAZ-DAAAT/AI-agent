from __future__ import annotations

from typing import Any

import json
from dataclasses import dataclass

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState


class _FakeApp:
    def __init__(self) -> None:
        self.invoked_with: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.invoked_with = payload
        return {}


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


def test_sql_agent_includes_derivation_contract_requests_in_supervisor_plan_context(monkeypatch) -> None:
    fake_app = _FakeApp()
    _patch_build_app(monkeypatch, fake_app)
    state = OrchestrationState(
        run_id="run_1",
        user_query="seller 표본 수를 고려해 분석해줘",
        plan=AnalysisPlan(
            goal="seller별 분석",
            required_derivations=[{
                "name": "seller_sample_order_count",
                "entity": "seller_id",
                "grain": "seller_id",
                "source_columns": ["orders.order_id"],
                "definition": "COUNT(DISTINCT order_id) per seller_id",
                "preferred_name": "seller_sample_order_count",
            }],
            analysis_heuristics=[{
                "name": "low_n_threshold",
                "default_policy": "record n<30 as a limitation",
                "must_record": True,
            }],
        ),
    )

    SQLAgent()._run_main_sql_agent(state)

    assert fake_app.invoked_with is not None
    reason = fake_app.invoked_with["planner_selection_reason"]
    payload = json.loads(reason.split("Supervisor analysis_plan:\n", 1)[1])
    assert payload["required_derivations"] == state.plan.required_derivations
    assert payload["analysis_heuristics"] == state.plan.analysis_heuristics


def test_analysis_data_contract_marks_unimplemented_required_derivation() -> None:
    contract = SQLAgent._analysis_data_contract(
        mart_design={
            "grain": "seller_id x month",
            "grain_columns": ["seller_id", "month"],
            "column_plan": [
                {
                    "output_column": "seller_order_count",
                    "role": "measure",
                    "source_columns": ["orders.order_id"],
                    "calculation_type": "derived",
                    "calculation_rule": "COUNT(*) at seller/month grain",
                    "aggregation_method": "COUNT",
                }
            ],
        },
        sql_draft={"target_table": "analytics.seller_month", "source_tables": ["orders"]},
        generated_sql="SELECT seller_id, month, COUNT(*) AS seller_order_count FROM orders GROUP BY seller_id, month",
        required_derivations=[{
            "name": "seller_sample_order_count",
            "entity": "seller_id",
            "grain": "seller_id",
            "source_columns": ["orders.order_id"],
            "definition": "COUNT(DISTINCT order_id) per seller_id",
            "preferred_name": "seller_sample_order_count",
            "not_for": ["seller/month mart row count"],
        }],
    )

    assert contract["unimplemented_derivations"][0]["preferred_name"] == "seller_sample_order_count"
    rule = contract["sample_size_rules"]["seller_id"]
    assert rule["preferred_column"] is None
    assert rule["do_not_infer_from_name_only"] is True
    assert "seller_order_count" in rule["count_like_columns_require_contract_match"]


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
