from __future__ import annotations

import json
from types import SimpleNamespace

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AgentStatus
from DATA_Analyst_Assistant_Agent.supervisor import branch as branch_module
from DATA_Analyst_Assistant_Agent.supervisor.branch import (
    _branch_plan_context_from_sql_artifacts,
    branch_from,
)


class FakeRuntimeAdapter:
    def __init__(self, artifacts: dict[str, str]) -> None:
        self.artifacts = artifacts

    def read_artifact_text(self, artifact_id: str) -> str:
        return self.artifacts[artifact_id]


class FakeBackendAdapter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def append_run_event(self, run_id: str, event_type: str, message: str, **kwargs) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                **kwargs,
            }
        )


def _sql_payload() -> dict:
    return {
        "mart_design": {
            "mart_name": "mart_seller_delivery_review_monthly",
            "target_schema": "analytics",
            "grain": "seller x order x month",
            "grain_columns": ["seller_id", "order_id", "order_month"],
            "source_tables": ["orders", "order_items", "order_reviews", "sellers"],
            "source_grains": {"orders": ["order_id"]},
            "deduplication_keys": ["seller_id", "order_id", "order_month"],
            "column_plan": [
                {
                    "output_column": "order_month",
                    "source_columns": ["orders.order_purchase_timestamp"],
                    "calculation_type": "derived",
                    "calculation_rule": "DATE_FORMAT(order_purchase_timestamp, '%Y-%m-01')",
                    "aggregation_method": "MIN",
                },
                {
                    "output_column": "delivery_days",
                    "source_columns": [
                        "orders.order_purchase_timestamp",
                        "orders.order_delivered_customer_date",
                    ],
                    "calculation_type": "derived",
                    "calculation_rule": "TIMESTAMPDIFF(DAY, order_purchase_timestamp, order_delivered_customer_date)",
                    "aggregation_method": "AVG",
                },
            ],
        },
        "sql_draft": {
            "sql": "CREATE TABLE analytics.mart_seller_delivery_review_monthly AS SELECT 1 AS delivery_days",
            "target_table": "analytics.mart_seller_delivery_review_monthly",
            "source_tables": ["orders", "order_items", "order_reviews", "sellers"],
            "business_grain": "seller x order x month",
        },
    }


def test_branch_plan_context_restores_sql_contract_from_artifact() -> None:
    runtime = SimpleNamespace(
        adapter=FakeRuntimeAdapter({"art_sql": json.dumps(_sql_payload(), ensure_ascii=False)})
    )

    context = _branch_plan_context_from_sql_artifacts(["art_sql"], runtime)

    assert context.target_table == "analytics.mart_seller_delivery_review_monthly"
    assert context.generated_sql.startswith("CREATE TABLE analytics.mart_seller_delivery_review_monthly")
    assert context.source_tables == ["orders", "order_items", "order_reviews", "sellers"]
    assert context.business_grain == "seller x order x month"
    assert context.analysis_data_contract["row_grain"] == "seller x order x month"
    assert context.analysis_data_contract["derived_columns"][0]["output_column"] == "order_month"


def test_branch_from_passes_hydrated_sql_context_to_analysis(monkeypatch) -> None:
    runtime = SimpleNamespace(
        adapter=FakeRuntimeAdapter({"art_sql": json.dumps(_sql_payload(), ensure_ascii=False)})
    )
    backend_adapter = FakeBackendAdapter()
    captured = {}

    def fake_run_agent(agent_name, state, runtime):
        captured["agent_name"] = agent_name
        captured["state"] = state
        return AgentEnvelope(
            status=AgentStatus.failed,
            agent_name=agent_name,
            summary="contract invalid",
            error="contract invalid",
            artifact_refs=[ArtifactRef(artifact_id="art_analysis_failed", type=ArtifactType.file)],
        )

    monkeypatch.setattr(branch_module, "_run_agent", fake_run_agent)

    result = branch_from(
        "analysis",
        "표본 30 미만인 판매자는 제외하고 다시 분석해줘",
        upstream_artifact_ids={"sql_agent": ["art_sql"], "eda_agent": ["art_eda"]},
        original_question="배송과 리뷰 관계를 분석해줘",
        run_id="run_branch",
        thread_id="thread_branch",
        runtime=runtime,
        backend_adapter=backend_adapter,
        parent_node_id="node_eda",
        target_table="analytics.mart_seller_delivery_review_monthly",
    )

    state = captured["state"]
    assert captured["agent_name"] == "analysis_agent"
    assert state.plan.generated_sql.startswith("CREATE TABLE analytics.mart_seller_delivery_review_monthly")
    assert state.plan.mart_design["grain"] == "seller x order x month"
    assert state.plan.analysis_data_contract["row_grain"] == "seller x order x month"
    assert result.failed_agent == "analysis_agent"
    assert backend_adapter.events[-1]["event_type"] == "agent.failed"
    assert not any(event["event_type"] == "agent.completed" for event in backend_adapter.events)
