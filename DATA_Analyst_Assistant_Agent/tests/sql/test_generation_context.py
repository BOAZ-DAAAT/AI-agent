"""generate_sql 최소 컨텍스트 투영 회귀 테스트."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from DATA_Analyst_Assistant_Agent.agents.sql.generation_context import (
    ComprehensiveSQLGenerationContext,
    SimpleSQLGenerationContext,
    build_generation_context,
)
from DATA_Analyst_Assistant_Agent.agents.sql.nodes import generate as generate_node


def _schema() -> dict:
    long_description = "필요한 컬럼 설명 " + ("아주 긴 설명 " * 30) + "\n둘째 줄은 제외"
    return {
        "customers": {
            "description": "프롬프트에서 제외할 고객 테이블 설명",
            "primary_key": ["customer_id"],
            "foreign_keys": [],
            "sample_data": [{"고객 샘플 비밀": "제외"}],
            "columns": [
                {"name": "customer_id", "type": "VARCHAR", "nullable": False, "description": "고객 조인 키"},
                {"name": "customer_unique_id", "type": "VARCHAR", "nullable": False, "description": long_description},
                {"name": "unused_customer_secret", "type": "TEXT", "nullable": True, "description": "제외 대상"},
            ],
        },
        "orders": {
            "description": "프롬프트에서 제외할 주문 테이블 설명",
            "primary_key": ["order_id"],
            "foreign_keys": [
                {
                    "column": "customer_id",
                    "references": {"table": "customers", "column": "customer_id"},
                }
            ],
            "sample_data": [{"주문 샘플 비밀": "제외"}],
            "columns": [
                {"name": "order_id", "type": "VARCHAR", "nullable": False, "description": "주문 키"},
                {"name": "customer_id", "type": "VARCHAR", "nullable": False, "description": "고객 조인 키"},
                {"name": "amount", "type": "DECIMAL", "nullable": False, "description": "주문 금액"},
                {"name": "created_at", "type": "DATETIME", "nullable": False, "description": "주문 시각"},
                {"name": "unused_order_secret", "type": "TEXT", "nullable": True, "description": "제외 대상"},
            ],
        },
        "unrelated": {
            "description": "무관한 테이블 sentinel",
            "columns": [{"name": "secret", "description": "무관한 컬럼 sentinel"}],
        },
    }


def _simple_state(**overrides) -> dict:
    plan = {
        "route_kind": "simple",
        "filters": ["orders.created_at >= '2024-01-01'"],
        "selected_join_tables": ["customers", "orders"],
        "required_columns": ["customers.customer_unique_id", "orders.amount"],
        "business_keys": {"customers": "customer_id", "orders": "order_id"},
        "dimensions": ["고객"],
        "required_aggregations": ["SUM"],
        "validation_contract": {"expected_result_shape": "table_preview", "전체 계약 sentinel": "제외"},
        "question_plan_reasoning": "질문 reasoning sentinel",
        "table_plan_reasoning": "테이블 reasoning sentinel",
    }
    state = {
        "user_question": "2024년 고객별 주문 금액을 보여줘",
        "plan": plan,
        "schema_text": json.dumps(_schema(), ensure_ascii=False),
        "integrity_text": "\n".join([
            "Integrity context (only failures/warnings/stale checks):",
            "- [FAIL] customers.customer_unique_id: null_check — 관련 실패",
            "- [FAIL] customers.unused_customer_secret: secret_check — 제외할 실패",
            "- [WARNING] orders.amount: warning_check — 실패 상태 아님",
            "- [ERROR] orders.Table-Level: row_check — 테이블 실패",
            "- [ACTION_REQUIRED] unrelated: table_check — 무관한 실패",
        ]),
        "mart_design": {},
        "planning_stages": {"sentinel": "전체 planning artifact 제외"},
        "retry_count": 0,
        "retry_hint": {},
        "feedback": "",
        "error": "",
    }
    state.update(overrides)
    return state


def _mart_design() -> dict:
    return {
        "mart_name": "customer_order_mart",
        "target_schema": "임의 스키마는 무시",
        "grain": "고객 × 주문",
        "grain_columns": ["customer_unique_id", "order_id"],
        "source_grains": {"customers": ["customer_id"], "orders": ["order_id"]},
        "deduplication_keys": ["customer_unique_id", "order_id"],
        "column_plan": [
            {
                "output_column": "customer_unique_id",
                "role": "dimension",
                "source_columns": ["customers.customer_unique_id"],
                "calculation_type": "passthrough",
                "calculation_rule": "고객 키를 보존",
                "aggregation_method": "none",
                "inclusion_reason": "프롬프트 제외 sentinel",
            },
            {
                "output_column": "order_id",
                "role": "dimension",
                "source_columns": ["orders.order_id"],
                "calculation_type": "passthrough",
                "calculation_rule": "주문 키를 보존",
                "aggregation_method": "none",
                "inclusion_reason": "프롬프트 제외 sentinel",
            },
            {
                "output_column": "amount_sum",
                "role": "measure",
                "source_columns": ["orders.amount"],
                "calculation_type": "derived",
                "calculation_rule": "주문 금액 합계",
                "aggregation_method": "SUM",
                "inclusion_reason": "프롬프트 제외 sentinel",
            },
        ],
        "metric_support": [
            {
                "metric_name": "매출",
                "calculation_grain": ["customer_unique_id"],
                "required_mart_columns": ["customer_unique_id", "amount_sum"],
                "downstream_calculation": "고객별 amount_sum을 합산한다",
            }
        ],
        "aggregation_policy": "aggregate_to_common_grain",
        "source_tables": ["customers", "orders"],
        "load_strategy": "full_refresh",
        "design_reasoning": "설계 reasoning sentinel",
    }


def _comprehensive_state(**overrides) -> dict:
    state = _simple_state()
    state["plan"] = {**state["plan"], "route_kind": "comprehensive", "target_metrics": ["매출"]}
    state["mart_design"] = _mart_design()
    state.update(overrides)
    return state


def test_internal_context_models_validate_and_serialize_compact_json():
    common = {
        "user_question": "질문",
        "filters": [],
        "selected_tables": ["orders"],
        "required_columns": ["orders.order_id"],
        "business_keys": {},
        "previous_feedback": "",
        "schema": {"orders": {"columns": []}},
        "integrity_failures": [],
    }
    simple = SimpleSQLGenerationContext(
        **common, dimensions=[], required_aggregations=["COUNT"], expected_result_shape="scalar"
    )
    comprehensive = ComprehensiveSQLGenerationContext(
        **common,
        target_table="analytics.orders_mart",
        source_grains={"orders": ["order_id"]},
        final_grain={"description": "주문", "grain_columns": ["order_id"], "deduplication_keys": ["order_id"]},
        column_plan=[{
            "output_column": "order_id",
            "source_columns": ["orders.order_id"],
            "role": "dimension",
            "calculation_type": "passthrough",
            "calculation_rule": "주문 키 보존",
            "aggregation_method": "none",
        }],
        metric_support=[{
            "metric_name": "주문 수",
            "calculation_grain": ["order_id"],
            "required_mart_columns": ["order_id"],
            "downstream_calculation": "행 수를 계산한다",
        }],
        aggregation_policy="preserve_common_grain",
    )

    assert "\n" not in simple.model_dump_json(by_alias=True)
    assert '"schema":' in comprehensive.model_dump_json(by_alias=True)
    with pytest.raises(ValidationError):
        SimpleSQLGenerationContext(**{**common, "required_columns": []}, dimensions=[], required_aggregations=[])


def test_simple_projection_keeps_contract_keys_and_removes_unrelated_context():
    result = build_generation_context(_simple_state(), "simple", "재생성 피드백")
    payload = json.loads(result.context_json)
    prompt_text = result.context_json

    assert payload["previous_feedback"] == "재생성 피드백"
    assert payload["expected_result_shape"] == "table_preview"
    assert payload["dimensions"] == ["고객"]
    assert payload["required_aggregations"] == ["SUM"]
    assert "question_plan_reasoning" not in prompt_text
    assert "전체 planning artifact 제외" not in prompt_text
    assert "전체 계약 sentinel" not in prompt_text
    assert "무관한 테이블 sentinel" not in prompt_text
    assert "무관한 컬럼 sentinel" not in prompt_text
    assert "샘플 비밀" not in prompt_text
    assert "unused_customer_secret" not in json.dumps(payload["schema"], ensure_ascii=False)

    schema = payload["schema"]
    assert set(schema) == {"customers", "orders"}
    assert {column["name"] for column in schema["customers"]["columns"]} == {"customer_id", "customer_unique_id"}
    assert {column["name"] for column in schema["orders"]["columns"]} == {
        "order_id", "customer_id", "amount", "created_at"
    }
    assert len(next(column for column in schema["customers"]["columns"] if column["name"] == "customer_unique_id")["description"]) <= 160
    assert schema["orders"]["foreign_keys"]
    assert payload["integrity_failures"] == [
        "- [FAIL] customers.customer_unique_id: null_check — 관련 실패",
        "- [ERROR] orders.Table-Level: row_check — 테이블 실패",
    ]


def test_comprehensive_projection_uses_validated_mart_contract():
    result = build_generation_context(_comprehensive_state(), "comprehensive", "")
    payload = json.loads(result.context_json)
    text = result.context_json

    assert payload["target_table"] == "analytics.customer_order_mart"
    assert payload["source_grains"] == {"customers": ["customer_id"], "orders": ["order_id"]}
    assert payload["final_grain"] == {
        "description": "고객 × 주문",
        "grain_columns": ["customer_unique_id", "order_id"],
        "deduplication_keys": ["customer_unique_id", "order_id"],
    }
    assert payload["column_plan"][2]["source_columns"] == ["orders.amount"]
    assert payload["column_plan"][2]["output_column"] == "amount_sum"
    assert payload["aggregation_policy"] == "aggregate_to_common_grain"
    assert payload["metric_support"][0]["downstream_calculation"] == "고객별 amount_sum을 합산한다"
    assert "inclusion_reason" not in text
    assert "design_reasoning" not in text
    assert "dimension_columns" not in text
    assert "measure_columns" not in text
    assert "key_columns" not in text


@pytest.mark.parametrize(
    ("state_update", "reason_code"),
    [
        ({"schema_text": "{not-json"}, "malformed_schema_json"),
        ({"plan_required": ["missing_column"]}, "ambiguous_or_missing_column_reference"),
        ({"plan_required": ["customer_id"]}, "ambiguous_or_missing_column_reference"),
        ({"selected_with_empty": True}, "empty_table_projection"),
    ],
)
def test_schema_projection_falls_back_to_original_scoped_text(state_update, reason_code):
    state = _simple_state()
    if "schema_text" in state_update:
        state["schema_text"] = state_update["schema_text"]
    if "plan_required" in state_update:
        state["plan"] = {**state["plan"], "required_columns": state_update["plan_required"]}
    if state_update.get("selected_with_empty"):
        schema = _schema()
        schema["audit"] = {"columns": [{"name": "event"}], "primary_key": [], "foreign_keys": []}
        state["schema_text"] = json.dumps(schema, ensure_ascii=False)
        state["plan"] = {**state["plan"], "selected_join_tables": ["customers", "orders", "audit"]}
    original = state["schema_text"]

    result = build_generation_context(state, "simple", "")
    payload = json.loads(result.context_json)

    assert payload["schema"] == original
    assert state["schema_text"] == original
    assert result.diagnostics["fallback_reason_codes"]["schema"] == reason_code
    assert "schema" in result.diagnostics["fallback_components"]


def test_malformed_integrity_falls_back_but_empty_integrity_is_normal():
    malformed_state = _simple_state(integrity_text="해석할 수 없는 정합성 한 줄")
    malformed = build_generation_context(malformed_state, "simple", "")
    empty = build_generation_context(_simple_state(integrity_text=""), "simple", "")

    assert json.loads(malformed.context_json)["integrity_failures"] == malformed_state["integrity_text"]
    assert malformed.diagnostics["fallback_reason_codes"]["integrity"] == "malformed_integrity_line"
    assert json.loads(empty.context_json)["integrity_failures"] == []
    assert "integrity" not in empty.diagnostics["fallback_components"]


def test_generate_sql_records_diagnostics_without_mutating_full_schema(monkeypatch):
    state = _simple_state(retry_count=1, feedback="검증 실패를 수정")
    original_schema = state["schema_text"]
    captured: dict[str, str] = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps({
            "sql": "SELECT SUM(o.amount) AS amount_sum FROM orders o",
            "sql_type": "select",
            "source_tables": ["orders"],
            "source_column_refs": ["orders.amount"],
            "derived_columns": ["amount_sum"],
            "output_columns": ["amount_sum"],
            "reasoning": "합계 계산",
        }, ensure_ascii=False)

    monkeypatch.setattr(generate_node, "try_llm_json", fake_llm)
    result = generate_node.generate_sql(state)

    assert state["schema_text"] == original_schema
    assert "질문 reasoning sentinel" not in captured["prompt"]
    assert "검증 실패를 수정" in captured["prompt"]
    diagnostic = result["generation_context_diagnostics"][-1]
    assert diagnostic["retry_count"] == 1
    assert diagnostic["path"] == "simple"
    assert diagnostic["final_prompt_chars"] == len(captured["prompt"])
    assert diagnostic["legacy_total_chars"] > diagnostic["projected_total_chars"]


def test_invalid_context_uses_existing_generation_failure_contract(monkeypatch):
    state = _simple_state()
    state["plan"] = {**state["plan"], "required_columns": []}
    monkeypatch.setattr(generate_node, "try_llm_json", lambda _: pytest.fail("LLM을 호출하면 안 됩니다"))

    result = generate_node.generate_sql(state)

    assert result["generation_failure_reason"] == "generation_context_invalid"
    assert result["retry_hint"]["details"]["generation_reason_code"] == "generation_context_invalid"


def test_fixture_contexts_reduce_dynamic_characters_by_at_least_thirty_percent():
    results = [
        build_generation_context(_simple_state(), "simple", ""),
        build_generation_context(_comprehensive_state(), "comprehensive", ""),
    ]
    reductions = [
        1 - result.diagnostics["projected_total_chars"] / result.diagnostics["legacy_total_chars"]
        for result in results
    ]

    assert all(reduction >= 0.30 for reduction in reductions)
    assert sum(reductions) / len(reductions) >= 0.30
