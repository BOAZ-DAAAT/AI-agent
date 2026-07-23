"""SQL Agent 2단계 테이블 계획 회귀 테스트."""

from __future__ import annotations

import json
from typing import Literal

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context, finalize_plan, mart_design, plan
from DATA_Analyst_Assistant_Agent.agents.sql.nodes import generate as generate_node
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.retry import increase_retry
from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app, route_after_retry
from DATA_Analyst_Assistant_Agent.agents.sql.generation_context import build_generation_context
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.generate import generate_mart_prompt, generate_query_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.mart_design import mart_design_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.state import MartDesign, QuestionPlan
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import build_intent_contract
from DATA_Analyst_Assistant_Agent.agents.sql.validator import integrity_loader


def question_payload(**overrides):
    payload = {
        "route_kind": "simple",
        "question_type": "aggregation",
        "target_metrics": ["주문 수", "매출"],
        "analysis_entities": ["고객"],
        "dimensions": ["월"],
        "filters": ["2024년"],
        "candidate_tables": ["orders", "customers"],
        "required_aggregations": ["COUNT", "SUM"],
        "reasoning": "고객별 주문 지표를 계산할 후보다.",
    }
    payload.update(overrides)
    return payload


def base_state(**overrides):
    state = {
        "user_question": "2024년 고객별 주문 수와 매출을 보여줘",
        "schema_text": "{}",
        "integrity_text": "{}",
        "question_plan": question_payload(),
        "final_table_plan": {},
        "planning_stages": {},
        "plan": {},
        "mart_design": {},
        "sql_draft": {},
        "previous_sql_draft": {},
        "validation": {},
        "validation_findings": [],
        "retry_hint": {},
        "retry_count": 0,
        "feedback": "",
        "error": "",
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "failed_sql_component": None,
        "execution_error_info": {},
    }
    state.update(overrides)
    return state


def mart_design_payload(**overrides):
    payload = {
        "mart_name": "customer_order_category_mart",
        "target_schema": "analytics",
        "grain": "customer_unique_id × order_id × category",
        "grain_columns": ["customer_unique_id", "order_id", "category"],
        "source_tables": ["customers", "orders", "order_items"],
        "source_grains": {
            "customers": ["customer_id"],
            "orders": ["order_id"],
            "order_items": ["order_id", "order_item_id"],
        },
        "deduplication_keys": ["customer_unique_id", "order_id", "category"],
        "column_plan": [
            {
                "output_column": "customer_unique_id",
                "role": "dimension",
                "source_columns": ["customers.customer_unique_id"],
                "calculation_type": "passthrough",
                "calculation_rule": "동일 고객을 식별하는 고유 고객 키",
                "aggregation_method": "none",
                "inclusion_reason": "고객 단위 후속 계산에 필요",
            },
            {
                "output_column": "order_id",
                "role": "dimension",
                "source_columns": ["orders.order_id"],
                "calculation_type": "passthrough",
                "calculation_rule": "주문 식별자",
                "aggregation_method": "none",
                "inclusion_reason": "주문 수 계산에 필요",
            },
            {
                "output_column": "category",
                "role": "dimension",
                "source_columns": ["order_items.category"],
                "calculation_type": "passthrough",
                "calculation_rule": "상품 카테고리",
                "aggregation_method": "none",
                "inclusion_reason": "카테고리별 분석에 필요",
            },
            {
                "output_column": "item_amount_sum",
                "role": "measure",
                "source_columns": ["order_items.price"],
                "calculation_type": "derived",
                "calculation_rule": "공통 grain에 속한 상품 금액의 합계",
                "aggregation_method": "SUM",
                "inclusion_reason": "매출 후속 계산에 필요",
            },
        ],
        "metric_support": [
            {
                "metric_name": "매출",
                "calculation_grain": ["category"],
                "required_mart_columns": ["category", "item_amount_sum"],
                "downstream_calculation": "카테고리별로 item_amount_sum을 합산한다.",
            },
            {
                "metric_name": "재구매율",
                "calculation_grain": ["category"],
                "required_mart_columns": ["customer_unique_id", "order_id", "category"],
                "downstream_calculation": "고객·카테고리별 COUNT(DISTINCT order_id)가 2 이상인지 판정한 뒤 카테고리 grain에서 고객 비율을 계산한다.",
            },
        ],
        "aggregation_policy": "aggregate_to_common_grain",
        "incremental_column": None,
        "load_strategy": "full_refresh",
        "design_reasoning": "모든 목표 지표를 계산할 수 있는 공통 분석 grain이다.",
    }
    payload.update(overrides)
    return payload


def comprehensive_plan():
    return {
        **question_payload(route_kind="comprehensive", target_metrics=["매출", "재구매율"]),
        "selected_join_tables": ["customers", "orders", "order_items"],
        "required_columns": ["customers.customer_unique_id", "orders.order_id", "order_items.category", "order_items.price"],
        "business_keys": {"customers": "customers.customer_unique_id", "orders": "orders.order_id"},
    }


def test_question_plan_contract_has_exactly_nine_fields():
    assert list(QuestionPlan.model_fields) == [
        "route_kind",
        "question_type",
        "target_metrics",
        "analysis_entities",
        "dimensions",
        "filters",
        "candidate_tables",
        "required_aggregations",
        "reasoning",
    ]


def test_mart_design_preserves_common_grain_contract_and_derives_legacy_lists():
    payload = mart_design_payload(
        key_columns=["wrong_key"],
        dimension_columns=["wrong_dimension"],
        measure_columns=["wrong_measure"],
    )

    dumped = MartDesign(**payload).model_dump()

    assert dumped["grain"] == "customer_unique_id × order_id × category"
    assert dumped["source_grains"]["order_items"] == ["order_id", "order_item_id"]
    assert dumped["deduplication_keys"] == dumped["grain_columns"]
    assert dumped["key_columns"] == dumped["grain_columns"]
    assert dumped["dimension_columns"] == ["customer_unique_id", "order_id", "category"]
    assert dumped["measure_columns"] == ["item_amount_sum"]
    assert dumped["metric_support"][1]["metric_name"] == "재구매율"


def test_mart_design_repairs_empty_source_columns_from_calculation_rule():
    payload = mart_design_payload()
    payload["column_plan"].append(
        {
            "output_column": "purchase_segment",
            "role": "attribute",
            "source_columns": [],
            "calculation_type": "derived",
            "calculation_rule": "item_amount_sum and category 기준으로 segment를 만든다.",
            "aggregation_method": "none",
            "inclusion_reason": "분석용 파생 세그먼트",
        }
    )

    design = mart_design.validate_mart_design_state(payload, comprehensive_plan())

    repaired = next(item for item in design.column_plan if item.output_column == "purchase_segment")
    assert repaired.source_columns == ["category", "item_amount_sum"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.pop("grain_columns"),
        lambda payload: payload["column_plan"].append(dict(payload["column_plan"][0])),
        lambda payload: payload.update(deduplication_keys=["order_id"]),
        lambda payload: payload.update(aggregation_policy="invalid_policy"),
        lambda payload: payload["metric_support"][0].update(required_mart_columns=["unknown_column"]),
        lambda payload: payload["metric_support"].pop(),
    ],
)
def test_mart_design_node_rejects_invalid_contract_as_retryable(monkeypatch, mutator):
    payload = mart_design_payload()
    mutator(payload)
    monkeypatch.setattr(mart_design, "try_llm_json", lambda _: json.dumps(payload, ensure_ascii=False))
    state = base_state(plan=comprehensive_plan())

    result = mart_design.design_mart(state)

    assert result["retry_hint"]["reason_code"] == "sql_mart_design_failed"
    assert result["retry_hint"]["retryable"] is True


def test_preserve_common_grain_rejects_deduplication_or_aggregation():
    payload = mart_design_payload(aggregation_policy="preserve_common_grain")
    with pytest.raises(ValueError, match="모든 aggregation_method가 none"):
        MartDesign(**payload)


def test_planning_catalog_only_contains_table_name_and_description():
    catalog = json.loads(integrity_loader.load_schema_catalog_text())
    tables = catalog.get("tables", catalog)
    assert tables
    assert all(set(table) == {"description"} for table in tables.values())
    assert "sample_data" not in json.dumps(catalog)


def test_plan_question_normalizes_new_fields_and_ignores_extra(monkeypatch):
    payload = question_payload(extra_field="무시", selected_join_tables=["orders"])
    monkeypatch.setattr(plan, "try_llm_json", lambda _: json.dumps(payload, ensure_ascii=False))

    result = plan.plan_question(base_state())

    assert result["question_plan"] == question_payload()
    assert result["planning_stages"]["question_plan"] == question_payload()
    assert result["plan"] == {}
    assert "extra_field" not in result["question_plan"]
    assert "selected_join_tables" not in result["question_plan"]


def test_plan_question_promotes_simple_when_required_derivation_exists(monkeypatch):
    monkeypatch.setattr(
        plan,
        "try_llm_json",
        lambda _: json.dumps(question_payload(route_kind="simple"), ensure_ascii=False),
    )

    result = plan.plan_question(
        base_state(
            required_derivations=[
                {"name": "배송 지연 일수", "preferred_name": "delivery_delay_days"}
            ]
        )
    )

    assert result["question_plan"]["route_kind"] == "comprehensive"


def test_mart_prompt_separates_derivations_and_heuristics():
    prompt_text = mart_design_prompt(
        base_state(
            plan=comprehensive_plan(),
            required_derivations=[
                {
                    "name": "배송 지연 일수",
                    "preferred_name": "delivery_delay_days",
                    "grain": "order_id",
                    "source_columns": ["delivered_at", "estimated_at"],
                    "definition": "두 날짜의 일수 차이",
                }
            ],
            analysis_heuristics=[
                {"name": "이상치 민감도 기록", "default_policy": "하류에서 비교"}
            ],
        )
    )

    assert '"preferred_name": "delivery_delay_days"' in prompt_text
    assert '"name": "이상치 민감도 기록"' in prompt_text
    assert "unimplemented_derivations" in prompt_text
    assert "명시적으로 요구하지 않은 항목은 SQL 컬럼이나 필터로 구현하지 않고" in prompt_text


def test_mart_design_requires_each_derivation_in_exactly_one_location():
    derivations = [
        {"name": "배송 지연 일수", "preferred_name": "delivery_delay_days"}
    ]
    implemented = mart_design_payload()
    implemented["column_plan"].append(
        {
            "output_column": "delivery_delay_days",
            "role": "measure",
            "source_columns": ["orders.delivered_at", "orders.estimated_at"],
            "calculation_type": "derived",
            "calculation_rule": "실제 배송일과 예상 배송일의 일수 차이",
            "aggregation_method": "none",
            "inclusion_reason": "필수 파생계약",
        }
    )
    design = mart_design.validate_mart_design_state(
        implemented,
        comprehensive_plan(),
        derivations,
    )
    assert design.column_plan[-1].output_column == "delivery_delay_days"

    unimplemented = mart_design_payload(
        unimplemented_derivations=[
            {
                "name": "delivery_delay_days",
                "reason": "예상 배송일 컬럼이 없음",
                "required_columns": ["orders.estimated_at"],
            }
        ]
    )
    design = mart_design.validate_mart_design_state(
        unimplemented,
        comprehensive_plan(),
        derivations,
    )
    assert design.unimplemented_derivations[0].name == "delivery_delay_days"

    with pytest.raises(ValueError, match="missing=.*delivery_delay_days"):
        mart_design.validate_mart_design_state(
            mart_design_payload(),
            comprehensive_plan(),
            derivations,
        )

    duplicated = dict(implemented)
    duplicated["unimplemented_derivations"] = [
        {
            "name": "delivery_delay_days",
            "reason": "중복 선언",
            "required_columns": [],
        }
    ]
    with pytest.raises(ValueError, match="duplicated=.*delivery_delay_days"):
        mart_design.validate_mart_design_state(
            duplicated,
            comprehensive_plan(),
            derivations,
        )


def test_mart_design_node_reports_missing_derivation_as_invalid_payload(monkeypatch):
    monkeypatch.setattr(
        mart_design,
        "try_llm_json",
        lambda _: json.dumps(mart_design_payload(), ensure_ascii=False),
    )
    result = mart_design.design_mart(
        base_state(
            plan=comprehensive_plan(),
            required_derivations=[
                {"name": "배송 지연 일수", "preferred_name": "delivery_delay_days"}
            ],
        )
    )

    assert result["validation_findings"][0]["code"] == "invalid_mart_design_payload"
    assert "delivery_delay_days" in result["feedback"]


def test_mart_redesign_retry_preserves_structured_input_contracts():
    state = base_state(
        required_derivations=[
            {"name": "배송 지연 일수", "preferred_name": "delivery_delay_days"}
        ],
        analysis_heuristics=[{"name": "이상치 민감도 기록"}],
        validation={
            "result": "invalid",
            "feedback": "파생계약 누락",
            "retry_hint": {
                "reason_code": "sql_mart_design_failed",
                "retryable": True,
            },
        },
    )

    merged = {**state, **increase_retry(state)}

    assert merged["required_derivations"] == state["required_derivations"]
    assert merged["analysis_heuristics"] == state["analysis_heuristics"]


def test_scoped_schema_contains_all_column_details_and_keys_without_samples():
    schema = json.loads(integrity_loader.load_scoped_schema_text(["customers", "orders"]))
    tables = schema.get("tables", schema)
    assert set(tables) == {"customers", "orders"}
    assert tables["customers"]["columns"]
    assert set(tables["customers"]["columns"][0]) == {"name", "type", "nullable", "description"}
    assert "primary_key" in tables["orders"]
    assert "foreign_keys" in tables["orders"]
    assert "sample_data" not in json.dumps(schema)


@pytest.mark.parametrize(
    ("candidates", "scoped", "reason"),
    [([], "", "no_tables"), (["missing"], "", "no_valid_tables")],
)
def test_schema_refresh_fails_retryably_without_valid_candidates(monkeypatch, candidates, scoped, reason):
    monkeypatch.setattr(context, "load_scoped_schema_text", lambda _: scoped)
    state = base_state(question_plan=question_payload(candidate_tables=candidates))

    result = context.refresh_schema_context(state)

    assert result["schema_refresh"]["status"] == "failed"
    assert result["schema_refresh"]["reason"] == reason
    assert result["retry_hint"]["reason_code"] == "sql_plan_failed"
    assert result["retry_hint"]["retryable"] is True


def test_schema_refresh_continues_with_valid_subset(monkeypatch):
    scoped = json.dumps({"orders": {"columns": [{"name": "order_id"}]}})
    monkeypatch.setattr(context, "load_scoped_schema_text", lambda _: scoped)

    result = context.refresh_schema_context(base_state())

    assert result["schema_refresh"] == {
        "status": "refreshed",
        "candidate_tables": ["orders", "customers"],
        "applied_tables": ["orders"],
        "reason": "",
    }


def test_finalize_table_plan_merges_reasoning_and_plural_metrics(monkeypatch):
    table_payload = {
        "selected_join_tables": ["orders", "customers"],
        "required_columns": ["orders.customer_id", "customers.customer_unique_id"],
        "business_keys": {"customers": "customers.customer_unique_id"},
        "reasoning": "실고객 키로 주문을 묶는다.",
        "ignored": "제거",
    }
    monkeypatch.setattr(finalize_plan, "try_llm_json", lambda _: json.dumps(table_payload, ensure_ascii=False))

    result = finalize_plan.finalize_table_plan(base_state())
    merged = result["plan"]

    assert merged["question_plan_reasoning"] == question_payload()["reasoning"]
    assert merged["table_plan_reasoning"] == table_payload["reasoning"]
    assert merged["business_keys"] == {"customers": "customers.customer_unique_id"}
    assert merged["validation_contract"]["target_metrics"] == ["주문 수", "매출"]
    assert "target_metric" not in merged["validation_contract"]
    assert result["planning_stages"]["final_table_plan"].get("ignored") is None


@pytest.mark.parametrize(
    ("response", "reason_code"),
    [
        (None, "llm_empty_response"),
        ("{bad", "llm_json_parse_failed"),
        ("[]", "llm_json_not_object"),
        (json.dumps({"selected_join_tables": ["orders"]}), "invalid_final_table_plan"),
        (json.dumps({"selected_join_tables": [], "required_columns": ["orders.order_id"], "business_keys": {}, "reasoning": "근거"}), "invalid_final_table_plan"),
        (json.dumps({"selected_join_tables": ["orders"], "required_columns": [], "business_keys": {}, "reasoning": "근거"}), "invalid_final_table_plan"),
        (json.dumps({"selected_join_tables": ["orders"], "required_columns": ["orders.order_id"], "business_keys": {}, "reasoning": " "}), "invalid_final_table_plan"),
    ],
)
def test_finalize_table_plan_rejects_invalid_responses(monkeypatch, response, reason_code):
    monkeypatch.setattr(finalize_plan, "try_llm_json", lambda _: response)

    result = finalize_plan.finalize_table_plan(base_state())

    assert result["retry_hint"]["reason_code"] == "sql_plan_failed"
    assert result["retry_hint"]["details"]["plan_reason_code"] == reason_code


def test_replan_retry_clears_stale_two_stage_and_sql_state():
    state = base_state(
        final_table_plan={"old": True},
        planning_stages={"old": True},
        plan={"old": True},
        mart_design={"old": True},
        sql_draft={"sql": "old"},
        retry_hint={"reason_code": "sql_plan_failed", "retryable": True},
    )

    result = increase_retry(state)

    assert result["retry_count"] == 1
    assert result["question_plan"] == {}
    assert result["final_table_plan"] == {}
    assert result["mart_design"] == {}
    assert result["sql_draft"] == {}


def test_mart_design_retry_keeps_planning_and_clears_only_downstream_state():
    state = base_state(
        plan=comprehensive_plan(),
        final_table_plan={"kept": True},
        planning_stages={"kept": True},
        mart_design={"old": True},
        sql_draft={"sql": "old"},
        retry_hint={"reason_code": "sql_mart_design_failed", "retryable": True},
    )

    result = increase_retry(state)

    assert result["mart_design"] == {}
    assert result["sql_draft"] == {}
    assert "plan" not in result
    assert "final_table_plan" not in result
    assert route_after_retry({**state, **result}) == "redesign"


@pytest.mark.parametrize("reason_code", ["sql_generation_failed", "invalid_join_plan"])
def test_initial_generation_errors_retry_generation(reason_code):
    state = base_state(
        plan=comprehensive_plan(),
        mart_design=mart_design_payload(),
        sql_draft={"sql": "old"},
        retry_hint={"reason_code": reason_code, "retryable": True},
    )

    result = increase_retry(state)

    assert result["sql_draft"] == {}
    assert "mart_design" not in result
    assert "plan" not in result
    assert route_after_retry({**state, **result}) == "regenerate"


@pytest.mark.parametrize("reason_code", ["result_shape_mismatch", "intent_mismatch", "mysql_dialect_error", "execution_error"])
def test_local_sql_errors_retry_repair(reason_code):
    state = base_state(
        plan=comprehensive_plan(),
        mart_design=mart_design_payload(),
        sql_draft={"sql": "SELECT broken", "sql_type": "select"},
        validation_findings=[{"category": reason_code, "detail": "정확한 오류"}],
        retry_hint={"reason_code": reason_code, "retryable": True},
        error="정확한 오류",
        failed_statement_sql="SELECT broken",
    )

    result = increase_retry(state)

    assert result["sql_draft"] == {}
    assert result["previous_sql_draft"]["sql"] == "SELECT broken"
    assert "validation_findings" not in result
    assert "error" not in result
    assert route_after_retry({**state, **result}) == "repair"


@pytest.mark.parametrize("reason_code", ["missing_table", "missing_column"])
def test_missing_identifier_is_not_retryable_route(reason_code):
    from DATA_Analyst_Assistant_Agent.agents.sql.graph import route_after_validation

    state = base_state(
        validation={"result": "invalid"},
        retry_hint={"reason_code": reason_code, "retryable": False},
    )
    assert route_after_validation(state) == "finalize"


def test_sql_regeneration_retry_keeps_planning_state():
    result = increase_retry(base_state(retry_hint={"reason_code": "mysql_dialect_error", "retryable": True}))
    assert "question_plan" not in result
    assert "plan" not in result


def test_downstream_prompts_prioritize_required_columns_and_business_keys():
    state = base_state(
        plan={
            **question_payload(),
            "selected_join_tables": ["customers"],
            "required_columns": ["customers.customer_unique_id"],
            "business_keys": {"customers": "customers.customer_unique_id"},
        }
    )
    context_result = build_generation_context(state, "simple", "")
    query_prompt = generate_query_prompt(context_result.context)
    mart_prompt = mart_design_prompt(state)

    for prompt_text in (query_prompt, mart_prompt):
        assert "required_columns" in prompt_text
        assert "business_keys" in prompt_text
        assert "customers.customer_unique_id" in prompt_text


def test_mart_design_prompt_includes_analysis_required_ratio_policy():
    prompt_text = mart_design_prompt(base_state(plan=comprehensive_plan()))

    for expected in (
        "분석 필수 파생변수",
        "연속형 비율",
        "분자·분모·연산 순서",
        "0이거나 NULL",
        "required_mart_columns에 해당 output_column",
        "마트 컬럼을 직접 사용",
    ):
        assert expected in prompt_text


def test_mart_generation_prompt_contains_design_and_postcheck_semantics():
    state = base_state(plan=comprehensive_plan(), mart_design=mart_design_payload())

    context_result = build_generation_context(state, "comprehensive", "")
    prompt_text = generate_mart_prompt(context_result.context)

    for expected in (
        "customer_unique_id × order_id × category",
        "source_grains",
        "deduplication_keys",
        "column_plan",
        "aggregation_method",
        "metric_support",
        "분석 필수 파생변수",
        "분자·분모·연산 순서",
        "0/NULL 처리 규칙",
        "column_plan에 없는 최종 표시용",
        "임의 컬럼·집계·필터",
        "row_count",
        "duplicate_grain_count",
        "null_grain_count",
        "한 행 SELECT",
    ):
        assert expected in prompt_text


def test_comprehensive_generation_normalizes_business_grain_to_mart_design(monkeypatch):
    state = base_state(plan=comprehensive_plan(), mart_design=mart_design_payload())
    captured: dict[str, object] = {}

    def fake_structured_output(_prompt, schema):
        captured["schema"] = schema
        return {
            "sql": "CREATE TABLE analytics.customer_order_category_mart AS SELECT 1 AS customer_unique_id, 1 AS order_id, 'x' AS category, 10 AS item_amount_sum",
            "sql_type": "create_table_as",
            "target_table": "customer_order_category_mart",
            "source_tables": ["customers", "orders", "order_items"],
            "source_column_refs": ["customers.customer_unique_id", "orders.order_id", "order_items.category", "order_items.price"],
            "derived_columns": ["item_amount_sum"],
            "output_columns": ["customer_unique_id", "order_id", "category", "item_amount_sum"],
            "business_grain": "LLM이 임의로 바꾼 grain",
            "precheck_sql": "SELECT COUNT(*) FROM orders",
            "postcheck_sql": "SELECT 1 AS row_count, 0 AS duplicate_grain_count, 0 AS null_grain_count FROM customer_order_category_mart",
            "reasoning": "계약 구현",
        }

    monkeypatch.setattr(generate_node, "invoke_llm_structured", fake_structured_output)

    result = generate_node.generate_sql(state)

    assert result["sql_draft"]["business_grain"] == mart_design_payload()["grain"]
    assert result["sql_draft"]["target_table"] == "analytics.customer_order_category_mart"
    assert getattr(captured["schema"], "model_fields")["sql_type"].annotation == Literal["create_table_as"]


def test_sql_type_structured_output_accepts_case_and_ctas_aliases():
    simple = generate_node._SimpleSQLDraft.model_validate(
        {
            "sql": "SELECT 1",
            "sql_type": "SELECT",
            "reasoning": "대문자 select 허용",
        }
    )
    comprehensive = generate_node._ComprehensiveSQLDraft.model_validate(
        {
            "sql": "CREATE TABLE analytics.example AS SELECT 1",
            "sql_type": "CREATE_TABLE_AS_SELECT",
            "target_table": "analytics.example",
            "reasoning": "CTAS 별칭 허용",
        }
    )

    assert simple.sql_type == "select"
    assert comprehensive.sql_type == "create_table_as"


def test_legacy_mart_design_is_discarded_before_sql_generation():
    state = base_state(
        plan=comprehensive_plan(),
        mart_design={"mart_name": "legacy", "grain": "old"},
        sql_draft={"sql": "old"},
    )

    result = generate_node.generate_sql(state)

    assert result["mart_design"] == {}
    assert result["retry_hint"]["reason_code"] == "sql_mart_design_failed"


def test_build_intent_contract_uses_plural_metrics_only():
    plan_payload = {
        **question_payload(),
        "selected_join_tables": ["orders"],
        "required_columns": ["orders.order_id"],
        "validation_contract": {},
    }
    contract = build_intent_contract(plan_payload)
    assert contract["target_metrics"] == ["주문 수", "매출"]
    assert "target_metric" not in contract


def test_graph_wires_two_stage_order_for_both_routes():
    graph = build_app().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("plan_question", "refresh_integrity_context") in edges
    assert ("refresh_integrity_context", "refresh_schema_context") in edges
    assert ("refresh_schema_context", "finalize_table_plan") in edges
    assert ("finalize_table_plan", "generate_sql") in edges
    assert ("finalize_table_plan", "design_mart") in edges
    assert ("design_mart", "generate_sql") in edges
    assert ("repair_sql", "prevalidate_sql") in edges


def graph_state(**overrides):
    state = {
        "user_question": "주문 수를 보여줘",
        "required_db_schema": "",
        "clarification_request": "",
        "planner_selection_reason": "",
        "schema_text": "",
        "integrity_text": "",
        "integrity_dataset_name": "default",
        "integrity_preplan": {},
        "integrity_refresh": {},
        "schema_refresh": {},
        "question_plan": {},
        "final_table_plan": {},
        "planning_stages": {},
        "plan": {},
        "mart_design": {},
        "sql_draft": {},
        "previous_sql_draft": {},
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
        "feedback": "",
        "error": "",
        "generation_source": "llm",
        "generation_failure_reason": "",
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "failed_sql_component": None,
        "execution_error_info": {},
        "final_answer": "",
    }
    state.update(overrides)
    return state


def test_final_plan_failure_restarts_from_question_planning(monkeypatch):
    from DATA_Analyst_Assistant_Agent.agents.sql import planner_support
    from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute

    calls = {"question": 0, "final": 0, "sql": 0}

    class Response:
        def __init__(self, content):
            self.content = content

    class LLM:
        def with_structured_output(self, schema_model):
            parent = self

            class Structured:
                def invoke(self, prompt_text):
                    return schema_model.model_validate_json(parent.invoke(prompt_text).content)

            return Structured()

        def invoke(self, prompt_text):
            if "MySQL 기반 SQL/데이터마트 planner" in prompt_text:
                calls["question"] += 1
                return Response(json.dumps(question_payload(
                    target_metrics=["주문 수"],
                    analysis_entities=["주문"],
                    dimensions=[],
                    filters=[],
                    candidate_tables=["orders"],
                    required_aggregations=["COUNT"],
                ), ensure_ascii=False))
            if "MySQL 물리 테이블 계획자" in prompt_text:
                calls["final"] += 1
                if calls["final"] == 1:
                    return Response("{}")
                return Response(json.dumps({
                    "selected_join_tables": ["orders"],
                    "required_columns": ["orders.order_id"],
                    "business_keys": {"orders": "orders.order_id"},
                    "reasoning": "주문 식별자로 집계한다.",
                }, ensure_ascii=False))
            if "MySQL 조회 SQL" in prompt_text:
                calls["sql"] += 1
                return Response(json.dumps({
                    "sql": "SELECT COUNT(*) AS order_count FROM orders;",
                    "sql_type": "select",
                    "source_tables": ["orders"],
                    "source_column_refs": ["orders.order_id"],
                    "derived_columns": ["order_count"],
                    "output_columns": ["order_count"],
                    "reasoning": "주문 수 집계",
                }, ensure_ascii=False))
            raise AssertionError(prompt_text[:100])

    schema = json.dumps({"orders": {"description": "주문", "columns": [{"name": "order_id", "type": "VARCHAR", "nullable": False, "description": "주문 키"}], "primary_key": ["order_id"], "foreign_keys": []}}, ensure_ascii=False)
    monkeypatch.setattr(context, "load_schema_catalog_text", lambda: json.dumps({"orders": {"description": "주문"}}, ensure_ascii=False))
    monkeypatch.setattr(context, "load_scoped_schema_text", lambda _: schema)
    monkeypatch.setattr(planner_support, "get_llm", lambda: LLM())
    monkeypatch.setattr(execute, "can_use_live_db", lambda: True)
    monkeypatch.setattr(execute, "run_sql_fetchall", lambda _: [(1,)])

    result = build_app().invoke(graph_state())

    assert calls == {"question": 2, "final": 2, "sql": 1}
    assert result["retry_count"] == 1
    assert result["validation"]["result"] == "valid"
    assert result["planning_stages"]["question_plan"]["candidate_tables"] == ["orders"]


def test_mart_design_failure_retries_only_design_stage(monkeypatch):
    from DATA_Analyst_Assistant_Agent.agents.sql import planner_support
    from DATA_Analyst_Assistant_Agent.agents.sql.nodes import execute

    calls = {"question": 0, "final": 0, "design": 0, "sql": 0}

    class Response:
        def __init__(self, content):
            self.content = content

    class LLM:
        def with_structured_output(self, schema_model):
            parent = self

            class Structured:
                def invoke(self, prompt_text):
                    return schema_model.model_validate_json(parent.invoke(prompt_text).content)

            return Structured()

        def invoke(self, prompt_text):
            if "MySQL 기반 SQL/데이터마트 planner" in prompt_text:
                calls["question"] += 1
                return Response(json.dumps(question_payload(
                    route_kind="comprehensive",
                    target_metrics=["매출", "재구매율"],
                    candidate_tables=["customers", "orders", "order_items"],
                    required_aggregations=["SUM"],
                ), ensure_ascii=False))
            if "MySQL 물리 테이블 계획자" in prompt_text:
                calls["final"] += 1
                return Response(json.dumps({
                    "selected_join_tables": ["customers", "orders", "order_items"],
                    "required_columns": ["customers.customer_unique_id", "orders.order_id", "order_items.category", "order_items.price"],
                    "business_keys": {"customers": "customers.customer_unique_id", "orders": "orders.order_id"},
                    "reasoning": "고객과 주문 키로 공통 grain을 만든다.",
                }, ensure_ascii=False))
            if "분석용 데이터마트 설계자" in prompt_text:
                calls["design"] += 1
                if calls["design"] == 1:
                    return Response("{}")
                return Response(json.dumps(mart_design_payload(), ensure_ascii=False))
            if "MySQL 재사용 데이터마트 SQL" in prompt_text:
                calls["sql"] += 1
                return Response(json.dumps({
                    "sql": "CREATE TABLE analytics.customer_order_category_mart AS SELECT c.customer_unique_id, o.order_id, oi.category, SUM(oi.price) AS item_amount_sum FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_unique_id, o.order_id, oi.category",
                    "sql_type": "create_table_as",
                    "target_table": "analytics.customer_order_category_mart",
                    "source_tables": ["customers", "orders", "order_items"],
                    "source_column_refs": ["customers.customer_unique_id", "customers.customer_id", "orders.customer_id", "orders.order_id", "order_items.order_id", "order_items.category", "order_items.price"],
                    "derived_columns": ["item_amount_sum"],
                    "output_columns": ["customer_unique_id", "order_id", "category", "item_amount_sum"],
                    "business_grain": "잘못된 grain",
                    "precheck_sql": None,
                    "postcheck_sql": "SELECT COUNT(*) AS row_count, 0 AS duplicate_grain_count, 0 AS null_grain_count FROM analytics.customer_order_category_mart",
                    "reasoning": "공통 분석 grain으로 집계한다.",
                }, ensure_ascii=False))
            raise AssertionError(prompt_text[:100])

    schema = json.dumps({
        "customers": {"description": "고객", "columns": [{"name": "customer_id"}, {"name": "customer_unique_id"}]},
        "orders": {"description": "주문", "columns": [{"name": "order_id"}, {"name": "customer_id"}]},
        "order_items": {"description": "상품", "columns": [{"name": "order_id"}, {"name": "category"}, {"name": "price"}]},
    }, ensure_ascii=False)
    monkeypatch.setattr(context, "load_schema_catalog_text", lambda: json.dumps({name: {"description": value["description"]} for name, value in json.loads(schema).items()}, ensure_ascii=False))
    monkeypatch.setattr(context, "load_scoped_schema_text", lambda _: schema)
    monkeypatch.setattr(planner_support, "get_llm", lambda: LLM())
    monkeypatch.setattr(execute, "can_use_live_db", lambda: False)

    result = build_app().invoke(graph_state())

    assert calls == {"question": 1, "final": 1, "design": 2, "sql": 1}
    assert result["retry_count"] == 1
    assert result["validation"]["result"] == "valid"
    assert result["sql_draft"]["business_grain"] == mart_design_payload()["grain"]


def test_sql_generation_retry_does_not_repeat_two_stage_planning(monkeypatch):
    state = graph_state(
        retry_count=0,
        retry_hint={"reason_code": "mysql_dialect_error", "retryable": True},
        question_plan=question_payload(candidate_tables=["orders"]),
        final_table_plan={"selected_join_tables": ["orders"], "required_columns": ["orders.order_id"], "business_keys": {}, "reasoning": "근거"},
        planning_stages={"kept": True},
        plan={"route_kind": "simple"},
    )
    update = increase_retry(state)
    assert update["retry_count"] == 1
    assert "planning_stages" not in update


def test_repurchase_customer_key_reaches_sql_generation_prompt(monkeypatch):
    question_plan = question_payload(
        target_metrics=["재구매 고객 수"],
        analysis_entities=["고객"],
        dimensions=[],
        filters=[],
        candidate_tables=["customers", "orders"],
        required_aggregations=["COUNT"],
        reasoning="실고객 기준 재구매를 분석한다.",
    )
    state = base_state(question_plan=question_plan)
    monkeypatch.setattr(finalize_plan, "try_llm_json", lambda _: json.dumps({
        "selected_join_tables": ["customers", "orders"],
        "required_columns": ["customers.customer_unique_id", "customers.customer_id", "orders.customer_id"],
        "business_keys": {"customers": "customers.customer_unique_id"},
        "reasoning": "customer_unique_id로 동일 고객을 식별한다.",
    }, ensure_ascii=False))
    final_update = finalize_plan.finalize_table_plan(state)
    state.update(final_update)
    prompts_seen: list[str] = []
    monkeypatch.setattr(
        generate_node,
        "invoke_llm_structured",
        lambda prompt_text, _schema: prompts_seen.append(prompt_text) or {
            "sql": "SELECT COUNT(DISTINCT c.customer_unique_id) FROM customers c JOIN orders o ON o.customer_id = c.customer_id;",
            "sql_type": "select",
            "source_tables": ["customers", "orders"],
            "source_column_refs": ["customers.customer_unique_id", "customers.customer_id", "orders.customer_id"],
            "derived_columns": [],
            "output_columns": ["repurchase_customers"],
            "reasoning": "실고객 키 사용",
        },
    )

    generate_node.generate_sql(state)

    assert prompts_seen
    assert '"customers"' in prompts_seen[0]
    assert "customers.customer_unique_id" in prompts_seen[0]
