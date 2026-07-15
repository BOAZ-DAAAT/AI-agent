"""SQL Agent 2단계 테이블 계획 회귀 테스트."""

from __future__ import annotations

import json

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql.nodes import context, finalize_plan, plan
from DATA_Analyst_Assistant_Agent.agents.sql.nodes import generate as generate_node
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.retry import increase_retry
from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.generate import generate_query_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.prompts.mart_design import mart_design_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.state import QuestionPlan
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
        "validation": {},
        "validation_findings": [],
        "retry_hint": {},
        "retry_count": 0,
        "max_retries": 2,
        "feedback": "",
        "error": "",
        "failed_statement_index": None,
        "failed_statement_sql": "",
    }
    state.update(overrides)
    return state


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


def test_sql_regeneration_retry_keeps_planning_state():
    result = increase_retry(base_state(retry_hint={"reason_code": "mysql_dialect_error", "retryable": True}))
    assert "question_plan" not in result
    assert "plan" not in result


def test_downstream_prompts_prioritize_required_columns_and_business_keys():
    state = base_state(
        plan={
            **question_payload(),
            "required_columns": ["customers.customer_unique_id"],
            "business_keys": {"customers": "customers.customer_unique_id"},
        }
    )
    query_prompt = generate_query_prompt(state, "")
    mart_prompt = mart_design_prompt(state)

    for prompt_text in (query_prompt, mart_prompt):
        assert "required_columns" in prompt_text
        assert "business_keys" in prompt_text
        assert "customers.customer_unique_id" in prompt_text


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
            if "MySQL SQL 작성기다. 조회 SQL" in prompt_text:
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
        "try_llm_json",
        lambda prompt_text: prompts_seen.append(prompt_text) or json.dumps({
            "sql": "SELECT COUNT(DISTINCT c.customer_unique_id) FROM customers c JOIN orders o ON o.customer_id = c.customer_id;",
            "sql_type": "select",
            "source_tables": ["customers", "orders"],
            "source_column_refs": ["customers.customer_unique_id", "customers.customer_id", "orders.customer_id"],
            "derived_columns": [],
            "output_columns": ["repurchase_customers"],
            "reasoning": "실고객 키 사용",
        }),
    )

    generate_node.generate_sql(state)

    assert prompts_seen
    assert '"customers"' in prompts_seen[0]
    assert "customers.customer_unique_id" in prompts_seen[0]
