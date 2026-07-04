from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.nodes.generate import generate_sql


def _state() -> dict:
    return {
        "user_question": "2017년 6월의 거래를 날짜별로 보여주고 추이를 분석해줘",
        "required_db_schema": "",
        "clarification_request": "",
        "planner_selection_reason": "",
        "schema_text": "",
        "integrity_text": "",
        "plan": {
            "task_type": "query_answer",
            "route_kind": "simple",
        },
        "mart_design": {},
        "sql_draft": {},
        "sql_result": None,
        "row_count": 0,
        "precheck_result": None,
        "postcheck_result": None,
        "mart_quality_result": {},
        "validation": {},
        "validation_findings": [],
        "retry_hint": {"reason_code": "missing_table"},
        "validation_summary": {},
        "retry_count": 1,
        "max_retries": 2,
        "feedback": "이전 쿼리가 필요한 테이블을 못 찾았습니다.",
        "error": "",
        "final_answer": "",
    }


def test_generate_sql_retries_llm_instead_of_forcing_deterministic_fallback(monkeypatch) -> None:
    llm_sql = {
        "sql": "SELECT DATE(order_purchase_timestamp) AS order_date, COUNT(*) AS order_count FROM orders GROUP BY 1;",
        "sql_type": "select",
        "source_tables": ["orders"],
        "columns_used": ["order_purchase_timestamp", "order_id"],
        "reasoning": "retry llm result",
    }

    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.sql.nodes.generate.try_llm_json",
        lambda prompt: '{"sql":"SELECT DATE(order_purchase_timestamp) AS order_date, COUNT(*) AS order_count FROM orders GROUP BY 1;","sql_type":"select","source_tables":["orders"],"columns_used":["order_purchase_timestamp","order_id"],"reasoning":"retry llm result"}',
    )

    result = generate_sql(_state())

    assert result["sql_draft"]["sql"] == llm_sql["sql"]
    assert result["sql_draft"]["source_tables"] == ["orders"]
