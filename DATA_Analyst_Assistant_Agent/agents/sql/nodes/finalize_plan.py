"""finalize_table_plan 노드: 후보 상세 메타데이터로 물리 테이블 계획을 확정한다."""

from __future__ import annotations

import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.plan import _plan_failure
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import try_llm_json
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, FinalTablePlan


def _merge_plan(question_plan: dict[str, Any], table_plan: dict[str, Any]) -> dict[str, Any]:
    route_kind = str(question_plan["route_kind"])
    validation_contract = {
        "required_tables": list(table_plan["selected_join_tables"]),
        "required_columns": list(table_plan["required_columns"]),
        "target_metrics": list(question_plan["target_metrics"]),
        "dimensions": list(question_plan["dimensions"]),
        "required_aggregations": list(question_plan["required_aggregations"]),
        "expected_result_shape": "datamart_creation" if route_kind == "comprehensive" else "table_preview",
        "expected_aliases": [],
        "target_table": None,
        "mart_policy": "prefer_row_preserving" if route_kind == "comprehensive" else None,
    }
    return {
        **{key: value for key, value in question_plan.items() if key != "reasoning"},
        **{key: value for key, value in table_plan.items() if key != "reasoning"},
        "question_plan_reasoning": question_plan["reasoning"],
        "table_plan_reasoning": table_plan["reasoning"],
        "validation_contract": validation_contract,
    }


def finalize_table_plan(state: AgentState) -> dict[str, Any]:
    response = try_llm_json(prompts.finalize_table_plan_prompt(state))
    if not response:
        return _plan_failure(
            reason_code="llm_empty_response",
            detail="LLM이 최종 테이블 계획을 반환하지 않았습니다.",
            retryable=True,
        )

    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _plan_failure(
            reason_code="llm_json_parse_failed",
            detail="LLM 최종 테이블 계획 응답을 JSON으로 파싱하지 못했습니다.",
            retryable=True,
        )
    if not isinstance(parsed, dict):
        return _plan_failure(
            reason_code="llm_json_not_object",
            detail="LLM 최종 테이블 계획 응답이 JSON object가 아닙니다.",
            retryable=True,
        )

    try:
        table_plan = FinalTablePlan(**parsed).model_dump()
    except Exception as exc:
        return _plan_failure(
            reason_code="invalid_final_table_plan",
            detail=f"LLM 최종 테이블 계획이 필수 구조를 만족하지 못했습니다: {exc}",
            retryable=True,
        )

    question_plan = dict(state.get("question_plan") or {})
    merged_plan = _merge_plan(question_plan, table_plan)
    return {
        "final_table_plan": table_plan,
        "planning_stages": {
            "question_plan": question_plan,
            "final_table_plan": table_plan,
        },
        "plan": merged_plan,
    }
