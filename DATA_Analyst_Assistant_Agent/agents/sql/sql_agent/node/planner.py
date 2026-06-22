from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.prompt.templates import mart_design_prompt, planner_prompt
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.planner_support import (
    default_validation_contract,
    default_mart_design,
    default_plan_from_state,
    safe_json_parse,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import ALLOWED_MART_SCHEMA, AgentState


def plan_question(state: AgentState):
    fallback = default_plan_from_state(state)
    response = try_llm_json(planner_prompt(state))
    parsed = safe_json_parse(response, fallback) if response else fallback
    parsed["original_question"] = state["user_question"]
    parsed.setdefault("route_kind", fallback["route_kind"])
    parsed.setdefault("selected_join_tables", fallback["selected_join_tables"])
    parsed.setdefault("candidate_tables", fallback["candidate_tables"])
    parsed.setdefault("relevant_tables", parsed.get("selected_join_tables", []))
    parsed.setdefault("reasoning", fallback["reasoning"])
    if parsed.get("route_kind") not in {"simple", "comprehensive"}:
        parsed["route_kind"] = fallback["route_kind"]
    if parsed["route_kind"] == "comprehensive":
        parsed["task_type"] = "data_mart_build"
        parsed["requested_output"] = "create_table"
    else:
        parsed["task_type"] = "query_answer"
        parsed["requested_output"] = "execute_and_answer"
    contract = default_validation_contract(
        question=state["user_question"],
        route_kind=parsed.get("route_kind", fallback["route_kind"]),
        target_metric=parsed.get("target_metric") or fallback.get("target_metric") or "",
        dimensions=list(parsed.get("dimensions") or fallback.get("dimensions") or []),
        selected_tables=list(parsed.get("selected_join_tables") or fallback.get("selected_join_tables") or []),
        mart_name=parsed.get("mart_name") or fallback.get("mart_name"),
    )
    parsed.setdefault("expected_result_shape", contract["expected_result_shape"])
    parsed.setdefault("required_columns", contract["required_columns"])
    parsed.setdefault("required_aggregations", contract["required_aggregations"])
    parsed.setdefault("validation_contract", contract)
    return {"plan": parsed}


def design_mart(state: AgentState):
    route_kind = state["plan"].get("route_kind", "simple")
    task_type = state["plan"].get("task_type")
    if route_kind != "comprehensive" and task_type != "data_mart_build":
        return {"mart_design": {}}

    fallback = default_mart_design(state)
    response = try_llm_json(mart_design_prompt(state))
    parsed = safe_json_parse(response, fallback) if response else fallback
    parsed.setdefault("target_schema", ALLOWED_MART_SCHEMA)
    parsed.setdefault("source_tables", fallback["source_tables"])
    parsed.setdefault("design_reasoning", fallback["design_reasoning"])
    return {"mart_design": parsed}
