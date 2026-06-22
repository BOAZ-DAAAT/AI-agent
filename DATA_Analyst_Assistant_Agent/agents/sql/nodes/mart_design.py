"""design_mart 노드: 데이터마트 설계(task_type=data_mart_build 일 때만)."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, MartDesign


def design_mart(state: AgentState):
    if state["plan"].get("task_type") != "data_mart_build":
        return {"mart_design": {}}

    response = get_llm().invoke(prompts.mart_design_prompt(state)).content

    fallback = MartDesign(
        mart_name=state["plan"].get("mart_name") or "mart_unknown",
        target_schema=ALLOWED_MART_SCHEMA,
        grain=state["plan"].get("grain") or "grain 미정",
        source_tables=state["plan"].get("relevant_tables", []),
        key_columns=[],
        measure_columns=[],
        dimension_columns=[],
        incremental_column=None,
        load_strategy=state["plan"].get("load_strategy") or "full_refresh",
        design_reasoning="마트 설계 파싱 실패"
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    return {"mart_design": parsed}
