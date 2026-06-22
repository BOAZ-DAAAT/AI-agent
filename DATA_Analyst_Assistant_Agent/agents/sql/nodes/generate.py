"""generate_sql 노드: 질문 분석/마트 설계 기반 SQL 생성."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import clean_sql, get_llm, safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft


def generate_sql(state: AgentState):
    feedback = state.get("feedback", "").strip()
    task_type = state["plan"].get("task_type", "query_answer")

    if task_type == "data_mart_build":
        prompt = prompts.generate_mart_prompt(state, feedback)
    else:
        prompt = prompts.generate_query_prompt(state, feedback)

    response = get_llm().invoke(prompt).content

    fallback = SQLDraft(
        sql="SELECT 1;",
        sql_type="select",
        target_table=None,
        source_tables=[],
        columns_used=[],
        business_grain=None,
        precheck_sql=None,
        postcheck_sql=None,
        reasoning="SQL 생성 파싱 실패"
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    parsed["sql"] = clean_sql(parsed.get("sql", "SELECT 1;"))

    if parsed.get("precheck_sql"):
        parsed["precheck_sql"] = clean_sql(parsed["precheck_sql"])
    if parsed.get("postcheck_sql"):
        parsed["postcheck_sql"] = clean_sql(parsed["postcheck_sql"])

    return {"sql_draft": parsed}
