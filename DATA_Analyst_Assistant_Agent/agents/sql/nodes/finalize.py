"""finalize_answer 노드: 검증 통과 시 최종 답변 생성."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import get_llm
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def finalize_answer(state: AgentState):
    if state["validation"].get("result") != "valid":
        return {
            "final_answer": (
                "검증 실패\n"
                f"사유: {state['validation'].get('reason')}\n"
                f"마지막 SQL: {state['sql_draft'].get('sql')}"
            )
        }

    if state["plan"].get("task_type") == "data_mart_build":
        answer = get_llm().invoke(prompts.finalize_mart_prompt(state)).content.strip()
        return {"final_answer": answer}

    answer = get_llm().invoke(prompts.finalize_answer_prompt(state)).content.strip()
    if _looks_like_markdown_table(answer):
        answer = get_llm().invoke(prompts.finalize_rewrite_prompt(state, answer)).content.strip()
    return {"final_answer": answer}


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    separator_lines = [line for line in table_lines if set(line.replace("|", "").strip()) <= {"-", ":"}]
    return len(table_lines) >= 3 and bool(separator_lines)
