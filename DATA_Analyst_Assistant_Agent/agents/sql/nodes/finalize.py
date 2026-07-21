from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import format_result_rows
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import require_route_kind
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState


def _format_statement_results(state: AgentState) -> str:
    statement_results = list(state.get("statement_results") or [])
    if not statement_results:
        return format_result_rows(state.get("sql_result"), max_rows=5)

    chunks: list[str] = []
    for item in statement_results[:5]:
        chunks.append(
            f"[statement {int(item.get('index', 0)) + 1}] row_count={item.get('row_count', 0)} "
            f"preview={format_result_rows(item.get('rows'), max_rows=3)}"
        )
    return "\n".join(chunks)


def _failure_reason(state: AgentState) -> str:
    retry_messages = ((state.get("retry_hint") or {}).get("details") or {}).get("messages") or []
    if retry_messages:
        return str(retry_messages[0])
    return str(state.get("error") or state["validation"].get("reason") or "")


def finalize_answer(state: AgentState):
    if state["validation"].get("result") != "valid":
        return {
            "final_answer": (
                "validation failed\n"
                f"reason: {_failure_reason(state)}\n"
                f"last SQL: {state['sql_draft'].get('sql')}"
            )
        }

    route_kind = require_route_kind(state.get("plan", {}))
    sql_text = state.get("sql_draft", {}).get("sql", "")
    if route_kind == "comprehensive":
        return {
            "final_answer": (
                "comprehensive route created and executed a datamart SQL.\n"
                f"target table: {state.get('sql_draft', {}).get('target_table') or 'unspecified'}\n"
                f"result preview: {_format_statement_results(state)}\n"
                f"final SQL:\n{sql_text}"
            )
        }
    return {
        "final_answer": (
            "simple route created and executed a query SQL.\n"
            f"rows: {state.get('row_count', 0)}\n"
            f"result preview: {_format_statement_results(state)}\n"
            f"final SQL:\n{sql_text}"
        )
    }
