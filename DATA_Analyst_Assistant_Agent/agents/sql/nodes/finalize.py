"""finalize_answer 노드: 검증 통과 후 최종 답변 생성."""

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


def finalize_answer(state: AgentState):
    if state["validation"].get("result") != "valid":
        return {
            "final_answer": (
                "검증 실패\n"
                f"사유: {state['validation'].get('reason')}\n"
                f"마지막 SQL: {state['sql_draft'].get('sql')}"
            )
        }

    route_kind = require_route_kind(state.get("plan", {}))
    sql_text = state.get("sql_draft", {}).get("sql", "")
    if route_kind == "comprehensive":
        return {
            "final_answer": (
                "comprehensive 경로로 datamart 생성 SQL을 작성하고 실행했습니다.\n"
                f"대상 테이블: {state.get('sql_draft', {}).get('target_table') or '미지정'}\n"
                f"실행 결과 미리보기: {_format_statement_results(state)}\n"
                f"최종 SQL:\n{sql_text}"
            )
        }
    return {
        "final_answer": (
            "simple 경로로 조회 SQL을 작성하고 실행했습니다.\n"
            f"행 수: {state.get('row_count', 0)}\n"
            f"실행 결과 미리보기: {_format_statement_results(state)}\n"
            f"최종 SQL:\n{sql_text}"
        )
    }
