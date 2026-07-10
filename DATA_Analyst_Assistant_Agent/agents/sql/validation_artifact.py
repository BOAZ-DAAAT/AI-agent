from __future__ import annotations

from typing import Any


def build_validation_summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": result.get("plan") or {},
        "sql_draft": result.get("sql_draft") or {},
        "statement_results": result.get("statement_results") or [],
        "validation": result.get("validation") or {},
        "validation_findings": result.get("validation_findings") or [],
        "retry_hint": result.get("retry_hint") or {},
        "row_count": result.get("row_count", 0),
        "generation_source": result.get("generation_source") or "llm",
        "fallback_reason": result.get("fallback_reason") or "",
        "failed_statement_index": result.get("failed_statement_index"),
        "failed_statement_sql": result.get("failed_statement_sql") or "",
        "final_answer": result.get("final_answer") or "",
    }
