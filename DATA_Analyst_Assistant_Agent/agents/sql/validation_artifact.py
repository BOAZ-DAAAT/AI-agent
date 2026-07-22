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
        "generation_failure_reason": result.get("generation_failure_reason") or "",
        "failed_statement_index": result.get("failed_statement_index"),
        "failed_statement_sql": result.get("failed_statement_sql") or "",
        "classification": result.get("classification") or "none",
        "repair_strategy": result.get("repair_strategy") or "none",
        "repair_attempted": bool(result.get("repair_attempted")),
        "repair_validation_result": result.get("repair_validation_result") or {},
        "final_answer": result.get("final_answer") or "",
    }
