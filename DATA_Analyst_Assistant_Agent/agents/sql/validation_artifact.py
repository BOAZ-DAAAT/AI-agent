from __future__ import annotations

from typing import Any


def build_validation_summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": result.get("plan") or {},
        "sql_draft": result.get("sql_draft") or {},
        "validation": result.get("validation") or {},
        "validation_findings": result.get("validation_findings") or [],
        "retry_hint": result.get("retry_hint") or {},
        "row_count": result.get("row_count", 0),
        "final_answer": result.get("final_answer") or "",
    }
