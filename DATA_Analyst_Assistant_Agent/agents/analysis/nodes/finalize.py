from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import AnalysisExecutionPlan, AnalysisResult, HumanReview
from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck


def finalize_analysis(state: dict[str, Any]) -> dict[str, Any]:
    terminal_reason = state.get("terminal_reason") or "validated_result"
    result = state.get("result")
    checks = state.get("local_checks", [])
    if result is not None:
        return {"result": result, "local_checks": checks, "terminal_reason": terminal_reason}

    error = state.get("error") or "Analysis workflow did not produce a result."
    orchestration = state["orchestration_state"]
    plan = AnalysisExecutionPlan(
        objective=orchestration.goal or orchestration.user_query,
        question_type=state.get("question_type") or "descriptive",
        tool_names=[],
        requires_human_review=True,
        review_reason=error,
    )
    failed = AnalysisResult(
        run_id=orchestration.run_id,
        goal=orchestration.goal or orchestration.user_query,
        plan=plan,
        method_summary="Analysis did not complete; no statistical tools were executed.",
        key_findings=["Analysis workflow ended before producing executable evidence."],
        limitations=[error],
        source_artifacts={
            "sql": orchestration.artifact_ids.get("sql_agent", []),
            "eda": orchestration.artifact_ids.get("eda_agent", []),
        },
        human_review=HumanReview(required=True, reason=error),
    ).model_dump(mode="json")
    return {
        "result": failed,
        "local_checks": [
            *checks,
            LocalCheck(name="analysis_workflow_completed", passed=False, severity="error", detail=error),
        ],
        "terminal_reason": terminal_reason if terminal_reason != "retrying" else "workflow_failed",
    }
