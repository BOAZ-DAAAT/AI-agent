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

    error = state.get("error") or "분석 워크플로가 결과를 생성하지 못했습니다."
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
        method_summary="분석이 완료되지 않아 통계 도구가 실행되지 않았습니다.",
        key_findings=["실행 가능한 근거가 생성되기 전에 분석 워크플로가 종료되었습니다."],
        limitations=[error],
        source_artifacts={
            "sql": orchestration.artifact_ids.get("sql_agent", []),
            "eda": orchestration.artifact_ids.get("eda_agent", []),
        },
        human_review=HumanReview(required=True, reason=error),
        status="failed",
    ).model_dump(mode="json")
    return {
        "result": failed,
        "local_checks": [
            *checks,
            LocalCheck(name="analysis_workflow_completed", passed=False, severity="error", detail=error),
        ],
        "terminal_reason": terminal_reason if terminal_reason != "retrying" else "workflow_failed",
    }
