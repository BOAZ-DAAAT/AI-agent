from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.candidate import build_step_summary, commit_candidate
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
    stage_candidate_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    ValidationCheckResult,
    ValidationOutcome,
    ValidationRecord,
)


class RecordingBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def append_run_event(
        self,
        run_id: str,
        event_type: str,
        _message: str,
        **kwargs: object,
    ) -> None:
        self.events.append((run_id, event_type, kwargs))


def test_build_step_summary_is_deterministic_and_deduplicates_artifact_ids() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["artifact_1", "artifact_2"],
        artifacts=[
            ArtifactSummary(artifact_id="artifact_2"),
            ArtifactSummary(artifact_id="artifact_3"),
        ],
    )

    summary = build_step_summary(
        result,
        "call_analysis_agent",
        "call_report_agent",
    )

    assert summary.model_dump(mode="json") == {
        "step": "execute_subagent",
        "agent": "analysis_agent",
        "action": "call_analysis_agent",
        "summary": "분석 완료",
        "artifact_ids": ["artifact_1", "artifact_2", "artifact_3"],
        "next_action": "call_report_agent",
    }


def test_build_step_summary_uses_report_step() -> None:
    result = AgentCompactResult(
        agent="report_agent",
        status="success",
        summary="보고서 완료",
        artifact_ids=["report_1"],
    )

    summary = build_step_summary(result, "call_report_agent", "finalize")

    assert summary.step == "generate_report"


def _validated_state(
    result: AgentCompactResult,
    disposition: str,
):
    state = empty_supervisor_state(
        thread_id="thread_001",
        run_id="run_001",
        user_query="분석해줘",
        datasource_id="ds_001",
    )
    state["next_action"] = {
        "sql_agent": "call_sql_agent",
        "eda_agent": "call_eda_agent",
        "analysis_agent": "call_analysis_agent",
        "report_agent": "call_report_agent",
    }[result.agent]
    staged = stage_candidate_result(state, result, {})
    pending = staged["pending_result"] or {}
    staged["pending_validation"] = ValidationRecord(
        candidate_id=pending["candidate_id"],
        validation_id=pending["validation_id"],
        agent=result.agent,
        outcome=ValidationOutcome(
            disposition=disposition,
            reason="검증 결과",
            **({"terminal_state": "failed_terminal"} if disposition == "reject" else {}),
        ),
        checks=[],
    ).model_dump(mode="json")
    return staged


def test_commit_candidate_accepts_and_summarizes_exactly_once() -> None:
    state = _validated_state(
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["analysis_1"],
        ),
        "accept",
    )

    committed = commit_candidate(state, None)
    recommitted = commit_candidate(committed, None)

    assert len(committed["validation_history"]) == 1
    assert len(committed["agent_results"]) == 1
    assert len(committed["step_summaries"]) == 1
    assert committed["pending_result"] is None
    assert recommitted["validation_history"] == committed["validation_history"]
    assert recommitted["agent_results"] == committed["agent_results"]
    assert recommitted["step_summaries"] == committed["step_summaries"]


def test_commit_candidate_retry_rejects_and_increments_once() -> None:
    state = _validated_state(
        AgentCompactResult(
            agent="sql_agent",
            status="failed",
            summary="SQL 실패",
            retryable=True,
        ),
        "retry",
    )

    committed = commit_candidate(state, None)
    recommitted = commit_candidate(committed, None)

    assert committed["retry_counts"] == {"sql_agent": 1}
    assert committed["semantic_recovery_attempts"] == {}
    assert len(committed["rejected_results"]) == 1
    assert committed["next_action"] == "call_sql_agent"
    assert committed["step_summaries"] == []
    assert recommitted["retry_counts"] == committed["retry_counts"]
    assert recommitted["rejected_results"] == committed["rejected_results"]


def test_commit_candidate_emits_backend_event_once() -> None:
    backend = RecordingBackend()
    state = _validated_state(
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["analysis_1"],
        ),
        "accept",
    )

    committed = commit_candidate(state, backend)
    commit_candidate(committed, backend)

    assert [event[1] for event in backend.events] == ["evidence.promoted"]


def _semantic_recovery_state(
    *,
    agent: str = "analysis_agent",
    recommendation: str | None = "call_sql_agent",
    accepted_evidence: dict[str, list[dict[str, str]]] | None = None,
):
    result = AgentCompactResult(
        agent=agent,
        status="success",
        summary="의미적으로 부족한 후보",
        artifacts=[ArtifactSummary(artifact_id=f"artifact_{agent}")],
    )
    state = _validated_state(result, "accept")
    state["accepted_evidence"] = accepted_evidence or {}
    state["artifacts"] = dict(state["accepted_evidence"])
    pending = state["pending_result"] or {}
    state["pending_validation"] = ValidationRecord(
        candidate_id=pending["candidate_id"],
        validation_id=pending["validation_id"],
        agent=result.agent,
        outcome=ValidationOutcome(
            disposition="recover",
            reason="필수 근거가 누락되었습니다.",
            recovery_action=recommendation,
        ),
        checks=[
            ValidationCheckResult(
                name="semantic",
                passed=False,
                details={
                    "recommended_next_action": recommendation or "",
                    "missing_evidence": ["monthly_sales"],
                },
            )
        ],
    ).model_dump(mode="json")
    return state


def test_commit_semantic_recovery_quarantines_candidate_and_schedules_first_target() -> None:
    backend = RecordingBackend()
    state = _semantic_recovery_state(recommendation="call_sql_agent")

    committed = commit_candidate(state, backend)

    assert committed["pending_result"] is None
    assert len(committed["rejected_results"]) == 1
    assert committed["quarantined_artifacts"][0]["artifact_id"] == "artifact_analysis_agent"
    assert committed["agent_results"] == []
    assert committed["completed_agents"] == []
    assert committed["failed_agents"] == []
    assert committed["terminal_state"] == "running"
    assert committed["next_action"] == "call_sql_agent"
    assert committed["semantic_recovery_attempts"] == {"sql_agent": 1}
    assert [event["type"] for event in committed["run_events"][-2:]] == [
        "semantic_recovery.quarantined",
        "semantic_recovery.scheduled",
    ]
    assert [event[1] for event in backend.events] == [
        "semantic_recovery.quarantined",
        "semantic_recovery.scheduled",
    ]


def test_commit_semantic_recovery_records_full_precondition_correction_path() -> None:
    state = _semantic_recovery_state(recommendation="call_report_agent")

    committed = commit_candidate(state, None)

    assert committed["next_action"] == "call_sql_agent"
    assert committed["semantic_recovery_attempts"] == {"sql_agent": 1}
    semantic_details = committed["validation_history"][-1]["checks"][-1]["details"]
    assert semantic_details["original_recovery_action"] == "call_report_agent"
    assert semantic_details["precondition_path"] == [
        "call_report_agent",
        "call_analysis_agent",
        "call_sql_agent",
    ]
    assert semantic_details["recovery_action"] == "call_sql_agent"
    assert len(semantic_details["precondition_reasons"]) == 2
    assert semantic_details["budget"] == {
        "agent": "sql_agent",
        "attempt_before": 0,
        "attempt_after": 1,
    }
    assert any("리포트" in limitation for limitation in committed["limitations"])
    assert any(
        event["type"] == "semantic_recovery.precondition_adjusted"
        for event in committed["run_events"]
    )


def test_commit_semantic_recovery_budget_exhaustion_uses_limited_report() -> None:
    state = _semantic_recovery_state(
        recommendation="call_sql_agent",
        accepted_evidence={"sql_agent": [{"artifact_id": "accepted_sql"}]},
    )
    state["semantic_recovery_attempts"] = {"sql_agent": 1}

    committed = commit_candidate(state, None)

    assert committed["next_action"] == "call_report_agent"
    assert committed["terminal_state"] == "running"
    assert committed["semantic_recovery_attempts"] == {
        "sql_agent": 1,
        "report_agent": 1,
    }
    assert any(
        event["type"] == "semantic_recovery.budget_exhausted"
        for event in committed["run_events"]
    )
    assert committed["run_events"][-1]["type"] == "semantic_recovery.limited_report"
    assert any("제한적 Report" in limitation for limitation in committed["limitations"])


def test_semantic_recovery_budget_is_counted_by_corrected_actual_target() -> None:
    state = _semantic_recovery_state(
        recommendation="call_eda_agent",
        accepted_evidence={"sql_agent": [{"artifact_id": "accepted_sql"}]},
    )
    state["semantic_recovery_attempts"] = {"sql_agent": 1}

    committed = commit_candidate(state, None)

    assert committed["next_action"] == "call_eda_agent"
    assert committed["semantic_recovery_attempts"] == {
        "sql_agent": 1,
        "eda_agent": 1,
    }


def test_limited_report_budget_exhaustion_ends_with_recoverable_context() -> None:
    state = _semantic_recovery_state(
        recommendation=None,
        accepted_evidence={"sql_agent": [{"artifact_id": "accepted_sql"}]},
    )
    state["semantic_recovery_attempts"] = {"report_agent": 1}

    committed = commit_candidate(state, None)

    assert committed["terminal_state"] == "failed_with_recoverable_context"
    assert committed["next_action"] == "finalize"
    semantic_details = committed["validation_history"][-1]["checks"][-1]["details"]
    assert "소진" in semantic_details["fallback_reason"]


def test_commit_semantic_recovery_without_accepted_evidence_fails_terminally() -> None:
    state = _semantic_recovery_state(recommendation=None)

    committed = commit_candidate(state, None)

    assert committed["terminal_state"] == "failed_terminal"
    assert committed["next_action"] == "finalize"
    assert committed["failed_agents"] == []


def test_limited_report_semantic_failure_ends_with_recoverable_context() -> None:
    state = _semantic_recovery_state(
        agent="report_agent",
        recommendation=None,
        accepted_evidence={"analysis_agent": [{"artifact_id": "accepted_analysis"}]},
    )
    state["semantic_recovery_attempts"] = {"report_agent": 1}

    committed = commit_candidate(state, None)

    assert committed["terminal_state"] == "failed_with_recoverable_context"
    assert committed["next_action"] == "finalize"
    assert committed["semantic_recovery_attempts"] == {"report_agent": 1}


def test_semantic_warning_reason_is_added_to_limitations_once() -> None:
    state = _validated_state(
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="제한적 분석",
            artifact_ids=["analysis_1"],
        ),
        "accept_with_limitations",
    )
    state["pending_validation"]["outcome"]["reason"] = "일부 기간 데이터가 희소합니다."
    state["limitations"] = ["일부 기간 데이터가 희소합니다."]

    committed = commit_candidate(state, None)

    assert committed["limitations"] == ["일부 기간 데이터가 희소합니다."]


def test_semantic_warning_is_preserved_while_candidate_waits_for_approval() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="승인 필요",
        artifact_ids=["analysis_1"],
        approval={"required": True, "reason": "사용자 검토가 필요합니다."},
    )
    state = _validated_state(result, "await_approval")
    state["pending_validation"]["checks"] = [
        ValidationCheckResult(
            name="semantic",
            passed=True,
            findings=[
                {
                    "code": "semantic_validation_warning",
                    "source": "supervisor",
                    "severity": "warning",
                    "disposition": "limitation",
                    "message": "일부 기간 데이터가 희소합니다.",
                }
            ],
        ).model_dump(mode="json")
    ]

    committed = commit_candidate(state, None)

    assert committed["terminal_state"] == "needs_user_approval"
    assert committed["limitations"] == ["일부 기간 데이터가 희소합니다."]
