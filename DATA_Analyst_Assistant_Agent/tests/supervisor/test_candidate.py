from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.candidate import build_step_summary, commit_candidate
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
    stage_candidate_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
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
