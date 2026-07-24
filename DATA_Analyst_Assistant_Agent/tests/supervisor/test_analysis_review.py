from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisSelectionResponse,
    ReviewRequest,
)
from DATA_Analyst_Assistant_Agent.supervisor.analysis_review import (
    AnalysisReviewDecision,
    AnalysisReviewResumePayload,
    InvalidAnalysisReviewRequest,
    extract_analysis_review_request,
    validate_analysis_review_resume,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, ArtifactSummary
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    PendingApproval,
    SupervisorState,
    begin_or_retry_agent_node,
    empty_supervisor_state,
    to_orchestration_state,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorInterruptPayload
from DATA_Analyst_Assistant_Agent.supervisor.candidate import commit_candidate
from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    make_collect_analysis_review_node,
    make_resolve_analysis_review_node,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import stage_candidate_result


def _review_request(*, requires_followup_analysis: bool = True) -> dict:
    evidence = {"sample_size": 12}
    if not requires_followup_analysis:
        evidence["analyzed_option_ids"] = ["mean", "median"]
    return {
        "decision_type": "aggregation_method",
        "question": "어떤 대표값을 사용할까요?",
        "proposal": "대표값 선택이 필요합니다.",
        "rationale": ["분포가 비대칭입니다."],
        "evidence": evidence,
        "options": [
            {
                "id": "mean",
                "label": "평균",
                "method": "산술 평균",
                "assumptions": ["극단값 영향이 제한적임"],
                "advantages": ["해석이 익숙함"],
                "limitations": ["극단값에 민감함"],
                "impact": "평균 중심으로 보고합니다.",
                "recommended": False,
            },
            {
                "id": "median",
                "label": "중앙값",
                "method": "50% 분위수",
                "assumptions": ["순서 통계량 사용 가능"],
                "advantages": ["극단값에 강건함"],
                "limitations": ["합계와 직접 연결되지 않음"],
                "impact": "중앙값과 사분위 범위를 보고합니다.",
                "recommended": True,
            },
        ],
        "recommended_option_id": "median",
        "allow_free_text": True,
        "free_text_prompt": "다른 분석 제약을 입력해 주세요.",
        "impact_if_approved": "선택한 대표값으로 해석합니다.",
        "requires_followup_analysis": requires_followup_analysis,
    }


def _analysis_result(review_request: object) -> AgentCompactResult:
    return AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="분석 선택 필요",
        artifact_ids=["analysis_001"],
        artifacts=[
            ArtifactSummary(
                artifact_id="analysis_001",
                kind="analysis_result",
                metadata={"kind": "analysis_result"},
                preview={"review_request": review_request},
                content_hash="hash_001",
            )
        ],
        approval={"required": True, "approval_type": "analysis.review"},
    )


def _pending_review_state(*, requires_followup_analysis: bool) -> dict:
    state = empty_supervisor_state(
        thread_id="thread_001",
        run_id="run_001",
        user_query="매출 분석",
        datasource_id=None,
    )
    result = _analysis_result(
        _review_request(requires_followup_analysis=requires_followup_analysis)
    )
    state, _, _ = begin_or_retry_agent_node(state, "analysis_agent")
    state = stage_candidate_result(state, result)
    candidate = state["pending_result"]
    state["pending_validation"] = {
        "candidate_id": candidate["candidate_id"],
        "validation_id": candidate["validation_id"],
        "agent": "analysis_agent",
        "outcome": {
            "disposition": "await_approval",
            "reason": "분석 방법 선택 필요",
            "reason_code": "none",
            "terminal_state": "running",
        },
        "checks": [],
    }
    return state


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"approval_id": " approval_001 ", "selected_option_id": " median "},
            {"approval_id": "approval_001", "selected_option_id": "median", "free_text": None},
        ),
        (
            {"approval_id": " approval_001 ", "free_text": " 중앙값을 사용해 주세요. "},
            {
                "approval_id": "approval_001",
                "selected_option_id": None,
                "free_text": "중앙값을 사용해 주세요.",
            },
        ),
    ],
)
def test_analysis_review_resume_payload_normalizes_strings(payload: dict, expected: dict) -> None:
    assert AnalysisReviewResumePayload.model_validate(payload).model_dump() == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"approval_id": "approval_001"},
        {"approval_id": " ", "selected_option_id": "median"},
        {"approval_id": "approval_001", "selected_option_id": " "},
        {"approval_id": "approval_001", "free_text": " "},
        {
            "approval_id": "approval_001",
            "selected_option_id": "median",
            "free_text": "평균도 함께",
        },
        {"approval_id": "approval_001", "selected_option_id": "median", "unknown": True},
    ],
)
def test_analysis_review_resume_payload_rejects_invalid_shapes(payload: dict) -> None:
    with pytest.raises(ValidationError):
        AnalysisReviewResumePayload.model_validate(payload)


def test_analysis_selection_response_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisSelectionResponse.model_validate({"selected_option_id": "median", "unknown": True})


@pytest.mark.parametrize(
    "analyzed_option_ids",
    [None, [], ["mean"], ["mean", "mean"], ["mean", "median", "mode"]],
)
def test_no_followup_requires_each_option_to_be_preanalyzed_once(
    analyzed_option_ids: list[str] | None,
) -> None:
    payload = _review_request(requires_followup_analysis=False)
    if analyzed_option_ids is None:
        payload["evidence"].pop("analyzed_option_ids")
    else:
        payload["evidence"]["analyzed_option_ids"] = analyzed_option_ids

    with pytest.raises(ValidationError):
        ReviewRequest.model_validate(payload)


def test_extract_analysis_review_request_returns_full_valid_request() -> None:
    request = extract_analysis_review_request(_analysis_result(_review_request()))

    assert request is not None
    assert request.model_dump(mode="json") == _review_request()


def test_extract_analysis_review_request_returns_none_when_key_is_absent() -> None:
    result = _analysis_result(None)
    result.artifacts[0].preview = {}

    assert extract_analysis_review_request(result) is None


def test_extract_analysis_review_request_treats_null_as_unstructured_review() -> None:
    assert extract_analysis_review_request(_analysis_result(None)) is None


def test_extract_analysis_review_request_fails_closed_when_key_is_invalid() -> None:
    with pytest.raises(InvalidAnalysisReviewRequest, match="invalid_analysis_review_request"):
        extract_analysis_review_request(_analysis_result({"question": "불완전"}))


def test_validate_analysis_review_resume_resolves_selected_option() -> None:
    request = ReviewRequest.model_validate(_review_request(requires_followup_analysis=False))
    decision = validate_analysis_review_resume(
        {"approval_id": "approval_001", "selected_option_id": "median"},
        pending_approval={"approval_id": "approval_001", "candidate_id": "candidate_001"},
        pending_candidate_id="candidate_001",
        review_request=request,
    )

    assert decision == AnalysisReviewDecision(
        approval_id="approval_001",
        candidate_id="candidate_001",
        review_request=request,
        selection_response=AnalysisSelectionResponse(selected_option_id="median"),
        selected_option=request.options[1],
    )


@pytest.mark.parametrize(
    ("payload", "approval_id", "candidate_id", "message"),
    [
        ({"approval_id": "stale", "selected_option_id": "median"}, "approval_001", "candidate_001", "approval_id"),
        ({"approval_id": "approval_001", "selected_option_id": "unknown"}, "approval_001", "candidate_001", "option"),
        ({"approval_id": "approval_001", "free_text": "평균"}, "approval_001", "other_candidate", "candidate"),
    ],
)
def test_validate_analysis_review_resume_rejects_stale_or_invalid_selection(
    payload: dict,
    approval_id: str,
    candidate_id: str,
    message: str,
) -> None:
    request_payload = _review_request()
    if payload.get("free_text"):
        request_payload["allow_free_text"] = False
    with pytest.raises(ValueError, match=message):
        validate_analysis_review_resume(
            payload,
            pending_approval={"approval_id": approval_id, "candidate_id": "candidate_001"},
            pending_candidate_id=candidate_id,
            review_request=ReviewRequest.model_validate(request_payload),
        )


def test_pending_approval_defaults_to_boolean_resume_contract() -> None:
    approval = PendingApproval(
        approval_id="approval_001",
        agent="sql_agent",
        reason="승인 필요",
        approval_type="sql.review",
    )

    assert approval.review_request is None
    assert approval.expected_resume == {"approved": "boolean"}


def test_orchestration_state_exposes_only_analysis_review_decision_history() -> None:
    state = empty_supervisor_state(
        thread_id="thread_001",
        run_id="run_001",
        user_query="매출 분석",
        datasource_id=None,
    )
    state["analysis_selection_response"] = {"selected_option_id": "median"}
    state["analysis_selection_review_request"] = _review_request()
    state["analysis_review_decisions"] = [{"approval_id": "approval_001"}]

    public = to_orchestration_state(state)

    assert public.analysis_review_decisions == [{"approval_id": "approval_001"}]
    assert "analysis_selection_response" not in public.__class__.model_fields
    assert "analysis_selection_review_request" not in public.__class__.model_fields


def test_interrupt_contract_requires_type_specific_fields() -> None:
    clarification = SupervisorInterruptPayload(
        type="clarification",
        status="waiting_input",
        run_id="run_001",
        thread_id="thread_001",
        question="기간은?",
        node="collect_clarification",
    )
    review = SupervisorInterruptPayload(
        type="analysis_review",
        status="waiting_input",
        run_id="run_001",
        thread_id="thread_001",
        question="대표값은?",
        node="collect_analysis_review",
        approval_id="approval_001",
        review_request=_review_request(),
        expected_resume={
            "approval_id": "string",
            "selected_option_id": "string?",
            "free_text": "string?",
        },
    )

    assert clarification.approval_id is None
    assert review.approval_id == "approval_001"
    with pytest.raises(ValidationError):
        SupervisorInterruptPayload(
            type="analysis_review",
            status="waiting_input",
            run_id="run_001",
            thread_id="thread_001",
            question="대표값은?",
            node="collect_analysis_review",
        )
    with pytest.raises(ValidationError):
        SupervisorInterruptPayload(
            type="clarification",
            status="waiting_input",
            run_id="run_001",
            thread_id="thread_001",
            question="기간은?",
            node="collect_clarification",
            approval_id="approval_001",
            review_request=_review_request(),
        )


def test_commit_structured_analysis_review_routes_to_native_interrupt() -> None:
    state = _pending_review_state(requires_followup_analysis=True)

    committed = commit_candidate(state, None)

    candidate_id = state["pending_result"]["candidate_id"]
    assert committed["pending_approval"] == {
        "approval_id": f"run_001:analysis_agent:{candidate_id}:approval",
        "agent": "analysis_agent",
        "reason": "분석 선택 필요",
        "approval_type": "analysis.review",
        "candidate_id": candidate_id,
        "validation_id": state["pending_result"]["validation_id"],
        "content_hashes": {"analysis_001": "hash_001"},
        "review_request": _review_request(),
        "expected_resume": {
            "approval_id": "string",
            "selected_option_id": "string?",
            "free_text": "string?",
        },
    }
    assert committed["terminal_state"] == "running"
    assert committed["next_action"] == "collect_analysis_review"
    assert committed["final_answer"] == ""


def test_commit_unstructured_analysis_review_keeps_general_approval_flow() -> None:
    state = empty_supervisor_state(
        thread_id="thread_001",
        run_id="run_001",
        user_query="매출 분석",
        datasource_id=None,
    )
    state, _, _ = begin_or_retry_agent_node(state, "analysis_agent")
    state = stage_candidate_result(state, _analysis_result(None))
    candidate = state["pending_result"]
    state["pending_validation"] = {
        "candidate_id": candidate["candidate_id"],
        "validation_id": candidate["validation_id"],
        "agent": "analysis_agent",
        "outcome": {"disposition": "await_approval", "reason": "일반 검토"},
        "checks": [],
    }

    committed = commit_candidate(state, None)

    assert committed["terminal_state"] == "needs_user_approval"
    assert committed["next_action"] == "finalize"
    assert committed["pending_approval"]["approval_id"] == "run_001:analysis_agent:approval"
    assert committed["pending_approval"]["review_request"] is None
    assert committed["pending_approval"]["expected_resume"] == {"approved": "boolean"}


def test_collect_analysis_review_interrupts_then_normalizes_resume() -> None:
    state = commit_candidate(_pending_review_state(requires_followup_analysis=True), None)
    builder = StateGraph(SupervisorState)
    builder.add_node("collect", make_collect_analysis_review_node())
    builder.add_edge(START, "collect")
    builder.add_edge("collect", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread_001"}}

    paused = graph.invoke(state, config)
    payload = paused["__interrupt__"][0].value
    resumed = graph.invoke(
        Command(
            resume={
                "approval_id": state["pending_approval"]["approval_id"],
                "selected_option_id": " median ",
            }
        ),
        config,
    )

    assert payload["type"] == "analysis_review"
    assert payload["approval_id"] == state["pending_approval"]["approval_id"]
    assert payload["review_request"] == _review_request()
    assert resumed["analysis_selection_response"] == {
        "selected_option_id": "median",
        "free_text": None,
    }


def test_resolve_followup_review_quarantines_candidate_without_consuming_retry() -> None:
    state = commit_candidate(_pending_review_state(requires_followup_analysis=True), None)
    state["quarantined_artifacts"] = [
        {"artifact_id": "older_artifact"},
        {"artifact_id": "older_reasoned", "quarantine_reason": "older_reason"},
    ]
    state["analysis_selection_response"] = {
        "selected_option_id": "median",
        "free_text": None,
    }

    resolved = make_resolve_analysis_review_node()(state)

    assert resolved["pending_result"] is None
    assert resolved["pending_approval"] is None
    assert resolved["next_action"] == "call_analysis_agent"
    assert resolved["retry_counts"] == {}
    assert resolved["analysis_selection_response"]["selected_option_id"] == "median"
    assert resolved["analysis_selection_review_request"] == _review_request()
    assert resolved["rejected_results"][-1]["metadata"]["reason_code"] == "analysis.review_superseded"
    assert "quarantine_reason" not in resolved["quarantined_artifacts"][0]
    assert resolved["quarantined_artifacts"][1]["quarantine_reason"] == "older_reason"
    assert all(
        artifact["quarantine_reason"] == "analysis.review_superseded"
        for artifact in resolved["quarantined_artifacts"][2:]
    )
    assert resolved["analysis_review_decisions"][0]["selected_option"]["label"] == "중앙값"


def test_resolve_preanalyzed_review_promotes_without_reanalysis() -> None:
    pending = _pending_review_state(requires_followup_analysis=False)
    pending["pending_validation"]["checks"] = [
        {
            "name": "semantic",
            "passed": True,
            "findings": [],
            "details": {"recommended_next_action": "call_analysis_agent"},
        }
    ]
    state = commit_candidate(pending, None)
    state["analysis_selection_response"] = {
        "selected_option_id": "mean",
        "free_text": None,
    }

    resolved = make_resolve_analysis_review_node()(state)

    assert resolved["pending_result"] is None
    assert resolved["pending_approval"] is None
    assert resolved["completed_agents"] == ["analysis_agent"]
    assert resolved["next_action"] == "decide_next_action"
    assert resolved["analysis_selection_response"] is None
    assert resolved["analysis_selection_review_request"] is None
    assert resolved["analysis_review_decisions"][0]["selected_option"]["id"] == "mean"


def test_resolve_analysis_review_is_idempotent_for_decision_history() -> None:
    state = commit_candidate(_pending_review_state(requires_followup_analysis=True), None)
    state["analysis_selection_response"] = {"free_text": "중앙값을 사용", "selected_option_id": None}
    once = make_resolve_analysis_review_node()(state)
    replay_state = {
        **state,
        "analysis_review_decisions": once["analysis_review_decisions"],
    }

    replayed = make_resolve_analysis_review_node()(replay_state)

    assert len(replayed["analysis_review_decisions"]) == 1


def test_resolve_analysis_review_allows_second_distinct_decision() -> None:
    state = commit_candidate(_pending_review_state(requires_followup_analysis=True), None)
    state["analysis_selection_response"] = {
        "selected_option_id": "median",
        "free_text": None,
    }
    state["analysis_review_decisions"] = [
        {
            "approval_id": "previous_approval",
            "candidate_id": "previous_candidate",
        }
    ]

    resolved = make_resolve_analysis_review_node()(state)

    assert resolved["terminal_state"] == "running"
    assert resolved["next_action"] == "call_analysis_agent"
    assert len(resolved["analysis_review_decisions"]) == 2


def test_resolve_analysis_review_rejects_third_distinct_decision() -> None:
    state = commit_candidate(_pending_review_state(requires_followup_analysis=True), None)
    state["analysis_selection_response"] = {
        "selected_option_id": "median",
        "free_text": None,
    }
    state["analysis_review_decisions"] = [
        {"approval_id": "approval_001", "candidate_id": "candidate_001"},
        {"approval_id": "approval_002", "candidate_id": "candidate_002"},
    ]

    resolved = make_resolve_analysis_review_node()(state)

    assert resolved["terminal_state"] == "failed_terminal"
    assert resolved["next_action"] == "finalize"
    assert resolved["current_step"] == "resolve_analysis_review"
    assert resolved["final_answer"] == "analysis review는 최대 2회까지 허용됩니다."


def test_resolve_analysis_review_replay_is_allowed_at_limit() -> None:
    state = commit_candidate(_pending_review_state(requires_followup_analysis=True), None)
    state["analysis_selection_response"] = {
        "selected_option_id": "median",
        "free_text": None,
    }
    pending_approval = state["pending_approval"]
    pending_result = state["pending_result"]
    state["analysis_review_decisions"] = [
        {
            "approval_id": pending_approval["approval_id"],
            "candidate_id": pending_result["candidate_id"],
        },
        {"approval_id": "other_approval", "candidate_id": "other_candidate"},
    ]

    resolved = make_resolve_analysis_review_node()(state)

    assert resolved["terminal_state"] == "running"
    assert resolved["next_action"] == "call_analysis_agent"
    assert len(resolved["analysis_review_decisions"]) == 2


def test_review_option_ids_are_trimmed_before_resume_matching() -> None:
    payload = _review_request()
    payload["options"][0]["id"] = " mean "
    request = ReviewRequest.model_validate(payload)

    decision = validate_analysis_review_resume(
        {"approval_id": "approval_001", "selected_option_id": " mean "},
        pending_approval={"approval_id": "approval_001", "candidate_id": "candidate_001"},
        pending_candidate_id="candidate_001",
        review_request=request,
    )

    assert request.options[0].id == "mean"
    assert decision.selected_option is not None
    assert decision.selected_option.id == "mean"
