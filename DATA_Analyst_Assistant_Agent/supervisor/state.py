from __future__ import annotations

import json
from uuid import uuid4

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AnalysisPlan,
    ApprovalRequirement,
    OrchestrationState,
    RetryHint,
    SupervisorTerminalState,
    ValidationFinding,
)


AgentName = Literal["sql_agent", "eda_agent", "analysis_agent", "insight"]
AgentStatusValue = Literal["success", "warning", "failed", "approval_required"]
NodeExecutionStatus = Literal["running", "waiting", "completed", "failed"]
ActiveNodeStatus = Literal["running", "waiting"]
NodeLifecycleEventType = Literal[
    "agent.started",
    "agent.progress",
    "agent.retrying",
    "agent.waiting",
    "agent.resumed",
    "agent.completed",
    "agent.discarded",
    "agent.failed",
]
NodeActivationEventType = Literal[
    "agent.started",
    "agent.retrying",
    "agent.resumed",
]
NextAction = Literal[
    "clarify",
    "create_plan",
    "decide_next_action",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "call_insight",
    "collect_analysis_review",
    "finalize",
    "fail",
]
ExecutionNextAction = Literal[
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "finalize",
    "fail",
]
PostExecutionNextAction = Literal[
    "decide_next_action",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "finalize",
    "fail",
]


class ArtifactSummary(BaseModel):
    artifact_id: str
    type: str = ""
    kind: str = ""
    summary: str = ""
    uri: str | None = None
    run_id: str = ""
    content_hash: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)


class AgentCompactResult(BaseModel):
    agent: AgentName
    status: AgentStatusValue
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    findings: list[ValidationFinding] = Field(default_factory=list)
    retry_hint: RetryHint = Field(default_factory=RetryHint)
    approval: ApprovalRequirement = Field(default_factory=ApprovalRequirement)
    fallback_used: bool = False
    retryable: bool = False
    error: str = ""


class StepSummary(BaseModel):
    step: str
    agent: AgentName | None = None
    action: str
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    next_action: str = ""


# 현재 실행 중인 에이전트 작업 노드와 상태를 관리
class ActiveNodeExecution(BaseModel):
    node_id: str
    agent_name: AgentName
    parent_node_id: str | None = None
    node_sequence: int = Field(ge=1)
    attempt: int = Field(default=1, ge=1)
    status: ActiveNodeStatus = "running"


class CompletedNodeExecution(BaseModel):
    node_id: str
    agent_name: AgentName
    parent_node_id: str | None = None
    node_sequence: int = Field(ge=1)
    attempt: int = Field(ge=1)
    status: Literal["completed"] = "completed"
    summary: StepSummary


class FailedNodeExecution(BaseModel):
    node_id: str
    agent_name: AgentName
    parent_node_id: str | None = None
    node_sequence: int = Field(ge=1)
    attempt: int = Field(ge=1)
    status: Literal["failed"] = "failed"
    reason: str
    reason_code: str = ""


class PendingApproval(BaseModel):
    approval_id: str
    agent: AgentName
    reason: str
    approval_type: str
    candidate_id: str = ""
    validation_id: str = ""
    content_hashes: dict[str, str] = Field(default_factory=dict)
    review_request: dict[str, Any] | None = None
    expected_resume: dict[str, str] = Field(
        default_factory=lambda: {"approved": "boolean"}
    )


class SupervisorState(TypedDict, total=False):
    thread_id: str
    current_run_id: str
    run_ids: list[str]
    project_id: str | None
    datasource_id: str | None
    catalog_summary: dict[str, Any] | None
    latest_user_query: str
    user_turns: list[dict[str, str]]
    clarified_query: str
    needs_clarification: bool
    clarification_question: str
    analysis_rule_context: dict[str, Any] | None
    analysis_rule_retrieval: dict[str, Any]
    analysis_plan: dict[str, Any]
    current_step: str
    next_action: NextAction
    terminal_state: str
    final_answer: str
    agent_results: list[dict[str, Any]]
    last_agent_result: dict[str, Any]
    artifacts: dict[str, list[dict[str, Any]]]
    validation_history: list[dict[str, Any]]
    step_summaries: list[dict[str, Any]]
    completed_agents: list[str]
    failed_agents: list[str]
    pending_approval: dict[str, Any] | None
    generated_sql: str
    error_state: dict[str, Any]
    retry_counts: dict[str, int]
    failure_streaks: dict[str, dict[str, Any]]
    max_retry_per_agent: int
    llm_decisions: list[dict[str, Any]]
    decision_errors: list[dict[str, Any]]
    pending_result: dict[str, Any] | None
    pending_validation: dict[str, Any] | None
    result_history: list[dict[str, Any]]
    accepted_evidence: dict[str, list[dict[str, Any]]]
    rejected_results: list[dict[str, Any]]
    quarantined_artifacts: list[dict[str, Any]]
    state_schema_version: int
    semantic_retry_counts: dict[str, int]
    semantic_recovery_attempts: dict[str, int]
    limitations: list[str]
    run_events: list[dict[str, Any]]
    analysis_selection_response: dict[str, Any] | None
    analysis_selection_review_request: dict[str, Any] | None
    analysis_review_decisions: list[dict[str, Any]]

    # 노드 상태 State
    active_node: dict[str, Any] | None
    last_completed_node_id: str | None
    node_sequence: int


def _ensure_json_serializable(state: SupervisorState) -> SupervisorState:
    try:
        json.dumps(state, ensure_ascii=False)
    except TypeError as exc:
        raise ValueError("SupervisorState must be JSON serializable") from exc
    return state


def empty_supervisor_state(
    *,
    thread_id: str,
    run_id: str,
    user_query: str,
    datasource_id: str | None,
    project_id: str | None = None,
    catalog_summary: dict[str, Any] | None = None,
) -> SupervisorState:
    state: SupervisorState = {
        "thread_id": thread_id,
        "current_run_id": run_id,
        "run_ids": [run_id],
        "project_id": project_id,
        "datasource_id": datasource_id,
        "catalog_summary": catalog_summary,
        "latest_user_query": user_query,
        "user_turns": [{"run_id": run_id, "query": user_query}],
        "clarified_query": "",
        "needs_clarification": False,
        "clarification_question": "",
        "analysis_rule_context": None,
        "analysis_rule_retrieval": {"status": "not_started"},
        "analysis_plan": {},
        "current_step": "created",
        "next_action": "create_plan",
        "terminal_state": "running",
        "final_answer": "",
        "agent_results": [],
        "last_agent_result": {},
        "artifacts": {},
        "validation_history": [],
        "step_summaries": [],
        "completed_agents": [],
        "failed_agents": [],
        "pending_approval": None,
        "generated_sql": "",
        "error_state": {},
        "retry_counts": {},
        "failure_streaks": {},
        "max_retry_per_agent": 1,
        "llm_decisions": [],
        "decision_errors": [],
        "pending_result": None,
        "pending_validation": None,
        "result_history": [],
        "accepted_evidence": {},
        "rejected_results": [],
        "quarantined_artifacts": [],
        "state_schema_version": 6,
        "semantic_retry_counts": {},
        "semantic_recovery_attempts": {},
        "limitations": [],
        "run_events": [],
        
        # 노드 관련
        "active_node": None,
        "last_completed_node_id": None,
        "node_sequence": 0,

        "analysis_selection_response": None,
        "analysis_selection_review_request": None,
        "analysis_review_decisions": [],
    }
    return _ensure_json_serializable(state)


def merge_agent_result(state: SupervisorState, result: AgentCompactResult) -> SupervisorState:
    if result.status == "approval_required" and not result.approval.required:
        raise ValueError("status=approval_required이지만 approval.required=false입니다.")
    staged = stage_candidate_result(state, result, {})
    if result.approval.required:
        candidate = staged["pending_result"] or {}
        staged["pending_approval"] = PendingApproval(
            approval_id=f"{state['current_run_id']}:{result.agent}:approval",
            agent=result.agent,
            reason=result.approval.reason or result.summary,
            approval_type=result.approval.approval_type or "agent_approval",
            candidate_id=str(candidate.get("candidate_id", "")),
            validation_id=str(candidate.get("validation_id", "")),
            content_hashes=dict(candidate.get("content_hashes") or {}),
        ).model_dump(mode="json")
        staged["terminal_state"] = SupervisorTerminalState.needs_user_approval.value
        return _ensure_json_serializable(staged)
    return promote_pending_result(staged)


def stage_candidate_result(
    state: SupervisorState,
    result: AgentCompactResult,
    state_updates: dict[str, Any] | None = None,
) -> SupervisorState:
    normalized = normalize_supervisor_state(state)
    candidate_id = f"candidate_{uuid4().hex}"
    validation_id = f"validation_{uuid4().hex}"
    candidate = {
        "candidate_id": candidate_id,
        "validation_id": validation_id,
        "result": result.model_dump(mode="json"),
        "state_updates": dict(state_updates or {}),
        "content_hashes": {
            artifact.artifact_id: artifact.content_hash
            for artifact in result.artifacts
            if artifact.content_hash
        },
    }
    history = list(normalized.get("result_history", []))
    history.append(candidate)
    events = list(normalized.get("run_events", []))
    events.append({"type": "result.staged", "candidate_id": candidate_id, "agent": result.agent})
    return _ensure_json_serializable(
        {**normalized, "pending_result": candidate, "result_history": history, "run_events": events}
    )


def promote_pending_result(
    state: SupervisorState,
    *,
    approval_granted: bool = False,
) -> SupervisorState:
    normalized = normalize_supervisor_state(state)
    candidate = normalized.get("pending_result")
    if not isinstance(candidate, dict):
        raise ValueError("승격할 pending_result가 없습니다.")
    result = AgentCompactResult.model_validate(candidate.get("result") or {})
    if result.status == "approval_required":
        if not result.approval.required:
            raise ValueError("status=approval_required이지만 approval.required=false입니다.")
        if not approval_granted:
            raise ValueError("approval_required 결과를 승격하려면 승인이 필요합니다.")
        result = result.model_copy(update={"status": "success"})
    elif result.approval.required and not approval_granted:
        raise ValueError("approval.required=true 결과를 승격하려면 승인이 필요합니다.")
    if result.status not in {"success", "warning"}:
        raise ValueError("성공 또는 경고 결과만 accepted evidence로 승격할 수 있습니다.")

    agent_results = list(normalized.get("agent_results", []))
    agent_results.append(result.model_dump(mode="json"))
    accepted = {agent: list(items) for agent, items in normalized.get("accepted_evidence", {}).items()}
    accepted.setdefault(result.agent, [])
    summaries = list(result.artifacts)
    known_ids = {item.artifact_id for item in summaries}
    summaries.extend(
        ArtifactSummary(artifact_id=artifact_id)
        for artifact_id in result.artifact_ids
        if artifact_id and artifact_id not in known_ids
    )
    for summary in summaries:
        payload = summary.model_dump(mode="json")
        if not any(item.get("artifact_id") == summary.artifact_id for item in accepted[result.agent]):
            accepted[result.agent].append(payload)

    completed = list(normalized.get("completed_agents", []))
    if result.agent not in completed:
        completed.append(result.agent)
    failed = [agent for agent in normalized.get("failed_agents", []) if agent != result.agent]
    failure_streaks = {
        agent: dict(streak)
        for agent, streak in normalized.get("failure_streaks", {}).items()
        if agent != result.agent
    }
    promoted: SupervisorState = {
        **normalized,
        "pending_result": None,
        "agent_results": agent_results,
        "accepted_evidence": accepted,
        "artifacts": {agent: list(items) for agent, items in accepted.items()},
        "completed_agents": completed,
        "failed_agents": failed,
        "failure_streaks": failure_streaks,
    }
    promoted = _apply_candidate_state_updates(promoted, candidate.get("state_updates") or {})
    pending_approval = promoted.get("pending_approval")
    if isinstance(pending_approval, dict) and pending_approval.get("agent") == result.agent:
        promoted["pending_approval"] = None
        if promoted.get("terminal_state") == SupervisorTerminalState.needs_user_approval.value:
            promoted["terminal_state"] = "running"
    events = list(promoted.get("run_events", []))
    events.append(
        {
            "type": "evidence.promoted",
            "candidate_id": candidate.get("candidate_id"),
            "validation_id": candidate.get("validation_id"),
            "agent": result.agent,
        }
    )
    promoted["run_events"] = events
    return _ensure_json_serializable(promoted)


def reject_pending_result(
    state: SupervisorState,
    reason: str,
    metadata: dict[str, Any] | None = None,
    *,
    event_type: str = "validation.rejected",
) -> SupervisorState:
    normalized = normalize_supervisor_state(state)
    candidate = normalized.get("pending_result")
    if not isinstance(candidate, dict):
        return normalized
    rejected = list(normalized.get("rejected_results", []))
    rejected.append({**candidate, "reason": reason, "metadata": dict(metadata or {})})
    quarantined = list(normalized.get("quarantined_artifacts", []))
    result = candidate.get("result") or {}
    quarantined.extend(result.get("artifacts") or [])
    events = list(normalized.get("run_events", []))
    events.append(
        {
            "type": event_type,
            "candidate_id": candidate.get("candidate_id"),
            "reason": reason,
            **dict(metadata or {}),
        }
    )
    return _ensure_json_serializable(
        {
            **normalized,
            "pending_result": None,
            "rejected_results": rejected,
            "quarantined_artifacts": quarantined,
            "run_events": events,
        }
    )


def normalize_supervisor_state(state: SupervisorState) -> SupervisorState:
    schema_version = int(state.get("state_schema_version", 0) or 0)
    if schema_version >= 2:
        accepted_evidence = {
            agent: list(items) for agent, items in state.get("accepted_evidence", {}).items()
        }
        completed_agents: list[str] = []
        for result in state.get("agent_results", []):
            agent = str(result.get("agent") or "")
            if result.get("status") in {"success", "warning"} and agent and agent not in completed_agents:
                completed_agents.append(agent)
        validation_history = list(state.get("validation_history", []))
        if schema_version == 2 and not validation_history:
            validation_history = _migrate_v2_validation_history(state)
        pending_approval = state.get("pending_approval")
        normalized_pending_approval = None
        if isinstance(pending_approval, dict):
            agent_name = str(pending_approval.get("agent") or "analysis_agent")
            legacy_resume_fields = (
                {
                    "review_request": None,
                    "expected_resume": {"approved": "boolean"},
                }
                if schema_version < 5
                else {}
            )
            normalized_pending_approval = PendingApproval.model_validate(
                {
                    "approval_id": (
                        pending_approval.get("approval_id")
                        or f"{state.get('current_run_id', '')}:{agent_name}:approval"
                    ),
                    "agent": agent_name,
                    "reason": (
                        pending_approval.get("reason")
                        or state.get("final_answer")
                        or "사용자 승인이 필요합니다."
                    ),
                    "approval_type": pending_approval.get("approval_type") or "agent_approval",
                    **pending_approval,
                    **legacy_resume_fields,
                }
            ).model_dump(mode="json")

        # v6: 노드 관련 필드 기본값
        active_node = state.get("active_node")
        normalized_active_node = (
            ActiveNodeExecution.model_validate(active_node).model_dump(mode="json")
            if isinstance(active_node, dict)
            else None
        )

        last_completed_node_id = state.get("last_completed_node_id")
        normalized_last_completed_node_id = (
            str(last_completed_node_id)
            if last_completed_node_id
            else None
        )

        normalized_node_sequence = max(
            int(state.get("node_sequence", 0) or 0),
            0,
        )
        

        normalized: SupervisorState = {
            **state,
            "active_node": normalized_active_node,
            "last_completed_node_id": normalized_last_completed_node_id,
            "node_sequence": normalized_node_sequence,
            "pending_approval": normalized_pending_approval,
            "pending_result": state.get("pending_result"),
            "result_history": list(state.get("result_history", [])),
            "accepted_evidence": accepted_evidence,
            "rejected_results": list(state.get("rejected_results", [])),
            "quarantined_artifacts": list(state.get("quarantined_artifacts", [])),
            "semantic_retry_counts": dict(state.get("semantic_retry_counts", {})),
            "semantic_recovery_attempts": dict(state.get("semantic_recovery_attempts", {})),
            "limitations": list(state.get("limitations", [])),
            "failure_streaks": {
                agent: dict(streak)
                for agent, streak in state.get("failure_streaks", {}).items()
            },
            "run_events": list(state.get("run_events", [])),
            "validation_history": validation_history,
            "state_schema_version": 6,
            "artifacts": {agent: list(items) for agent, items in accepted_evidence.items()},
            "completed_agents": completed_agents,
            "analysis_selection_response": (
                state.get("analysis_selection_response") if schema_version >= 5 else None
            ),
            "analysis_selection_review_request": (
                state.get("analysis_selection_review_request") if schema_version >= 5 else None
            ),
            "analysis_review_decisions": (
                list(state.get("analysis_review_decisions", [])) if schema_version >= 5 else []
            ),
            "analysis_rule_context": (
                state.get("analysis_rule_context") if schema_version >= 6 else None
            ),
            "analysis_rule_retrieval": (
                dict(state.get("analysis_rule_retrieval") or {"status": "not_started"})
                if schema_version >= 6
                else {"status": "not_started"}
            ),
        }
        normalized.pop("validation_results", None)
        normalized.pop("evidence_validation_results", None)
        normalized.pop("semantic_validation_results", None)
        return _ensure_json_serializable(normalized)

    quarantined = list(state.get("quarantined_artifacts", []))
    for agent, artifacts in (state.get("artifacts") or {}).items():
        for artifact in artifacts:
            quarantined.append({"agent": agent, **dict(artifact), "quarantine_reason": "legacy_unvalidated"})
    normalized = {
        **state,
        "active_node": None,
        "last_completed_node_id": None,
        "node_sequence": 0,
        "state_schema_version": 6,
        "validation_history": [],
        "pending_result": None,
        "result_history": list(state.get("result_history", [])),
        "accepted_evidence": {},
        "rejected_results": list(state.get("rejected_results", [])),
        "quarantined_artifacts": quarantined,
        "semantic_retry_counts": dict(state.get("semantic_retry_counts", {})),
        "semantic_recovery_attempts": dict(state.get("semantic_recovery_attempts", {})),
        "limitations": list(state.get("limitations", [])),
        "failure_streaks": {},
        "run_events": list(state.get("run_events", [])),
        "artifacts": {},
        "agent_results": [],
        "completed_agents": [],
        "analysis_selection_response": None,
        "analysis_selection_review_request": None,
        "analysis_review_decisions": [],
        "analysis_rule_context": None,
        "analysis_rule_retrieval": {"status": "not_started"},
    }
    return _ensure_json_serializable(normalized)


def begin_or_retry_agent_node(
    state: SupervisorState,
    agent_name: AgentName,
) -> tuple[
    SupervisorState,
    ActiveNodeExecution,
    NodeActivationEventType,
]:
    normalized = normalize_supervisor_state(state)
    active_node_payload = normalized.get("active_node")

    if isinstance(active_node_payload, dict):
        active_node = ActiveNodeExecution.model_validate(active_node_payload)

        if active_node.agent_name != agent_name:
            raise ValueError(
                "다른 Agent의 active_node가 남아 있습니다. "
                "Recovery 전환이라면 기존 active_node를 먼저 폐기해야 합니다. "
                f"active={active_node.agent_name}, requested={agent_name}"
            )

        if active_node.status == "waiting":
            next_node = active_node.model_copy(
                update={"status": "running"},
            )
            event_type: NodeActivationEventType = "agent.resumed"
        else:
            next_node = active_node.model_copy(
                update={"attempt": active_node.attempt + 1},
            )
            event_type = "agent.retrying"

        updated_state: SupervisorState = {
            **normalized,
            "active_node": next_node.model_dump(mode="json"),
        }
        return (
            _ensure_json_serializable(updated_state),
            next_node,
            event_type,
        )

    current_run_id = str(normalized.get("current_run_id") or "")
    if not current_run_id:
        raise ValueError("Agent 노드를 생성하려면 current_run_id가 필요합니다.")

    next_sequence = int(normalized.get("node_sequence", 0) or 0) + 1
    next_node = ActiveNodeExecution(
        node_id=f"{current_run_id}:node:{next_sequence}",
        agent_name=agent_name,
        parent_node_id=normalized.get("last_completed_node_id"),
        node_sequence=next_sequence,
        attempt=1,
        status="running",
    )

    updated_state = {
        **normalized,
        "active_node": next_node.model_dump(mode="json"),
        "node_sequence": next_sequence,
    }
    return (
        _ensure_json_serializable(updated_state),
        next_node,
        "agent.started",
    )


def discard_active_node_for_recovery(
    state: SupervisorState,
    *,
    next_agent_name: AgentName,
    reason: str,
) -> tuple[
    SupervisorState,
    ActiveNodeExecution,
    Literal["agent.discarded"],
]:
    normalized = normalize_supervisor_state(state)
    active_node_payload = normalized.get("active_node")

    if not isinstance(active_node_payload, dict):
        raise ValueError("폐기할 active_node가 없습니다.")

    active_node = ActiveNodeExecution.model_validate(active_node_payload)

    if active_node.agent_name == next_agent_name:
        raise ValueError(
            "같은 Agent의 재시도에는 노드를 폐기할 수 없습니다. "
            "begin_or_retry_agent_node()를 사용해야 합니다."
        )

    if not reason.strip():
        raise ValueError("노드 폐기 사유가 필요합니다.")

    updated_state: SupervisorState = {
        **normalized,
        "active_node": None,
    }

    return (
        _ensure_json_serializable(updated_state),
        active_node,
        "agent.discarded",
    )


def wait_active_node(
    state: SupervisorState,
    *,
    agent_name: AgentName,
    reason: str,
) -> tuple[
    SupervisorState,
    ActiveNodeExecution,
    Literal["agent.waiting"],
]:
    normalized = normalize_supervisor_state(state)
    active_node_payload = normalized.get("active_node")

    if not isinstance(active_node_payload, dict):
        raise ValueError("대기 상태로 전환할 active_node가 없습니다.")

    active_node = ActiveNodeExecution.model_validate(active_node_payload)

    if active_node.agent_name != agent_name:
        raise ValueError(
            "active_node의 Agent와 대기할 Agent가 다릅니다. "
            f"active={active_node.agent_name}, waiting={agent_name}"
        )

    if not reason.strip():
        raise ValueError("노드 대기 사유가 필요합니다.")

    waiting_node = active_node.model_copy(
        update={"status": "waiting"},
    )

    updated_state: SupervisorState = {
        **normalized,
        "active_node": waiting_node.model_dump(mode="json"),
    }

    return (
        _ensure_json_serializable(updated_state),
        waiting_node,
        "agent.waiting",
    )


def fail_active_node(
    state: SupervisorState,
    *,
    agent_name: AgentName,
    reason: str,
    reason_code: str = "",
) -> tuple[
    SupervisorState,
    FailedNodeExecution,
    Literal["agent.failed"],
]:
    normalized = normalize_supervisor_state(state)
    active_node_payload = normalized.get("active_node")

    if not isinstance(active_node_payload, dict):
        raise ValueError("실패 처리할 active_node가 없습니다.")

    active_node = ActiveNodeExecution.model_validate(active_node_payload)

    if active_node.agent_name != agent_name:
        raise ValueError(
            "active_node의 Agent와 실패 처리할 Agent가 다릅니다. "
            f"active={active_node.agent_name}, failed={agent_name}"
        )

    if not reason.strip():
        raise ValueError("노드 실패 사유가 필요합니다.")

    failed_node = FailedNodeExecution(
        node_id=active_node.node_id,
        agent_name=active_node.agent_name,
        parent_node_id=active_node.parent_node_id,
        node_sequence=active_node.node_sequence,
        attempt=active_node.attempt,
        reason=reason,
        reason_code=reason_code,
    )

    updated_state: SupervisorState = {
        **normalized,
        "active_node": None,
    }

    return (
        _ensure_json_serializable(updated_state),
        failed_node,
        "agent.failed",
    )


def complete_active_node(
    state: SupervisorState,
    *,
    agent_name: AgentName,
    step_summary: StepSummary,
) -> tuple[
    SupervisorState,
    CompletedNodeExecution,
    Literal["agent.completed"],
]:
    normalized = normalize_supervisor_state(state)
    active_node_payload = normalized.get("active_node")

    if not isinstance(active_node_payload, dict):
        raise ValueError("완료할 active_node가 없습니다.")

    active_node = ActiveNodeExecution.model_validate(active_node_payload)

    if active_node.agent_name != agent_name:
        raise ValueError(
            "active_node의 Agent와 완료할 Agent가 다릅니다. "
            f"active={active_node.agent_name}, completed={agent_name}"
        )

    if agent_name not in normalized.get("completed_agents", []):
        raise ValueError(
            "결과가 아직 accepted evidence로 승격되지 않았습니다."
        )

    if step_summary.agent != agent_name:
        raise ValueError("StepSummary의 Agent가 active_node와 다릅니다.")

    if not step_summary.summary.strip():
        raise ValueError("완료 노드에는 비어 있지 않은 summary가 필요합니다.")

    summary_payload = step_summary.model_dump(mode="json")
    if summary_payload not in normalized.get("step_summaries", []):
        raise ValueError("StepSummary가 SupervisorState에 저장되지 않았습니다.")

    completed_node = CompletedNodeExecution(
        node_id=active_node.node_id,
        agent_name=active_node.agent_name,
        parent_node_id=active_node.parent_node_id,
        node_sequence=active_node.node_sequence,
        attempt=active_node.attempt,
        summary=step_summary,
    )

    updated_state: SupervisorState = {
        **normalized,
        "active_node": None,
        "last_completed_node_id": completed_node.node_id,
    }

    return (
        _ensure_json_serializable(updated_state),
        completed_node,
        "agent.completed",
    )


def _migrate_v2_validation_history(state: SupervisorState) -> list[dict[str, Any]]:
    hard_results = list(state.get("validation_results", []))
    evidence_results = list(state.get("evidence_validation_results", []))
    semantic_results = list(state.get("semantic_validation_results", []))
    records: list[dict[str, Any]] = []
    for index, hard in enumerate(hard_results):
        legacy_id = f"legacy_validation_{index}"
        disposition = str(hard.get("decision") or ("accept" if hard.get("valid") else "reject"))
        if disposition not in {
            "accept", "accept_with_limitations", "retry", "await_approval", "reject"
        }:
            disposition = "reject"
        checks = [
            {
                "name": "result",
                "passed": bool(hard.get("hard_valid", hard.get("valid"))),
                "findings": list(hard.get("findings", [])),
                "details": dict(hard),
            }
        ]
        if index < len(evidence_results):
            evidence = evidence_results[index]
            checks.append(
                {
                    "name": "evidence",
                    "passed": bool(evidence.get("valid")),
                    "findings": list(evidence.get("findings", [])),
                    "details": dict(evidence),
                }
            )
        if index < len(semantic_results):
            semantic = semantic_results[index]
            checks.append(
                {
                    "name": "semantic",
                    "passed": bool(semantic.get("semantic_valid")),
                    "findings": [],
                    "details": dict(semantic),
                }
            )
        candidate_id = str(hard.get("candidate_id") or legacy_id)
        records.append(
            {
                "candidate_id": candidate_id,
                "validation_id": str(hard.get("validation_id") or legacy_id),
                "agent": str(hard.get("agent") or "sql_agent"),
                "outcome": {
                    "disposition": disposition,
                    "reason": str(hard.get("reason") or ""),
                    "reason_code": str(hard.get("reason_code") or "none"),
                    "retry_target": hard.get("agent") if disposition == "retry" else None,
                    "terminal_state": str(hard.get("terminal_state") or "running"),
                },
                "checks": checks,
            }
        )
    return records


def _apply_candidate_state_updates(state: SupervisorState, updates: dict[str, Any]) -> SupervisorState:
    merged: SupervisorState = dict(state)
    if updates.get("generated_sql"):
        merged["generated_sql"] = str(updates["generated_sql"])
    if "error_state" in updates:
        merged["error_state"] = dict(updates.get("error_state") or {})
    if "analysis_plan" in updates:
        plan = dict(merged.get("analysis_plan") or {})
        incoming = dict(updates.get("analysis_plan") or {})
        for key in ("generated_sql", "source_sql"):
            if key in incoming and not incoming[key]:
                incoming.pop(key)
        plan.update(incoming)
        merged["analysis_plan"] = plan
    if updates.get("planner_mode"):
        plan = dict(merged.get("analysis_plan") or {})
        plan["planner_mode"] = str(updates["planner_mode"])
        merged["analysis_plan"] = plan
    return merged


def artifact_ids_by_agent(state: SupervisorState) -> dict[str, list[str]]:
    if int(state.get("state_schema_version", 0) or 0) < 2:
        return {}
    ids: dict[str, list[str]] = {}

    def append_unique(agent: str, artifact_id: Any) -> None:
        artifact_id_value = str(artifact_id)
        if not artifact_id_value:
            return
        ids.setdefault(agent, [])
        if artifact_id_value not in ids[agent]:
            ids[agent].append(artifact_id_value)

    for result in state.get("agent_results", []):
        agent = str(result.get("agent", ""))
        if not agent:
            continue
        for artifact_id in result.get("artifact_ids", []):
            append_unique(agent, artifact_id)
    evidence = state.get("accepted_evidence", state.get("artifacts", {}))
    for agent, artifacts in evidence.items():
        agent_name = str(agent)
        if not agent_name:
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            append_unique(agent_name, artifact.get("artifact_id", ""))
    return ids


def to_orchestration_state(state: SupervisorState) -> OrchestrationState:
    state = normalize_supervisor_state(state)
    terminal_value = state.get("terminal_state")
    terminal_state = None
    if terminal_value and terminal_value != "running":
        try:
            terminal_state = SupervisorTerminalState(str(terminal_value))
        except ValueError as exc:
            valid_values = ", ".join(["running", *(item.value for item in SupervisorTerminalState)])
            raise ValueError(
                f"Invalid supervisor terminal_state: {terminal_value!r}. Expected one of: {valid_values}"
            ) from exc

    plan_payload = state.get("analysis_plan") or {}
    goal = str(plan_payload.get("goal") or state.get("latest_user_query") or "")
    route_kind = str(plan_payload.get("route_kind") or "simple")
    planner_mode = "llm" if plan_payload.get("planner_mode") == "llm" else "deterministic"
    generated_sql = str(state.get("generated_sql") or "")
    pending_approval = state.get("pending_approval")
    approval_ids: list[str] = []
    if isinstance(pending_approval, dict):
        approval_id = str(pending_approval.get("approval_id") or "")
        if approval_id:
            approval_ids.append(approval_id)
    _error_state = state.get("error_state") or {}
    _retry_context: dict = dict(state.get("retry_counts") or {})
    if _error_state:
        _retry_context.update(
            {
                "last_error": _error_state.get("message", ""),
                "retryable": _error_state.get("retryable", False),
            }
        )
    analysis_failure = (state.get("failure_streaks") or {}).get("analysis_agent")
    if isinstance(analysis_failure, dict):
        _retry_context["last_failure"] = {
            "reason_code": str(analysis_failure.get("reason_code") or "none"),
            "failure_reason": str(analysis_failure.get("failure_reason") or ""),
        }
    _agent_feedback: dict[str, dict[str, Any]] = {}
    for record in state.get("validation_history", []):
        agent = str(record.get("agent") or "")
        outcome = record.get("outcome") or {}
        if not agent or outcome.get("disposition") not in {"recover", "reject"}:
            continue
        semantic_check = next(
            (c for c in record.get("checks", []) if c.get("name") == "semantic"), None
        )
        details = (semantic_check or {}).get("details") or {}
        _agent_feedback[agent] = {
            "reason": str(outcome.get("reason") or ""),
            "missing_evidence": list(details.get("missing_evidence") or []),
            "source": "semantic" if semantic_check else "hard_failure",
        }
    if _agent_feedback:
        _retry_context["agent_feedback"] = _agent_feedback
    plan: AnalysisPlan | None = None
    if plan_payload:
        source_sql = str(plan_payload.get("source_sql") or generated_sql) if generated_sql else ""
        plan = AnalysisPlan(
            goal=goal,
            datasource_id=state.get("datasource_id"),
            catalog_summary=state.get("catalog_summary"),
            retry_context=_retry_context,
            planner_mode=planner_mode,
            metric=plan_payload.get("metric") or None,
            dimension=plan_payload.get("dimension") or None,
            filters=[str(item) for item in (plan_payload.get("filters") or []) if str(item).strip()],
            requires_mart_review=bool(plan_payload.get("requires_mart_review", False)),
            query_rules=dict(plan_payload.get("query_rules") or {}),
            route_kind=route_kind,
            generated_sql=generated_sql,
            source_sql=source_sql,
            target_table=plan_payload.get("target_table") or None,
            source_tables=[str(t) for t in (plan_payload.get("source_tables") or []) if t],
            business_grain=plan_payload.get("business_grain") or None,
        )
    limitations = [str(item) for item in state.get("limitations", []) if item]
    limitations.extend(
        str(finding.get("message") or "")
        for result in state.get("agent_results", [])
        for finding in result.get("findings", [])
        if finding.get("disposition") == "limitation" and finding.get("message")
    )

    return OrchestrationState(
        run_id=state["current_run_id"],
        thread_id=state.get("thread_id"),
        datasource_id=state.get("datasource_id"),
        catalog_summary=state.get("catalog_summary"),
        user_query=state.get("clarified_query") or state.get("latest_user_query", ""),
        final_answer=str(state.get("final_answer") or ""),
        goal=goal,
        plan=plan,
        retry_context=_retry_context,
        current_step=state.get("current_step", "created"),
        artifact_ids=artifact_ids_by_agent(state),
        approval_ids=approval_ids,
        error_state=dict(state.get("error_state", {})),
        terminal_state=terminal_state,
        remaining_agents=[],
        completed_agents=list(state.get("completed_agents", [])),
        last_agent=(state.get("completed_agents") or [None])[-1],
        route_kind=route_kind,
        planner_mode=planner_mode,
        generated_sql=generated_sql,
        retry_counts=dict(state.get("retry_counts", {})),
        max_retry_per_agent=int(state.get("max_retry_per_agent", 1)),
        limitations=list(dict.fromkeys(limitations)),
        analysis_review_decisions=list(state.get("analysis_review_decisions", [])),
    )
