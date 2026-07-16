from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    SupervisorInterruptPayload,
    SupervisorTerminalState,
    ValidationFinding,
)
from DATA_Analyst_Assistant_Agent.supervisor.analysis_review import (
    validate_analysis_review_resume,
)
from DATA_Analyst_Assistant_Agent.supervisor.decision import (
    AnalysisPlanDecision,
    ClarificationDecision,
    FinalizationDecision,
    SemanticValidationAdvisoryDecision,
    SupervisorDecision,
    build_clarification_context,
    build_finalization_context,
    build_next_action_context,
    build_plan_context,
    build_result_validation_context,
    invoke_supervisor_decision,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import (
    CLARIFY_DECISION_PROMPT,
    DECIDE_NEXT_ACTION_PROMPT,
    FINALIZE_DECISION_PROMPT,
    PLAN_DECISION_PROMPT,
    SEMANTIC_VALIDATION_ADVISORY_PROMPT,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    PendingApproval,
    StepSummary,
    SupervisorState,
    artifact_ids_by_agent,
    promote_pending_result,
    reject_pending_result,
    stage_candidate_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentContractError, AgentToolResult
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    ValidationCheckResult,
    ValidationOutcome,
    ValidationRecord,
    _check_completion_readiness,
    contract_check_from_decision,
    outcome_from_contract_decision,
    validate_subagent_result as validate_subagent_result_contract,
)
from DATA_Analyst_Assistant_Agent.supervisor.candidate import (
    commit_candidate,
    validate_candidate,
)




def _has_specific_analysis_intent(query: str) -> bool:
    q = " ".join(str(query or "").split()).lower()
    if not q:
        return False
    specific_tokens = [
        "요약", "월별", "추이", "차트", "프로파일", "품질", "데이터마트", "마트", "분석", "지난", "최근", "올해", "작년",
        "summary", "monthly", "trend", "chart", "profile", "quality", "datamart", "analysis", "last ", "recent", "this year", "yearly",
    ]
    if any(token in q for token in specific_tokens):
        return True
    words = [part for part in q.split() if part]
    return len(words) >= 4

ACTION_TO_AGENT: dict[NextAction, AgentName] = {
    "call_sql_agent": "sql_agent",
    "call_eda_agent": "eda_agent",
    "call_analysis_agent": "analysis_agent",
    "call_report_agent": "report_agent",
    "call_insight_agent": "insight_agent",
}

_NODE_ONLY_AGENTS = {"report_agent", "insight_agent"}

SUBAGENT_ACTION_TO_AGENT: dict[NextAction, AgentName] = {
    action: agent
    for action, agent in ACTION_TO_AGENT.items()
    if agent not in _NODE_ONLY_AGENTS
}

TERMINAL_STATES = {item.value for item in SupervisorTerminalState}
FINALIZE_PROTECTED_TERMINAL_STATES = {
    SupervisorTerminalState.failed_terminal.value,
    SupervisorTerminalState.needs_user_approval.value,
    SupervisorTerminalState.needs_clarification.value,
    SupervisorTerminalState.failed_with_recoverable_context.value,
}


def make_clarify_query_node(model: Any | None):
    def clarify_query_node(state: SupervisorState) -> SupervisorState:
        latest_query = str(state.get("clarified_query") or state.get("latest_user_query") or "")
        try:
            decision = invoke_supervisor_decision(
                state,
                model,
                CLARIFY_DECISION_PROMPT,
                ClarificationDecision,
                extra=build_clarification_context(state),
            )
        except Exception as exc:
            return _decision_failure_updates(state, "clarify_query", exc)

        updates: SupervisorState = {
            "clarified_query": decision.clarified_query,
            "needs_clarification": decision.needs_clarification,
            "clarification_question": decision.clarification_question,
            "current_step": "clarify_query",
            "llm_decisions": _append_llm_decision(state, "clarify_query", decision),
        }
        if decision.needs_clarification:
            updates["terminal_state"] = "running"
            updates["next_action"] = "clarify"
        else:
            updates["terminal_state"] = "running"
            updates["next_action"] = "create_plan"
        return updates

    return clarify_query_node


def make_collect_clarification_node():
    def collect_clarification_node(state: SupervisorState) -> SupervisorState:
        question = state.get("clarification_question") or "분석을 진행하기 위해 추가 정보가 필요합니다."
        payload = {
            "type": "clarification",
            "status": "waiting_input",
            "run_id": state["current_run_id"],
            "thread_id": state["thread_id"],
            "question": question,
            "node": "collect_clarification",
            "expected_resume": {"answer": "string"},
        }
        resume_value = interrupt(payload)
        answer = _clarification_answer_from_resume(resume_value)
        base_query = state.get("clarified_query") or state.get("latest_user_query", "")
        clarified_query = _clarified_query_from_answer(base_query, answer)
        return {
            "clarified_query": clarified_query,
            "needs_clarification": False,
            "clarification_question": "",
            "terminal_state": "running",
            "next_action": "create_plan",
            "current_step": "collect_clarification",
        }

    return collect_clarification_node


def make_collect_analysis_review_node():
    def collect_analysis_review_node(state: SupervisorState) -> SupervisorState:
        pending_approval = state.get("pending_approval")
        pending_result = state.get("pending_result")
        if not isinstance(pending_approval, dict) or not isinstance(pending_result, dict):
            raise ValueError("analysis review에 필요한 pending approval/candidate가 없습니다.")
        review_request = pending_approval.get("review_request")
        if not isinstance(review_request, dict):
            raise ValueError("analysis review request가 없습니다.")
        payload = SupervisorInterruptPayload(
            type="analysis_review",
            status="waiting_input",
            run_id=state["current_run_id"],
            thread_id=state["thread_id"],
            question=str(review_request.get("question") or pending_approval.get("reason") or ""),
            node="collect_analysis_review",
            approval_id=str(pending_approval.get("approval_id") or ""),
            review_request=review_request,
            expected_resume=dict(pending_approval.get("expected_resume") or {}),
        ).model_dump(mode="json")
        resume_value = interrupt(payload)
        decision = validate_analysis_review_resume(
            resume_value,
            pending_approval=pending_approval,
            pending_candidate_id=str(pending_result.get("candidate_id") or ""),
            review_request=review_request,
        )
        return {
            "analysis_selection_response": decision.selection_response.model_dump(mode="json"),
            "current_step": "collect_analysis_review",
        }

    return collect_analysis_review_node


def make_resolve_analysis_review_node(backend_adapter: Any | None = None):
    def resolve_analysis_review_node(state: SupervisorState) -> SupervisorState:
        pending_approval = state.get("pending_approval")
        pending_result = state.get("pending_result")
        selection = state.get("analysis_selection_response")
        if not isinstance(pending_approval, dict) or not isinstance(pending_result, dict):
            return _terminal_failure_updates(
                state,
                "resolve_analysis_review",
                "analysis review에 필요한 pending approval/candidate가 없습니다.",
            )
        if not isinstance(selection, dict):
            return _terminal_failure_updates(
                state,
                "resolve_analysis_review",
                "analysis review selection이 없습니다.",
            )
        decision = validate_analysis_review_resume(
            {"approval_id": pending_approval.get("approval_id"), **selection},
            pending_approval=pending_approval,
            pending_candidate_id=str(pending_result.get("candidate_id") or ""),
            review_request=pending_approval.get("review_request") or {},
        )
        decisions = list(state.get("analysis_review_decisions", []))
        decision_payload = decision.model_dump(mode="json")
        if not any(
            item.get("approval_id") == decision.approval_id
            and item.get("candidate_id") == decision.candidate_id
            for item in decisions
        ):
            decisions.append(decision_payload)

        if decision.review_request.requires_followup_analysis:
            previous_quarantine_count = len(state.get("quarantined_artifacts", []))
            rejected = reject_pending_result(
                state,
                "사용자 선택에 따라 기존 분석 후보를 재분석 대상으로 대체했습니다.",
                metadata={
                    "agent": "analysis_agent",
                    "reason_code": "analysis.review_superseded",
                    "disposition": "superseded",
                },
                event_type="analysis.review_superseded",
            )
            quarantined = list(rejected.get("quarantined_artifacts", []))
            rejected["quarantined_artifacts"] = [
                (
                    artifact
                    if index < previous_quarantine_count
                    else {
                        **artifact,
                        "quarantine_reason": "analysis.review_superseded",
                    }
                )
                for index, artifact in enumerate(quarantined)
            ]
            rejected.update(
                {
                    "pending_approval": None,
                    "pending_validation": None,
                    "analysis_selection_response": decision.selection_response.model_dump(mode="json"),
                    "analysis_selection_review_request": decision.review_request.model_dump(mode="json"),
                    "analysis_review_decisions": decisions,
                    "terminal_state": "running",
                    "next_action": "call_analysis_agent",
                    "final_answer": "",
                    "current_step": "resolve_analysis_review",
                }
            )
            return rejected

        promoted = commit_candidate(state, backend_adapter, approval_granted=True)
        if promoted.get("next_action") == "call_analysis_agent":
            promoted["next_action"] = "decide_next_action"
        promoted.update(
            {
                "pending_approval": None,
                "analysis_selection_response": None,
                "analysis_selection_review_request": None,
                "analysis_review_decisions": decisions,
                "terminal_state": "running",
                "final_answer": "",
                "current_step": "resolve_analysis_review",
            }
        )
        return promoted

    return resolve_analysis_review_node


def _clarified_query_from_answer(base_query: Any, answer: Any) -> str:
    base = _one_line_text(base_query)
    refined_answer = _one_line_text(answer)
    if not refined_answer:
        return base
    if not base:
        return refined_answer
    if _answer_can_stand_alone(base, refined_answer):
        return refined_answer
    return f"{refined_answer} 기준으로 {base}"


def _answer_can_stand_alone(base_query: str, answer: str) -> bool:
    if not answer:
        return False
    if base_query and base_query in answer:
        return True
    if len(base_query) > 12:
        return False
    base_terms = [term.strip() for term in base_query.split() if len(term.strip()) >= 2]
    return any(term in answer for term in base_terms)


def _one_line_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def make_create_analysis_plan_node(model: Any | None):
    def create_analysis_plan_node(state: SupervisorState) -> SupervisorState:
        try:
            decision = invoke_supervisor_decision(
                state,
                model,
                PLAN_DECISION_PROMPT,
                AnalysisPlanDecision,
                extra=build_plan_context(state),
            )
        except Exception as exc:
            return _decision_failure_updates(state, "create_analysis_plan", exc)

        plan: dict[str, Any] = {
            "goal": decision.goal,
            "route_kind": decision.route_kind,
            "planner_mode": "llm",
            "steps": list(decision.steps),
            "metric": decision.metric,
            "dimension": decision.dimension,
            "filters": list(decision.filters),
            "requires_mart_review": decision.requires_mart_review,
        }
        if state.get("datasource_id") is not None:
            plan["datasource_id"] = state.get("datasource_id")
        if state.get("catalog_summary") is not None:
            plan["catalog_summary"] = state.get("catalog_summary")

        return {
            "analysis_plan": plan,
            "current_step": "create_analysis_plan",
            "llm_decisions": _append_llm_decision(state, "create_analysis_plan", decision),
        }

    return create_analysis_plan_node


def make_decide_next_action_node(model: Any | None):
    def decide_next_action_node(state: SupervisorState) -> SupervisorState:
        if state.get("terminal_state") in TERMINAL_STATES:
            return {"next_action": "finalize", "current_step": "decide_next_action"}

        try:
            decision = invoke_supervisor_decision(
                state,
                model,
                DECIDE_NEXT_ACTION_PROMPT,
                SupervisorDecision,
                extra=build_next_action_context(state),
            )
        except Exception as exc:
            return _decision_failure_updates(state, "decide_next_action", exc)

        updates: SupervisorState = {
            "next_action": decision.next_action,
            "current_step": "decide_next_action",
            "llm_decisions": _append_llm_decision(state, "decide_next_action", decision),
        }
        if decision.next_action == "fail":
            reason = f"Supervisor가 명시적으로 fail을 선택했습니다: {decision.reason}"
            updates.update(
                {
                    "terminal_state": SupervisorTerminalState.failed_terminal.value,
                    "next_action": "finalize",
                    "final_answer": reason,
                }
            )
        return updates

    return decide_next_action_node


def make_completion_guard_node():
    def completion_guard_node(state: SupervisorState) -> SupervisorState:
        current_terminal_state = state.get("terminal_state")
        if current_terminal_state in FINALIZE_PROTECTED_TERMINAL_STATES:
            return {
                "next_action": "finalize",
                "current_step": "completion_guard",
            }

        decision = _check_completion_readiness(state)
        if decision.status == "ready":
            return {
                "next_action": "finalize",
                "current_step": "completion_guard",
            }
        if decision.status == "insight_required":
            return {
                "next_action": "call_insight_agent",
                "current_step": "completion_guard",
            }
        if decision.status == "report_required":
            return {
                "next_action": "call_report_agent",
                "current_step": "completion_guard",
            }
        return {
            "terminal_state": SupervisorTerminalState.failed_terminal.value,
            "next_action": "finalize",
            "final_answer": decision.reason,
            "current_step": "completion_guard",
        }

    return completion_guard_node


def make_execute_subagent_node(subagent_adapter: Any, model: Any | None):
    def execute_subagent_node(state: SupervisorState) -> SupervisorState:
        if state.get("terminal_state") in TERMINAL_STATES:
            return {"next_action": "finalize", "current_step": "execute_subagent"}
        if state.get("pending_approval") is not None:
            return {"next_action": "finalize", "current_step": "execute_subagent"}
        if state.get("pending_result") is not None:
            return _terminal_failure_updates(
                state,
                "execute_subagent",
                "검증되지 않은 pending_result가 남아 있어 새 agent를 실행할 수 없습니다.",
            )

        action = state.get("next_action")
        if action == "call_report_agent":
            return {
                "next_action": "call_report_agent",
                "current_step": "execute_subagent",
            }
        if action == "call_insight_agent":
            return {
                "next_action": "call_insight_agent",
                "current_step": "execute_subagent",
            }

        agent_name = SUBAGENT_ACTION_TO_AGENT.get(action)
        if agent_name is None:
            return _terminal_failure_updates(
                state,
                "execute_subagent",
                f"지원하지 않는 subagent action입니다: {action}",
            )

        try:
            tool_result: AgentToolResult = subagent_adapter.call(agent_name, state)
        except AgentContractError as exc:
            message = f"{agent_name} 결과의 에이전트 계약 검증에 실패했습니다: {exc}"
            failed_agents = list(state.get("failed_agents", []))
            if agent_name not in failed_agents:
                failed_agents.append(agent_name)
            return _terminal_failure_updates(
                state,
                "execute_subagent",
                message,
                extra_updates={
                    "pending_result": None,
                    "last_agent_result": {},
                    "failed_agents": failed_agents,
                    "completed_agents": list(state.get("completed_agents", [])),
                    "accepted_evidence": {
                        agent: list(items)
                        for agent, items in state.get("accepted_evidence", {}).items()
                    },
                    "error_state": {
                        "node": "execute_subagent",
                        "message": message,
                        "reason_code": "agent_contract_mismatch",
                        "retryable": False,
                    },
                },
            )

        updates = stage_candidate_result(state, tool_result.agent_result, tool_result.state_updates)
        updates["last_agent_result"] = tool_result.agent_result.model_dump(mode="json")
        updates["current_step"] = "execute_subagent"
        updates["next_action"] = action
        _emit_run_event(
            getattr(subagent_adapter, "backend_adapter", None),
            state,
            "result.staged",
            f"{agent_name} 후보 결과를 격리했습니다.",
            metadata={"candidate_id": (updates.get("pending_result") or {}).get("candidate_id")},
        )
        return updates

    return execute_subagent_node


def make_commit_candidate_node(backend_adapter: Any | None = None):
    def commit_candidate_node(state: SupervisorState) -> SupervisorState:
        return commit_candidate(state, backend_adapter)

    return commit_candidate_node


def make_generate_report_node(report_generator: Any):
    def generate_report_node(state: SupervisorState) -> SupervisorState:
        evidence_ids = artifact_ids_by_agent(state)
        has_evidence = any(
            bool(evidence_ids.get(agent_name))
            for agent_name in ("sql_agent", "eda_agent", "analysis_agent")
        )
        if not has_evidence:
            result = _failed_report_result(
                "리포트를 생성하려면 SQL, EDA, 분석 중 하나 이상의 근거 아티팩트가 필요합니다."
            )
        else:
            try:
                result = report_generator.generate(state)
            except Exception as exc:
                result = _failed_report_result(f"보고서 생성 또는 저장에 실패했습니다: {exc}")

        if result.agent != "report_agent":
            result = _failed_report_result(
                f"보고서 생성기가 잘못된 agent 결과를 반환했습니다: {result.agent}"
            )

        updates: SupervisorState = {
            **stage_candidate_result(state, result, {}),
            "last_agent_result": result.model_dump(mode="json"),
            "terminal_state": "running",
            "next_action": "call_report_agent",
            "current_step": "generate_report",
        }
        _emit_run_event(
            getattr(report_generator, "backend_adapter", None),
            state,
            "result.staged",
            "report_agent 후보 결과를 격리했습니다.",
            metadata={"candidate_id": (updates.get("pending_result") or {}).get("candidate_id")},
        )
        return updates

    return generate_report_node


def _failed_report_result(message: str) -> AgentCompactResult:
    return AgentCompactResult(
        agent="report_agent",
        status="failed",
        summary=message,
        retryable=False,
        error=message,
    )


class _InsightGeneratorAdapter:
    """subagent_adapter가 노출하는 generate_insight()를 generate_insight_node가 기대하는
    .generate() 인터페이스로 맞춰준다 (report_agent의 .generate()와 이름이 겹치지 않도록)."""

    def __init__(self, subagent_adapter: Any) -> None:
        self._subagent_adapter = subagent_adapter
        self.backend_adapter = getattr(subagent_adapter, "backend_adapter", None)

    def generate(self, state: SupervisorState) -> AgentCompactResult:
        return self._subagent_adapter.generate_insight(state)


def make_generate_insight_node(insight_generator: Any):
    def generate_insight_node(state: SupervisorState) -> SupervisorState:
        evidence_ids = artifact_ids_by_agent(state)
        has_evidence = any(
            bool(evidence_ids.get(agent_name))
            for agent_name in ("sql_agent", "eda_agent", "analysis_agent")
        )
        if not has_evidence:
            result = _failed_insight_result(
                "인사이트를 생성하려면 SQL, EDA, 분석 중 하나 이상의 근거 아티팩트가 필요합니다."
            )
        else:
            try:
                result = insight_generator.generate(state)
            except Exception as exc:
                result = _failed_insight_result(f"인사이트 생성 또는 저장에 실패했습니다: {exc}")

        if result.agent != "insight_agent":
            result = _failed_insight_result(
                f"인사이트 생성기가 잘못된 agent 결과를 반환했습니다: {result.agent}"
            )

        updates: SupervisorState = {
            **stage_candidate_result(state, result, {}),
            "last_agent_result": result.model_dump(mode="json"),
            "terminal_state": "running",
            "next_action": "call_insight_agent",
            "current_step": "generate_insight",
        }
        _emit_run_event(
            getattr(insight_generator, "backend_adapter", None),
            state,
            "result.staged",
            "insight_agent 후보 결과를 격리했습니다.",
            metadata={"candidate_id": (updates.get("pending_result") or {}).get("candidate_id")},
        )
        return updates

    return generate_insight_node


def _failed_insight_result(message: str) -> AgentCompactResult:
    return AgentCompactResult(
        agent="insight_agent",
        status="failed",
        summary=message,
        retryable=False,
        error=message,
    )


def make_stage_candidate_node():
    def stage_candidate_node(state: SupervisorState) -> SupervisorState:
        pending = state.get("pending_result")
        if not isinstance(pending, dict) or not isinstance(pending.get("result"), dict):
            return _terminal_failure_updates(state, "stage_candidate", "격리할 후보 결과가 없습니다.")
        return {"current_step": "stage_candidate"}

    return stage_candidate_node


def make_validate_candidate_node(model: Any | None):
    def validate_candidate_node(state: SupervisorState) -> SupervisorState:
        return validate_candidate(state, model)

    return validate_candidate_node


def _validation_candidate_updates(
    state: SupervisorState,
    result: AgentCompactResult,
    checks: list[ValidationCheckResult],
    outcome: ValidationOutcome,
) -> SupervisorState:
    pending = state.get("pending_result") or {}
    record = ValidationRecord(
        candidate_id=str(pending.get("candidate_id") or ""),
        validation_id=str(pending.get("validation_id") or ""),
        agent=result.agent,
        outcome=outcome,
        checks=checks,
    )
    return {
        "pending_validation": record.model_dump(mode="json"),
        "current_step": "validate_candidate",
    }


def make_resolve_validation_node(backend_adapter: Any | None = None):
    def resolve_validation_node(state: SupervisorState) -> SupervisorState:
        payload = state.get("pending_validation")
        if not isinstance(payload, dict):
            return _terminal_failure_updates(
                state,
                "resolve_validation",
                "해결할 통합 검증 결과가 없습니다.",
            )
        record = ValidationRecord.model_validate(payload)
        history = list(state.get("validation_history", []))
        history.append(record.model_dump(mode="json"))
        updates: SupervisorState = {
            "validation_history": history,
            "pending_validation": None,
            "current_step": "resolve_validation",
        }

        result_check = next((check for check in record.checks if check.name == "result"), None)
        failure_streak = result_check.details.get("failure_streak") if result_check else None
        if isinstance(failure_streak, dict):
            streaks = {agent: dict(item) for agent, item in state.get("failure_streaks", {}).items()}
            streaks[record.agent] = dict(failure_streak)
            updates["failure_streaks"] = streaks

        disposition = record.outcome.disposition
        if disposition in {"accept", "accept_with_limitations", "await_approval"}:
            updates["terminal_state"] = "running"
            return updates

        rejected = reject_pending_result(
            {**state, **updates},
            record.outcome.reason,
            metadata={
                "agent": record.agent,
                "reason_code": record.outcome.reason_code,
                "disposition": disposition,
            },
        )
        updates.update(rejected)
        updates["pending_validation"] = None
        updates["current_step"] = "resolve_validation"
        if disposition == "retry":
            retry_counts = dict(state.get("retry_counts", {}))
            retry_counts[record.agent] = int(retry_counts.get(record.agent, 0)) + 1
            updates["retry_counts"] = retry_counts
            updates["next_action"] = _action_for_agent(record.agent)
            updates["terminal_state"] = "running"
        else:
            failed = list(updates.get("failed_agents", state.get("failed_agents", [])))
            if record.agent not in failed:
                failed.append(record.agent)
            updates["failed_agents"] = failed
            updates["next_action"] = "finalize"
            updates["terminal_state"] = record.outcome.terminal_state
            updates["final_answer"] = record.outcome.reason
        _emit_run_event(
            backend_adapter,
            state,
            "validation.rejected",
            record.outcome.reason,
            metadata={
                "candidate_id": record.candidate_id,
                "validation_id": record.validation_id,
                "agent": record.agent,
                "reason_code": record.outcome.reason_code,
                "disposition": disposition,
            },
        )
        return updates

    return resolve_validation_node


def _action_for_agent(agent: AgentName) -> NextAction:
    return {
        "sql_agent": "call_sql_agent",
        "eda_agent": "call_eda_agent",
        "analysis_agent": "call_analysis_agent",
        "report_agent": "call_report_agent",
        "insight_agent": "call_insight_agent",
    }[agent]


def make_resolve_candidate_node(backend_adapter: Any | None = None):
    def resolve_candidate_node(state: SupervisorState) -> SupervisorState:
        if state.get("terminal_state") in TERMINAL_STATES:
            return {"current_step": "resolve_candidate"}
        pending = state.get("pending_result")
        if not isinstance(pending, dict):
            return _terminal_failure_updates(state, "resolve_candidate", "승격할 후보 결과가 없습니다.")
        result = AgentCompactResult.model_validate(pending.get("result") or {})
        if result.approval.required:
            approval = PendingApproval(
                approval_id=f"{state['current_run_id']}:{result.agent}:approval",
                agent=result.agent,
                reason=result.approval.reason or result.summary,
                approval_type=result.approval.approval_type or "agent_approval",
                candidate_id=str(pending.get("candidate_id", "")),
                validation_id=str(pending.get("validation_id", "")),
                content_hashes=dict(pending.get("content_hashes") or {}),
            ).model_dump(mode="json")
            return {
                "pending_approval": approval,
                "terminal_state": SupervisorTerminalState.needs_user_approval.value,
                "next_action": "finalize",
                "final_answer": approval["reason"],
                "current_step": "resolve_candidate",
            }
        promoted = promote_pending_result(state)
        promoted["current_step"] = "resolve_candidate"
        latest_record = (state.get("validation_history") or [{}])[-1]
        semantic_result = next(
            (
                check.get("details", {})
                for check in latest_record.get("checks", [])
                if check.get("name") == "semantic"
            ),
            {},
        )
        recommended_action = str(semantic_result.get("recommended_next_action") or "")
        if result.agent in {"report_agent", "insight_agent"}:
            promoted["next_action"] = "finalize"
        elif _semantic_action_allowed(result.agent, recommended_action) and recommended_action:
            promoted["next_action"] = recommended_action
        else:
            promoted["next_action"] = "decide_next_action"
        if result.agent == "report_agent":
            summaries = list(promoted.get("step_summaries", []))
            summaries.append(
                StepSummary(
                    step="generate_report",
                    agent="report_agent",
                    action="call_report_agent",
                    summary=result.summary,
                    artifact_ids=list(result.artifact_ids),
                    next_action="finalize",
                ).model_dump(mode="json")
            )
            promoted["step_summaries"] = summaries
        _emit_run_event(
            backend_adapter,
            state,
            "evidence.promoted",
            f"{result.agent} 후보 근거를 승격했습니다.",
            artifact_ids=list(result.artifact_ids),
            metadata={
                "candidate_id": pending.get("candidate_id"),
                "validation_id": pending.get("validation_id"),
            },
        )
        return promoted

    return resolve_candidate_node


def _semantic_action_allowed(agent: str, action: str) -> bool:
    if not action:
        return True
    if action in {"decide_next_action", "finalize", "fail"}:
        return True
    allowed_by_agent = {
        "sql_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"},
        "eda_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"},
        "analysis_agent": {"call_sql_agent", "call_eda_agent", "call_analysis_agent", "call_report_agent"},
        "report_agent": {"call_report_agent"},
        "insight_agent": {"call_insight_agent"},
    }
    return action in allowed_by_agent.get(agent, set())


def _emit_run_event(
    backend_adapter: Any | None,
    state: SupervisorState,
    event_type: str,
    message: str,
    *,
    artifact_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    append_event = getattr(backend_adapter, "append_run_event", None)
    run_id = state.get("current_run_id")
    if append_event is None or not run_id:
        return
    append_event(
        run_id,
        event_type,
        message,
        node_name="supervisor",
        artifact_ids=artifact_ids,
        metadata=metadata,
    )


def make_finalize_node(model: Any | None):
    def finalize_node(state: SupervisorState) -> SupervisorState:
        try:
            decision = invoke_supervisor_decision(
                state,
                model,
                FINALIZE_DECISION_PROMPT,
                FinalizationDecision,
                extra=build_finalization_context(state),
            )
        except Exception as exc:
            return _decision_failure_updates(state, "finalize", exc)

        terminal_state = decision.terminal_state
        current_terminal_state = state.get("terminal_state")
        if current_terminal_state in FINALIZE_PROTECTED_TERMINAL_STATES:
            terminal_state = current_terminal_state

        final_answer = state.get("final_answer") or decision.final_answer
        if terminal_state == SupervisorTerminalState.completed.value:
            completion = _check_completion_readiness(state)
            if completion.status != "ready":
                terminal_state = SupervisorTerminalState.failed_terminal.value
                final_answer = completion.reason
        return {
            "terminal_state": terminal_state,
            "final_answer": final_answer,
            "next_action": "finalize",
            "current_step": "finalize",
            "llm_decisions": _append_llm_decision(state, "finalize", decision),
        }

    return finalize_node


def clarify_query_node(state: SupervisorState) -> SupervisorState:
    return make_clarify_query_node(None)(state)


def create_analysis_plan_node(state: SupervisorState) -> SupervisorState:
    return make_create_analysis_plan_node(None)(state)


def collect_clarification_node(state: SupervisorState) -> SupervisorState:
    return make_collect_clarification_node()(state)


def resolve_candidate_node(state: SupervisorState) -> SupervisorState:
    return make_resolve_candidate_node()(state)


def finalize_node(state: SupervisorState) -> SupervisorState:
    return make_finalize_node(None)(state)


def completion_guard_node(state: SupervisorState) -> SupervisorState:
    return make_completion_guard_node()(state)


def build_graph(
    subagent_adapter: Any,
    model: Any | None = None,
    checkpointer: Any | None = None,
    *,
    report_generator: Any | None = None,
    insight_generator: Any | None = None,
):
    if report_generator is None:
        if hasattr(subagent_adapter, "generate"):
            report_generator = subagent_adapter
        else:
            backend_adapter = getattr(subagent_adapter, "backend_adapter", subagent_adapter)
            from DATA_Analyst_Assistant_Agent.supervisor.reporting import SupervisorReportGenerator

            report_generator = SupervisorReportGenerator(backend_adapter)

    if insight_generator is None:
        if hasattr(subagent_adapter, "generate_insight"):
            insight_generator = _InsightGeneratorAdapter(subagent_adapter)
        elif hasattr(report_generator, "generate_insight"):
            insight_generator = _InsightGeneratorAdapter(report_generator)
        else:
            backend_adapter = getattr(subagent_adapter, "backend_adapter", subagent_adapter)
            from DATA_Analyst_Assistant_Agent.supervisor.insighting import SupervisorInsightGenerator

            insight_generator = SupervisorInsightGenerator(backend_adapter)

    graph = StateGraph(SupervisorState)
    backend_adapter = getattr(subagent_adapter, "backend_adapter", None)
    graph.add_node("clarify_query", make_clarify_query_node(model))
    graph.add_node("collect_clarification", make_collect_clarification_node())
    graph.add_node("create_analysis_plan", make_create_analysis_plan_node(model))
    graph.add_node("decide_next_action", make_decide_next_action_node(model))
    graph.add_node("completion_guard", make_completion_guard_node())
    graph.add_node("execute_subagent", make_execute_subagent_node(subagent_adapter, model))
    graph.add_node("generate_report", make_generate_report_node(report_generator))
    graph.add_node("generate_insight", make_generate_insight_node(insight_generator))
    graph.add_node("validate_candidate", make_validate_candidate_node(model))
    graph.add_node("commit_candidate", make_commit_candidate_node(backend_adapter))
    graph.add_node("collect_analysis_review", make_collect_analysis_review_node())
    graph.add_node("resolve_analysis_review", make_resolve_analysis_review_node(backend_adapter))
    graph.add_node("finalize", make_finalize_node(model))

    graph.add_edge(START, "clarify_query")
    graph.add_conditional_edges(
        "clarify_query",
        _route_after_clarify,
        {
            "collect_clarification": "collect_clarification",
            "create_analysis_plan": "create_analysis_plan",
            "finalize": "finalize",
        },
    )
    graph.add_edge("collect_clarification", "create_analysis_plan")
    graph.add_edge("create_analysis_plan", "decide_next_action")
    graph.add_conditional_edges(
        "decide_next_action",
        _route_after_decide,
        {
            "execute_subagent": "execute_subagent",
            "generate_report": "generate_report",
            "completion_guard": "completion_guard",
            "collect_analysis_review": "collect_analysis_review",
            "finalize": "finalize",
        },
    )
    graph.add_edge("collect_analysis_review", "resolve_analysis_review")
    graph.add_conditional_edges(
        "resolve_analysis_review",
        _route_after_commit_candidate,
        {
            "decide_next_action": "decide_next_action",
            "execute_subagent": "execute_subagent",
            "generate_report": "generate_report",
            "generate_insight": "generate_insight",
            "completion_guard": "completion_guard",
            "collect_analysis_review": "collect_analysis_review",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_subagent",
        _route_after_execute,
        {
            "generate_report": "generate_report",
            "generate_insight": "generate_insight",
            "validate_candidate": "validate_candidate",
            "completion_guard": "completion_guard",
            "finalize": "finalize",
        },
    )
    graph.add_edge("validate_candidate", "commit_candidate")
    graph.add_conditional_edges(
        "commit_candidate",
        _route_after_commit_candidate,
        {
            "decide_next_action": "decide_next_action",
            "execute_subagent": "execute_subagent",
            "generate_report": "generate_report",
            "generate_insight": "generate_insight",
            "completion_guard": "completion_guard",
            "collect_analysis_review": "collect_analysis_review",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "completion_guard",
        _route_after_completion_guard,
        {
            "generate_insight": "generate_insight",
            "generate_report": "generate_report",
            "finalize": "finalize",
        },
    )
    graph.add_edge("generate_report", "validate_candidate")
    graph.add_edge("generate_insight", "validate_candidate")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def _merge_state_updates(state: SupervisorState, state_updates: dict[str, Any]) -> SupervisorState:
    updates: SupervisorState = dict(state)
    if "generated_sql" in state_updates and state_updates["generated_sql"]:
        updates["generated_sql"] = str(state_updates["generated_sql"])
    if "error_state" in state_updates:
        updates["error_state"] = dict(state_updates["error_state"] or {})
    if "analysis_plan" in state_updates:
        plan = dict(updates.get("analysis_plan") or {})
        incoming_plan = dict(state_updates["analysis_plan"] or {})
        for sql_key in ("generated_sql", "source_sql"):
            if sql_key in incoming_plan and not incoming_plan[sql_key]:
                incoming_plan.pop(sql_key)
        plan.update(incoming_plan)
        updates["analysis_plan"] = plan
    if "planner_mode" in state_updates and state_updates["planner_mode"]:
        plan = dict(updates.get("analysis_plan") or {})
        plan["planner_mode"] = str(state_updates["planner_mode"])
        updates["analysis_plan"] = plan
    return updates


def _append_llm_decision(
    state: SupervisorState,
    node: str,
    decision: BaseModel,
) -> list[dict[str, Any]]:
    entries = list(state.get("llm_decisions", []))
    entries.append(
        {
            "node": node,
            "schema": decision.__class__.__name__,
            "decision": decision.model_dump(mode="json"),
        }
    )
    return entries


def _decision_failure_updates(state: SupervisorState, node: str, exc: Exception) -> SupervisorState:
    message = f"{node} 단계의 Supervisor LLM decision에 실패했습니다: {exc}"
    errors = list(state.get("decision_errors", []))
    errors.append(
        {
            "node": node,
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }
    )

    current_terminal_state = state.get("terminal_state")
    if current_terminal_state in FINALIZE_PROTECTED_TERMINAL_STATES:
        return {
            "terminal_state": current_terminal_state,
            "next_action": "finalize",
            "final_answer": state.get("final_answer")
            or _default_final_answer_for_terminal_state(state, current_terminal_state),
            "current_step": node,
            "decision_errors": errors,
            "error_state": {
                "node": node,
                "message": message,
            },
        }

    return {
        "terminal_state": SupervisorTerminalState.failed_terminal.value,
        "next_action": "finalize",
        "final_answer": message,
        "current_step": node,
        "decision_errors": errors,
        "error_state": {
            "node": node,
            "message": message,
        },
    }


def _terminal_failure_updates(
    state: SupervisorState,
    node: str,
    message: str,
    *,
    llm_decisions: list[dict[str, Any]] | None = None,
    extra_updates: SupervisorState | None = None,
) -> SupervisorState:
    updates: SupervisorState = {
        "terminal_state": SupervisorTerminalState.failed_terminal.value,
        "next_action": "finalize",
        "final_answer": message,
        "current_step": node,
        "error_state": {
            "node": node,
            "message": message,
        },
    }
    if llm_decisions is not None:
        updates["llm_decisions"] = llm_decisions
    if extra_updates:
        updates.update(extra_updates)
    return updates


def _route_after_clarify(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") == "clarify" or state.get("needs_clarification"):
        return "collect_clarification"
    return "create_analysis_plan"


def _route_after_decide(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"

    next_action = state.get("next_action")
    if next_action == "call_report_agent":
        return "generate_report"
    if next_action in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    if next_action in {"finalize", "fail"}:
        return "completion_guard"

    raise ValueError(
        f"decide_next_action 이후 지원하지 않는 next_action입니다: {next_action!r}"
    )


def _route_after_execute(state: SupervisorState) -> str:
    if state.get("pending_result") is not None:
        return "validate_candidate"
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") == "call_report_agent":
        return "generate_report"
    if state.get("next_action") == "call_insight_agent":
        return "generate_insight"
    return "completion_guard"


def _route_after_commit_candidate(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    next_action = state.get("next_action")
    if next_action == "collect_analysis_review":
        return "collect_analysis_review"
    if next_action == "decide_next_action":
        return "decide_next_action"
    if next_action == "call_report_agent":
        return "generate_report"
    if next_action == "call_insight_agent":
        return "generate_insight"
    if next_action in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    if next_action in {"finalize", "fail"}:
        return "completion_guard"
    raise ValueError(f"commit_candidate 이후 지원하지 않는 next_action입니다: {next_action!r}")


def _route_after_resolve_validation(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    latest = (state.get("validation_history") or [{}])[-1]
    disposition = (latest.get("outcome") or {}).get("disposition")
    if disposition == "retry":
        return "summarize_step"
    if disposition == "reject":
        return "finalize"
    return "resolve_candidate"


def _route_after_resolve_candidate(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") == "finalize":
        return "completion_guard"
    return "summarize_step"


def _route_after_summarize(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"

    next_action = state.get("next_action")
    if next_action == "decide_next_action":
        return "decide_next_action"
    if next_action == "call_report_agent":
        return "generate_report"
    if next_action in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    if next_action in {"finalize", "fail"}:
        return "completion_guard"

    raise ValueError(
        f"summarize_step 이후 지원하지 않는 next_action입니다: {next_action!r}"
    )


def _route_after_completion_guard(state: SupervisorState) -> str:
    if state.get("next_action") == "call_insight_agent":
        return "generate_insight"
    if state.get("next_action") == "call_report_agent":
        return "generate_report"
    return "finalize"


def _default_final_answer_for_terminal_state(
    state: SupervisorState,
    terminal_state: str,
) -> str:
    if terminal_state == SupervisorTerminalState.needs_user_approval.value:
        pending_approval = state.get("pending_approval")
        if isinstance(pending_approval, dict):
            reason = str(pending_approval.get("reason") or "")
            if reason:
                return reason
        return "사용자 승인이 필요합니다."

    if terminal_state == SupervisorTerminalState.needs_clarification.value:
        return state.get("clarification_question") or "추가 확인이 필요합니다."

    error_state = state.get("error_state")
    if isinstance(error_state, dict):
        message = str(error_state.get("message") or "")
        if message:
            return message

    if terminal_state == SupervisorTerminalState.failed_with_recoverable_context.value:
        return "복구 가능한 컨텍스트가 있지만 현재 요청을 완료하지 못했습니다."

    if terminal_state == SupervisorTerminalState.failed_terminal.value:
        return "요청을 완료할 수 없습니다."

    return "요청 처리를 종료했습니다."


def _clarification_answer_from_resume(resume_value: Any) -> str:
    if not isinstance(resume_value, dict):
        raise ValueError("clarification resume payload는 {'answer': '...'} 형식이어야 합니다.")
    answer = str(resume_value.get("answer") or "").strip()
    if not answer:
        raise ValueError("clarification resume payload의 answer는 비어 있을 수 없습니다.")
    return answer
