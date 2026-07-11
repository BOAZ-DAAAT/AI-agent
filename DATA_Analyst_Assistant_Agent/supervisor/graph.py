from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.decision import (
    AnalysisPlanDecision,
    ClarificationDecision,
    ExecutionGuardDecision,
    FinalizationDecision,
    SemanticValidationAdvisoryDecision,
    StepSummaryDecision,
    SupervisorDecision,
    build_clarification_context,
    build_execution_guard_context,
    build_finalization_context,
    build_next_action_context,
    build_plan_context,
    build_result_validation_context,
    build_step_summary_context,
    invoke_supervisor_decision,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import (
    CLARIFY_DECISION_PROMPT,
    DECIDE_NEXT_ACTION_PROMPT,
    EXECUTION_GUARD_DECISION_PROMPT,
    FINALIZE_DECISION_PROMPT,
    PLAN_DECISION_PROMPT,
    SEMANTIC_VALIDATION_ADVISORY_PROMPT,
    STEP_SUMMARY_DECISION_PROMPT,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    StepSummary,
    SupervisorState,
    artifact_ids_by_agent,
    promote_pending_result,
    reject_pending_result,
    stage_candidate_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentContractError, AgentToolResult
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    _check_completion_readiness,
    validate_subagent_result as validate_subagent_result_contract,
)
from DATA_Analyst_Assistant_Agent.supervisor.evidence import (
    EvidenceVerification,
    verify_candidate_evidence,
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
}

SUBAGENT_ACTION_TO_AGENT: dict[NextAction, AgentName] = {
    action: agent
    for action, agent in ACTION_TO_AGENT.items()
    if agent != "report_agent"
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
        try:
            guard = invoke_supervisor_decision(
                state,
                model,
                EXECUTION_GUARD_DECISION_PROMPT,
                ExecutionGuardDecision,
                extra=build_execution_guard_context(state),
            )
        except Exception as exc:
            return _decision_failure_updates(state, "execute_subagent", exc)

        llm_decisions = _append_llm_decision(state, "execute_subagent", guard)
        if not guard.allowed:
            if guard.next_action not in ACTION_TO_AGENT and guard.next_action not in {"finalize", "fail"}:
                return _terminal_failure_updates(
                    state,
                    "execute_subagent",
                    f"실행 guard가 지원하지 않는 대체 action을 반환했습니다: {guard.next_action}",
                    llm_decisions=llm_decisions,
                )
            updates: SupervisorState = {
                "next_action": guard.next_action,
                "current_step": "guard_blocked",
                "llm_decisions": llm_decisions,
            }
            if guard.next_action == "fail":
                updates["terminal_state"] = SupervisorTerminalState.failed_terminal.value
                updates["next_action"] = "finalize"
                updates["final_answer"] = guard.reason
            return updates

        if guard.next_action == "call_report_agent":
            return {
                "next_action": "call_report_agent",
                "current_step": "guard_redirected",
                "llm_decisions": llm_decisions,
            }

        agent_name = SUBAGENT_ACTION_TO_AGENT.get(guard.next_action)
        if agent_name is None:
            return _terminal_failure_updates(
                state,
                "execute_subagent",
                f"실행 guard가 실행 가능한 agent action을 반환하지 않았습니다: {guard.next_action}",
                llm_decisions=llm_decisions,
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
                llm_decisions=llm_decisions,
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
        updates["current_step"] = "executed_subagent"
        updates["next_action"] = guard.next_action
        updates["llm_decisions"] = llm_decisions
        _emit_run_event(
            getattr(subagent_adapter, "backend_adapter", None),
            state,
            "result.staged",
            f"{agent_name} 후보 결과를 격리했습니다.",
            metadata={"candidate_id": (updates.get("pending_result") or {}).get("candidate_id")},
        )
        return updates

    return execute_subagent_node


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


def make_stage_candidate_node():
    def stage_candidate_node(state: SupervisorState) -> SupervisorState:
        pending = state.get("pending_result")
        if not isinstance(pending, dict) or not isinstance(pending.get("result"), dict):
            return _terminal_failure_updates(state, "stage_candidate", "격리할 후보 결과가 없습니다.")
        return {"current_step": "stage_candidate"}

    return stage_candidate_node


def make_validate_subagent_result_node(model: Any | None, backend_adapter: Any | None = None):
    def validate_subagent_result_node(state: SupervisorState) -> SupervisorState:
        payload = state.get("last_agent_result") or {}
        if not payload:
            return _terminal_failure_updates(
                state,
                "validate_subagent_result",
                "검증할 에이전트 결과가 없습니다.",
            )

        try:
            result = AgentCompactResult.model_validate(payload)
        except Exception:
            return _terminal_failure_updates(
                state,
                "validate_subagent_result",
                "에이전트 실행 결과 형식이 올바르지 않습니다.",
                extra_updates={"last_agent_result": {}},
            )

        decision = validate_subagent_result_contract(state, result)
        raw_next_action = decision.next_action
        next_action = raw_next_action
        terminal_state = state.get("terminal_state") or "running"
        final_answer = ""

        hard_valid = decision.valid or decision.decision == "await_approval"
        if raw_next_action == "fail":
            terminal_state = decision.terminal_state
            next_action = "finalize"
            final_answer = decision.final_answer or decision.reason
        elif terminal_state in TERMINAL_STATES:
            next_action = "finalize"
        else:
            terminal_state = "running"

        validation_results = list(state.get("validation_results", []))
        validation_results.append(
            {
                "agent": result.agent,
                "valid": decision.valid,
                "hard_valid": hard_valid,
                "decision": decision.decision,
                "next_action": next_action,
                "terminal_state": terminal_state,
                "reason": decision.reason,
                "reason_code": decision.reason_code,
                "failure_reason": decision.failure_reason,
                "repeated_failure": decision.repeated_failure,
            }
        )

        updates: SupervisorState = {
            "validation_results": validation_results,
            "next_action": next_action,
            "terminal_state": terminal_state,
            "current_step": "validate_subagent_result",
        }
        if decision.failure_streak is not None:
            failure_streaks = {
                agent: dict(streak)
                for agent, streak in state.get("failure_streaks", {}).items()
            }
            failure_streaks[result.agent] = dict(decision.failure_streak)
            updates["failure_streaks"] = failure_streaks
        if final_answer:
            updates["final_answer"] = final_answer

        if not hard_valid:
            failure_metadata = {
                "agent": result.agent,
                "reason_code": decision.reason_code,
                "failure_reason": decision.failure_reason,
                "repeated_failure": decision.repeated_failure,
            }
            rejected = reject_pending_result(
                {**state, **updates},
                decision.reason,
                metadata=failure_metadata,
            )
            updates.update(rejected)
            failed = list(updates.get("failed_agents", []))
            if raw_next_action == "fail" and result.agent not in failed:
                failed.append(result.agent)
            updates["failed_agents"] = failed
            _emit_run_event(
                backend_adapter,
                state,
                "validation.rejected",
                decision.reason,
                metadata={
                    "candidate_id": (state.get("pending_result") or {}).get("candidate_id"),
                    **failure_metadata,
                },
            )

        if not hard_valid and raw_next_action in SUBAGENT_ACTION_TO_AGENT:
            retry_counts = dict(state.get("retry_counts", {}))
            retry_counts[result.agent] = int(retry_counts.get(result.agent, 0)) + 1
            updates["retry_counts"] = retry_counts
        return updates

    return validate_subagent_result_node


def make_evidence_validate_node(backend_adapter: Any | None):
    def evidence_validate_node(state: SupervisorState) -> SupervisorState:
        latest = (state.get("validation_results") or [{}])[-1]
        if latest.get("hard_valid") is not True:
            return {"current_step": "evidence_validate"}
        if backend_adapter is None or not hasattr(backend_adapter, "get_artifact"):
            result = AgentCompactResult.model_validate((state.get("pending_result") or {}).get("result") or {})
            valid = bool(result.artifact_ids or result.artifacts)
            verification = EvidenceVerification(
                valid=valid,
                decision="accept" if valid else "reject",
                findings=[] if valid else [],
            )
        else:
            verification = verify_candidate_evidence(state, backend_adapter)

        entries = list(state.get("evidence_validation_results", []))
        payload = verification.model_dump(mode="json")
        payload["candidate_id"] = (state.get("pending_result") or {}).get("candidate_id")
        entries.append(payload)
        updates: SupervisorState = {
            "evidence_validation_results": entries,
            "current_step": "evidence_validate",
        }
        if isinstance(state.get("pending_result"), dict):
            pending = dict(state["pending_result"])
            pending["content_hashes"] = verification.content_hashes
            updates["pending_result"] = pending
        if not verification.valid:
            reason = "; ".join(item.message for item in verification.findings) or "필수 근거 아티팩트가 없습니다."
            rejected = reject_pending_result({**state, **updates}, reason)
            updates.update(rejected)
            updates.update(
                {
                    "terminal_state": SupervisorTerminalState.failed_terminal.value,
                    "next_action": "finalize",
                    "final_answer": reason,
                }
            )
            _emit_run_event(
                backend_adapter,
                state,
                "validation.rejected",
                reason,
                metadata={"candidate_id": (state.get("pending_result") or {}).get("candidate_id")},
            )
        return updates

    return evidence_validate_node


def make_semantic_validate_subagent_result_node(model: Any | None, backend_adapter: Any | None = None):
    def semantic_validate_subagent_result_node(state: SupervisorState) -> SupervisorState:
        latest_validation = (state.get("validation_results") or [{}])[-1]
        evidence_entries = state.get("evidence_validation_results") or []
        latest_evidence = evidence_entries[-1] if evidence_entries else {"valid": True}
        hard_valid = latest_validation.get("hard_valid", latest_validation.get("valid"))
        if state.get("terminal_state") in TERMINAL_STATES or hard_valid is not True or latest_evidence.get("valid") is not True:
            return {
                "semantic_validation_results": list(state.get("semantic_validation_results", [])),
                "llm_decisions": list(state.get("llm_decisions", [])),
                "current_step": "semantic_validate_subagent_result",
            }

        payload = state.get("last_agent_result") or {}
        agent = str(payload.get("agent") or latest_validation.get("agent") or "")
        try:
            decision = invoke_supervisor_decision(
                state,
                model,
                SEMANTIC_VALIDATION_ADVISORY_PROMPT,
                SemanticValidationAdvisoryDecision,
                extra=build_result_validation_context(state),
            )
        except Exception as exc:
            counts = dict(state.get("semantic_retry_counts", {}))
            candidate_id = str((state.get("pending_result") or {}).get("candidate_id") or agent)
            attempts = int(counts.get(candidate_id, 0)) + 1
            counts[candidate_id] = attempts
            if attempts <= 1:
                return {
                    "semantic_retry_counts": counts,
                    "current_step": "semantic_validate_retry",
                }
            reason = f"semantic validation 모델 호출에 반복 실패했습니다: {exc}"
            rejected = reject_pending_result(state, reason)
            return {
                **rejected,
                "semantic_retry_counts": counts,
                "terminal_state": SupervisorTerminalState.failed_with_recoverable_context.value,
                "next_action": "finalize",
                "final_answer": reason,
                "current_step": "semantic_validate_subagent_result",
            }

        advisory_result = decision.model_dump(mode="json")
        advisory_result["agent"] = agent
        advisory_result["source_validation_result"] = latest_validation
        advisory_results = list(state.get("semantic_validation_results", []))
        advisory_results.append(advisory_result)
        updates: SupervisorState = {
            "semantic_validation_results": advisory_results,
            "terminal_state": state.get("terminal_state", "running"),
            "current_step": "semantic_validate_subagent_result",
            "llm_decisions": _append_llm_decision(
                state,
                "semantic_validate_subagent_result",
                decision,
            ),
        }
        invalid = (
            not decision.semantic_valid
            or decision.severity == "error"
            or bool(decision.missing_evidence)
            or not _semantic_action_allowed(agent, decision.recommended_next_action)
        )
        if invalid:
            reason = decision.reason or "semantic validation을 통과하지 못했습니다."
            rejected = reject_pending_result({**state, **updates}, reason)
            updates.update(rejected)
            updates.update(
                {
                    "terminal_state": SupervisorTerminalState.failed_terminal.value,
                    "next_action": "finalize",
                    "final_answer": reason,
                }
            )
            _emit_run_event(
                backend_adapter,
                state,
                "validation.rejected",
                reason,
                metadata={"candidate_id": (state.get("pending_result") or {}).get("candidate_id")},
            )
        return updates

    return semantic_validate_subagent_result_node


def make_resolve_candidate_node(backend_adapter: Any | None = None):
    def resolve_candidate_node(state: SupervisorState) -> SupervisorState:
        if state.get("terminal_state") in TERMINAL_STATES:
            return {"current_step": "resolve_candidate"}
        pending = state.get("pending_result")
        if not isinstance(pending, dict):
            return _terminal_failure_updates(state, "resolve_candidate", "승격할 후보 결과가 없습니다.")
        result = AgentCompactResult.model_validate(pending.get("result") or {})
        if result.approval.required:
            approval = {
                "approval_id": f"{state['current_run_id']}:{result.agent}:approval",
                "agent": result.agent,
                "reason": result.approval.reason or result.summary,
                "approval_type": result.approval.approval_type or "agent_approval",
                "candidate_id": pending.get("candidate_id", ""),
                "validation_id": pending.get("validation_id", ""),
                "content_hashes": dict(pending.get("content_hashes") or {}),
            }
            return {
                "pending_approval": approval,
                "terminal_state": SupervisorTerminalState.needs_user_approval.value,
                "next_action": "finalize",
                "final_answer": approval["reason"],
                "current_step": "resolve_candidate",
            }
        promoted = promote_pending_result(state)
        promoted["current_step"] = "resolve_candidate"
        semantic_result = (state.get("semantic_validation_results") or [{}])[-1]
        recommended_action = str(semantic_result.get("recommended_next_action") or "")
        if result.agent == "report_agent":
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


def make_summarize_step_node(model: Any | None):
    def summarize_step_node(state: SupervisorState) -> SupervisorState:
        payload = state.get("last_agent_result") or {}
        if not payload:
            return _terminal_failure_updates(
                state,
                "summarize_step",
                "요약할 에이전트 결과가 없습니다.",
            )

        try:
            AgentCompactResult.model_validate(payload)
        except Exception:
            return _terminal_failure_updates(
                state,
                "summarize_step",
                "요약할 에이전트 결과 형식이 올바르지 않습니다.",
                extra_updates={"last_agent_result": {}},
            )

        try:
            decision = invoke_supervisor_decision(
                state,
                model,
                STEP_SUMMARY_DECISION_PROMPT,
                StepSummaryDecision,
                extra=build_step_summary_context(state),
            )
        except Exception as exc:
            return _decision_failure_updates(state, "summarize_step", exc)

        summary = StepSummary(
            step=decision.step,
            agent=decision.agent,
            action=decision.action,
            summary=decision.summary,
            artifact_ids=list(decision.artifact_ids),
            next_action=decision.next_action,
        )
        step_summaries = list(state.get("step_summaries", []))
        step_summaries.append(summary.model_dump(mode="json"))
        return {
            "step_summaries": step_summaries,
            "current_step": "summarize_step",
            "llm_decisions": _append_llm_decision(state, "summarize_step", decision),
        }

    return summarize_step_node


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


def validate_subagent_result_node(state: SupervisorState) -> SupervisorState:
    return make_validate_subagent_result_node(None)(state)


def semantic_validate_subagent_result_node(state: SupervisorState) -> SupervisorState:
    return make_semantic_validate_subagent_result_node(None)(state)


def evidence_validate_node(state: SupervisorState) -> SupervisorState:
    return make_evidence_validate_node(None)(state)


def resolve_candidate_node(state: SupervisorState) -> SupervisorState:
    return make_resolve_candidate_node()(state)


def summarize_step_node(state: SupervisorState) -> SupervisorState:
    return make_summarize_step_node(None)(state)


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
):
    if report_generator is None:
        if hasattr(subagent_adapter, "generate"):
            report_generator = subagent_adapter
        else:
            backend_adapter = getattr(subagent_adapter, "backend_adapter", subagent_adapter)
            from DATA_Analyst_Assistant_Agent.supervisor.reporting import SupervisorReportGenerator

            report_generator = SupervisorReportGenerator(backend_adapter)

    graph = StateGraph(SupervisorState)
    backend_adapter = getattr(subagent_adapter, "backend_adapter", None)
    graph.add_node("clarify_query", make_clarify_query_node(model))
    graph.add_node("collect_clarification", make_collect_clarification_node())
    graph.add_node("create_analysis_plan", make_create_analysis_plan_node(model))
    graph.add_node("decide_next_action", make_decide_next_action_node(model))
    graph.add_node("completion_guard", make_completion_guard_node())
    graph.add_node("execute_subagent", make_execute_subagent_node(subagent_adapter, model))
    graph.add_node("generate_report", make_generate_report_node(report_generator))
    graph.add_node("stage_candidate", make_stage_candidate_node())
    graph.add_node("hard_validate", make_validate_subagent_result_node(model, backend_adapter))
    graph.add_node("evidence_validate", make_evidence_validate_node(backend_adapter))
    graph.add_node(
        "semantic_validate",
        make_semantic_validate_subagent_result_node(model, backend_adapter),
    )
    graph.add_node("summarize_step", make_summarize_step_node(model))
    graph.add_node("resolve_candidate", make_resolve_candidate_node(backend_adapter))
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
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_subagent",
        _route_after_execute,
        {
            "execute_subagent": "execute_subagent",
            "generate_report": "generate_report",
            "stage_candidate": "stage_candidate",
            "completion_guard": "completion_guard",
            "finalize": "finalize",
        },
    )
    graph.add_edge("stage_candidate", "hard_validate")
    graph.add_conditional_edges(
        "hard_validate",
        _route_after_validate,
        {
            "evidence_validate": "evidence_validate",
            "summarize_step": "summarize_step",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "evidence_validate",
        _route_after_evidence_validate,
        {
            "semantic_validate": "semantic_validate",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "semantic_validate",
        _route_after_semantic_validate,
        {
            "semantic_validate": "semantic_validate",
            "resolve_candidate": "resolve_candidate",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "resolve_candidate",
        _route_after_resolve_candidate,
        {
            "summarize_step": "summarize_step",
            "completion_guard": "completion_guard",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "summarize_step",
        _route_after_summarize,
        {
            "decide_next_action": "decide_next_action",
            "execute_subagent": "execute_subagent",
            "generate_report": "generate_report",
            "completion_guard": "completion_guard",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "completion_guard",
        _route_after_completion_guard,
        {
            "generate_report": "generate_report",
            "finalize": "finalize",
        },
    )
    graph.add_edge("generate_report", "stage_candidate")
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
    if state.get("next_action") == "call_report_agent":
        return "generate_report"
    if state.get("next_action") in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    return "completion_guard"


def _route_after_execute(state: SupervisorState) -> str:
    if state.get("current_step") == "executed_subagent" and state.get("last_agent_result"):
        return "stage_candidate"
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") == "call_report_agent":
        return "generate_report"
    if state.get("current_step") == "guard_blocked" and state.get("next_action") in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    return "completion_guard"


def _route_after_validate(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    latest_validation = (state.get("validation_results") or [{}])[-1]
    if latest_validation.get("hard_valid") is True:
        return "evidence_validate"
    return "summarize_step"


def _route_after_evidence_validate(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    return "semantic_validate"


def _route_after_semantic_validate(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("current_step") == "semantic_validate_retry":
        return "semantic_validate"
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
