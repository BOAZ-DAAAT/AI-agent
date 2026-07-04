from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.decision import (
    AnalysisPlanDecision,
    ClarificationDecision,
    ExecutionGuardDecision,
    FinalizationDecision,
    StepSummaryDecision,
    SupervisorDecision,
    build_clarification_context,
    build_execution_guard_context,
    build_finalization_context,
    build_next_action_context,
    build_plan_context,
    build_step_summary_context,
    invoke_supervisor_decision,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import (
    CLARIFY_DECISION_PROMPT,
    DECIDE_NEXT_ACTION_PROMPT,
    EXECUTION_GUARD_DECISION_PROMPT,
    FINALIZE_DECISION_PROMPT,
    PLAN_DECISION_PROMPT,
    STEP_SUMMARY_DECISION_PROMPT,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    StepSummary,
    SupervisorState,
    merge_agent_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentToolResult
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    validate_subagent_result as validate_subagent_result_contract,
)


ACTION_TO_AGENT: dict[NextAction, AgentName] = {
    "call_sql_agent": "sql_agent",
    "call_eda_agent": "eda_agent",
    "call_analysis_agent": "analysis_agent",
    "call_report_agent": "report_agent",
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
            updates["terminal_state"] = SupervisorTerminalState.needs_clarification.value
            updates["next_action"] = "finalize"
        else:
            updates["terminal_state"] = "running"
            updates["next_action"] = "create_plan"
        return updates

    return clarify_query_node


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

        return {
            "next_action": decision.next_action,
            "current_step": "decide_next_action",
            "llm_decisions": _append_llm_decision(state, "decide_next_action", decision),
        }

    return decide_next_action_node


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

        agent_name = ACTION_TO_AGENT.get(guard.next_action)
        if agent_name is None:
            return _terminal_failure_updates(
                state,
                "execute_subagent",
                f"실행 guard가 실행 가능한 agent action을 반환하지 않았습니다: {guard.next_action}",
                llm_decisions=llm_decisions,
            )

        tool_result: AgentToolResult = subagent_adapter.call(agent_name, state)
        merged = merge_agent_result(state, tool_result.agent_result)
        updates = _merge_state_updates(merged, tool_result.state_updates)
        updates["last_agent_result"] = tool_result.agent_result.model_dump(mode="json")
        updates["current_step"] = "executed_subagent"
        updates["next_action"] = guard.next_action
        updates["llm_decisions"] = llm_decisions
        return updates

    return execute_subagent_node


def make_validate_subagent_result_node(model: Any | None):
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

        if raw_next_action == "fail":
            terminal_state = SupervisorTerminalState.failed_terminal.value
            next_action = "finalize"
            final_answer = decision.reason
        elif terminal_state in TERMINAL_STATES:
            next_action = "finalize"
        else:
            terminal_state = "running"

        validation_results = list(state.get("validation_results", []))
        validation_results.append(
            {
                "agent": result.agent,
                "valid": decision.valid,
                "next_action": next_action,
                "terminal_state": terminal_state,
                "reason": decision.reason,
            }
        )

        updates: SupervisorState = {
            "validation_results": validation_results,
            "next_action": next_action,
            "terminal_state": terminal_state,
            "current_step": "validate_subagent_result",
        }
        if final_answer:
            updates["final_answer"] = final_answer

        if decision.valid:
            updates["failed_agents"] = [
                agent for agent in state.get("failed_agents", []) if agent != result.agent
            ]
        else:
            updates["completed_agents"] = [
                agent for agent in state.get("completed_agents", []) if agent != result.agent
            ]

        if not decision.valid and raw_next_action in ACTION_TO_AGENT:
            retry_counts = dict(state.get("retry_counts", {}))
            retry_counts[result.agent] = int(retry_counts.get(result.agent, 0)) + 1
            updates["retry_counts"] = retry_counts
        return updates

    return validate_subagent_result_node


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


def validate_subagent_result_node(state: SupervisorState) -> SupervisorState:
    return make_validate_subagent_result_node(None)(state)


def summarize_step_node(state: SupervisorState) -> SupervisorState:
    return make_summarize_step_node(None)(state)


def finalize_node(state: SupervisorState) -> SupervisorState:
    return make_finalize_node(None)(state)


def build_graph(
    subagent_adapter: Any,
    model: Any | None = None,
    checkpointer: Any | None = None,
):
    graph = StateGraph(SupervisorState)
    graph.add_node("clarify_query", make_clarify_query_node(model))
    graph.add_node("create_analysis_plan", make_create_analysis_plan_node(model))
    graph.add_node("decide_next_action", make_decide_next_action_node(model))
    graph.add_node("execute_subagent", make_execute_subagent_node(subagent_adapter, model))
    graph.add_node("validate_subagent_result", make_validate_subagent_result_node(model))
    graph.add_node("summarize_step", make_summarize_step_node(model))
    graph.add_node("finalize", make_finalize_node(model))

    graph.add_edge(START, "clarify_query")
    graph.add_conditional_edges(
        "clarify_query",
        _route_after_clarify,
        {"create_analysis_plan": "create_analysis_plan", "finalize": "finalize"},
    )
    graph.add_edge("create_analysis_plan", "decide_next_action")
    graph.add_conditional_edges(
        "decide_next_action",
        _route_after_decide,
        {"execute_subagent": "execute_subagent", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "execute_subagent",
        _route_after_execute,
        {
            "execute_subagent": "execute_subagent",
            "validate_subagent_result": "validate_subagent_result",
            "finalize": "finalize",
        },
    )
    graph.add_edge("validate_subagent_result", "summarize_step")
    graph.add_conditional_edges(
        "summarize_step",
        _route_after_summarize,
        {
            "decide_next_action": "decide_next_action",
            "execute_subagent": "execute_subagent",
            "finalize": "finalize",
        },
    )
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
    return "create_analysis_plan"


def _route_after_decide(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") in ACTION_TO_AGENT:
        return "execute_subagent"
    return "finalize"


def _route_after_execute(state: SupervisorState) -> str:
    if state.get("current_step") == "executed_subagent" and state.get("last_agent_result"):
        return "validate_subagent_result"
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("current_step") == "guard_blocked" and state.get("next_action") in ACTION_TO_AGENT:
        return "execute_subagent"
    return "finalize"


def _route_after_summarize(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") in ACTION_TO_AGENT:
        return "execute_subagent"
    if state.get("next_action") in {"finalize", "fail"}:
        return "finalize"
    return "decide_next_action"
