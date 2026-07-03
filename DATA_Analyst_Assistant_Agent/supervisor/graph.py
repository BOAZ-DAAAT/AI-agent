from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.decision import decide_next_action
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    NextAction,
    SupervisorState,
    merge_agent_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.summarizer import summarize_agent_step
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentToolResult
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    guard_agent_preconditions,
    validate_subagent_result,
)


ACTION_TO_AGENT: dict[NextAction, AgentName] = {
    "call_sql_agent": "sql_agent",
    "call_eda_agent": "eda_agent",
    "call_analysis_agent": "analysis_agent",
    "call_report_agent": "report_agent",
}

TERMINAL_STATES = {
    SupervisorTerminalState.completed.value,
    SupervisorTerminalState.needs_user_approval.value,
    SupervisorTerminalState.needs_clarification.value,
    SupervisorTerminalState.failed_terminal.value,
}


def clarify_query_node(state: SupervisorState) -> SupervisorState:
    query = (state.get("clarified_query") or state.get("latest_user_query") or "").strip()
    if len(query) >= 4:
        return {
            "clarified_query": query,
            "needs_clarification": False,
            "current_step": "clarify_query",
        }

    question = "분석할 질문을 조금 더 구체적으로 알려주세요."
    return {
        "clarified_query": query,
        "needs_clarification": True,
        "clarification_question": question,
        "terminal_state": SupervisorTerminalState.needs_clarification.value,
        "next_action": "finalize",
        "current_step": "clarify_query",
    }


def create_analysis_plan_node(state: SupervisorState) -> SupervisorState:
    plan = dict(state.get("analysis_plan") or {})
    plan.setdefault("goal", state.get("clarified_query") or state.get("latest_user_query") or "")
    plan.setdefault("route_kind", "comprehensive")
    plan.setdefault("planner_mode", "deterministic")
    if state.get("datasource_id") is not None:
        plan.setdefault("datasource_id", state.get("datasource_id"))
    return {
        "analysis_plan": plan,
        "current_step": "create_analysis_plan",
    }


def make_decide_next_action_node(model: Any | None):
    def decide_next_action_node(state: SupervisorState) -> SupervisorState:
        if state.get("terminal_state") in TERMINAL_STATES:
            return {"next_action": "finalize", "current_step": "decide_next_action"}

        decision = decide_next_action(state, model=model)
        return {
            "next_action": decision.next_action,
            "current_step": "decide_next_action",
        }

    return decide_next_action_node


def make_execute_subagent_node(subagent_adapter: Any):
    def execute_subagent_node(state: SupervisorState) -> SupervisorState:
        action = state.get("next_action")
        agent_name = ACTION_TO_AGENT.get(action)  # type: ignore[arg-type]
        if agent_name is None:
            terminal_state = (
                SupervisorTerminalState.failed_terminal.value if action == "fail" else state.get("terminal_state")
            )
            return {
                "terminal_state": terminal_state,
                "next_action": "finalize",
                "current_step": "execute_subagent",
            }

        guard = guard_agent_preconditions(agent_name, state)
        if not guard.allowed:
            updates: SupervisorState = {
                "next_action": guard.next_action,
                "current_step": "guard_blocked",
            }
            if guard.next_action == "fail":
                updates["terminal_state"] = SupervisorTerminalState.failed_terminal.value
                updates["final_answer"] = guard.reason
            return updates

        tool_result: AgentToolResult = subagent_adapter.call(agent_name, state)
        merged = merge_agent_result(state, tool_result.agent_result)
        updates = _merge_state_updates(merged, tool_result.state_updates)
        updates["last_agent_result"] = tool_result.agent_result.model_dump(mode="json")
        updates["current_step"] = "executed_subagent"
        return updates

    return execute_subagent_node


def validate_subagent_result_node(state: SupervisorState) -> SupervisorState:
    payload = state.get("last_agent_result") or {}
    if not payload:
        return {
            "terminal_state": SupervisorTerminalState.failed_terminal.value,
            "next_action": "finalize",
            "final_answer": "검증할 에이전트 결과가 없습니다.",
            "current_step": "validate_subagent_result",
        }

    result = AgentCompactResult.model_validate(payload)
    decision = validate_subagent_result(state, result)
    validation_results = list(state.get("validation_results", []))
    validation_results.append(
        {
            "agent": result.agent,
            "valid": decision.valid,
            "next_action": decision.next_action,
            "reason": decision.reason,
        }
    )

    updates: SupervisorState = {
        "validation_results": validation_results,
        "next_action": decision.next_action,
        "current_step": "validate_subagent_result",
    }
    if decision.next_action == "fail":
        updates["terminal_state"] = SupervisorTerminalState.failed_terminal.value
        updates["next_action"] = "finalize"
        updates["final_answer"] = "에이전트 실행 결과 검증에 실패했습니다."
    elif not decision.valid and decision.next_action in ACTION_TO_AGENT:
        retry_counts = dict(state.get("retry_counts", {}))
        retry_counts[result.agent] = int(retry_counts.get(result.agent, 0)) + 1
        updates["retry_counts"] = retry_counts
    return updates


def summarize_step_node(state: SupervisorState) -> SupervisorState:
    payload = state.get("last_agent_result") or {}
    if not payload:
        return {"current_step": "summarize_step"}

    result = AgentCompactResult.model_validate(payload)
    summary = summarize_agent_step(
        state.get("current_step", "execute_subagent"),
        result,
        next_action=state.get("next_action", ""),
    )
    step_summaries = list(state.get("step_summaries", []))
    step_summaries.append(summary.model_dump(mode="json"))
    return {
        "step_summaries": step_summaries,
        "current_step": "summarize_step",
    }


def finalize_node(state: SupervisorState) -> SupervisorState:
    terminal_state = state.get("terminal_state")
    final_answer = state.get("final_answer", "")

    if terminal_state == SupervisorTerminalState.needs_clarification.value:
        return {
            "final_answer": final_answer or state.get("clarification_question", ""),
            "next_action": "finalize",
            "current_step": "finalize",
        }
    if terminal_state == SupervisorTerminalState.needs_user_approval.value:
        return {
            "final_answer": final_answer or "사용자 승인이 필요합니다.",
            "next_action": "finalize",
            "current_step": "finalize",
        }
    if terminal_state == SupervisorTerminalState.failed_terminal.value:
        return {
            "final_answer": final_answer or "작업을 완료할 수 없습니다.",
            "next_action": "finalize",
            "current_step": "finalize",
        }
    if state.get("next_action") == "fail":
        return {
            "terminal_state": SupervisorTerminalState.failed_terminal.value,
            "final_answer": final_answer or "작업을 완료할 수 없습니다.",
            "next_action": "finalize",
            "current_step": "finalize",
        }

    return {
        "terminal_state": SupervisorTerminalState.completed.value,
        "final_answer": final_answer or "최종 리포트 생성이 완료되었습니다.",
        "next_action": "finalize",
        "current_step": "finalize",
    }


def build_graph(
    *,
    subagent_adapter: Any,
    model: Any | None = None,
    checkpointer: Any | None = None,
):
    graph = StateGraph(SupervisorState)
    graph.add_node("clarify_query", clarify_query_node)
    graph.add_node("create_analysis_plan", create_analysis_plan_node)
    graph.add_node("decide_next_action", make_decide_next_action_node(model))
    graph.add_node("execute_subagent", make_execute_subagent_node(subagent_adapter))
    graph.add_node("validate_subagent_result", validate_subagent_result_node)
    graph.add_node("summarize_step", summarize_step_node)
    graph.add_node("finalize", finalize_node)

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
        plan.update(dict(state_updates["analysis_plan"] or {}))
        updates["analysis_plan"] = plan
    if "planner_mode" in state_updates and state_updates["planner_mode"]:
        plan = dict(updates.get("analysis_plan") or {})
        plan["planner_mode"] = str(state_updates["planner_mode"])
        updates["analysis_plan"] = plan
    return updates


def _route_after_clarify(state: SupervisorState) -> str:
    if state.get("terminal_state") == SupervisorTerminalState.needs_clarification.value:
        return "finalize"
    return "create_analysis_plan"


def _route_after_decide(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") in ACTION_TO_AGENT:
        return "execute_subagent"
    return "finalize"


def _route_after_execute(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("current_step") == "guard_blocked" and state.get("next_action") in ACTION_TO_AGENT:
        return "execute_subagent"
    if state.get("current_step") == "executed_subagent" and state.get("last_agent_result"):
        return "validate_subagent_result"
    return "finalize"


def _route_after_summarize(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"
    if state.get("next_action") in ACTION_TO_AGENT:
        return "execute_subagent"
    if state.get("next_action") == "finalize":
        return "finalize"
    return "decide_next_action"
