from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.supervisor.capabilities import agent_capabilities_context
from DATA_Analyst_Assistant_Agent.supervisor.prompts import DECIDE_NEXT_ACTION_PROMPT
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentName,
    NextAction,
    SupervisorState,
    artifact_ids_by_agent,
    normalize_supervisor_state,
)
from DATA_Analyst_Assistant_Agent.supervisor.validation import ResultValidationDecision


_SNAPSHOT_MAX_TEXT = 400
_SNAPSHOT_MAX_ITEMS = 8
_SNAPSHOT_MAX_DEPTH = 4

RouteKind = Literal["simple", "eda", "trend", "mart", "comprehensive"]
TerminalStateValue = Literal[
    "completed",
    "needs_user_approval",
    "needs_clarification",
    "failed_with_recoverable_context",
    "failed_terminal",
]
LLMSelectableNextAction = Literal[
    "clarify",
    "create_plan",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "call_report_agent",
    "finalize",
    "fail",
]
SemanticRecoveryRecommendation = Literal[
    "clarify",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "call_report_agent",
    "finalize",
    "fail",
    "",
]
DecisionModelT = TypeVar("DecisionModelT", bound=BaseModel)


class SupervisorDecision(BaseModel):
    next_action: LLMSelectableNextAction
    reason: str = ""


class ClarificationDecision(BaseModel):
    needs_clarification: bool
    clarified_query: str = ""
    clarification_question: str = ""
    reason: str = ""


class AnalysisPlanDecision(BaseModel):
    goal: str
    route_kind: RouteKind = "simple"
    steps: list[str] = Field(default_factory=list)
    metric: str | None = None
    dimension: str | None = None
    filters: list[str] = Field(default_factory=list)
    requires_mart_review: bool = False
    reason: str = ""


class ExecutionGuardDecision(BaseModel):
    allowed: bool
    next_action: LLMSelectableNextAction
    reason: str = ""


class SemanticValidationAdvisoryDecision(BaseModel):
    semantic_valid: bool
    severity: Literal["info", "warning", "error"] = "info"
    recommended_next_action: SemanticRecoveryRecommendation = ""
    reason: str = ""
    missing_evidence: list[str] = Field(default_factory=list)
    alignment_notes: list[str] = Field(default_factory=list)


class StepSummaryDecision(BaseModel):
    step: str
    agent: AgentName | None = None
    action: str
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    next_action: NextAction | Literal[""] = ""
    reason: str = ""


class FinalizationDecision(BaseModel):
    terminal_state: TerminalStateValue
    final_answer: str
    next_action: Literal["finalize"] = "finalize"
    reason: str = ""


def parse_decision_json(text: str) -> SupervisorDecision:
    return parse_decision_json_as(text, SupervisorDecision)


def parse_decision_json_as(text: str, schema: type[DecisionModelT]) -> DecisionModelT:
    return schema.model_validate_json(_extract_json_object(text))


def invoke_supervisor_decision(
    state: SupervisorState,
    model: Any | None,
    prompt: str,
    schema: type[DecisionModelT],
    extra: dict[str, Any] | None = None,
) -> DecisionModelT:
    if model is None:
        raise RuntimeError("Supervisor LLM decision model is required.")

    payload = extra if extra is not None else build_next_action_context(state)
    response = model.invoke(_decision_messages(prompt, payload))
    content = getattr(response, "content", response)
    return parse_decision_json_as(str(content), schema)


def decide_next_action(state: SupervisorState, model: Any | None = None) -> SupervisorDecision:
    return invoke_supervisor_decision(
        state,
        model,
        DECIDE_NEXT_ACTION_PROMPT,
        SupervisorDecision,
        extra=build_next_action_context(state),
    )


def build_clarification_context(state: SupervisorState) -> dict[str, Any]:
    return _bounded_context(
        {
            "latest_user_query": state.get("latest_user_query", ""),
            "clarified_query": state.get("clarified_query", ""),
            "user_turns": list(state.get("user_turns", []))[-3:],
            "datasource_id": state.get("datasource_id"),
            "catalog_summary": state.get("catalog_summary"),
        },
        max_text=500,
        max_items=8,
        depth=4,
    )


def build_plan_context(state: SupervisorState) -> dict[str, Any]:
    validation_results, _, _ = _validation_context_views(state)
    return _bounded_context(
        {
            "query": state.get("clarified_query") or state.get("latest_user_query", ""),
            "datasource_id": state.get("datasource_id"),
            "catalog_summary": state.get("catalog_summary"),
            "existing_plan": state.get("analysis_plan") or {},
            "retry_counts": state.get("retry_counts", {}),
            "completed_agents": list(state.get("completed_agents", [])),
            "failed_agents": list(state.get("failed_agents", [])),
            "artifacts": artifact_ids_by_agent(state),
            "validation_results": validation_results[-3:],
        },
        max_text=500,
        max_items=8,
        depth=4,
    )


def build_next_action_context(state: SupervisorState) -> dict[str, Any]:
    validation_results, _, semantic_results = _validation_context_views(state)
    return _bounded_context(
        {
            "query": state.get("clarified_query") or state.get("latest_user_query", ""),
            "available_next_actions": [
                "clarify",
                "create_plan",
                "call_sql_agent",
                "call_eda_agent",
                "call_analysis_agent",
                "call_report_agent",
                "finalize",
                "fail",
            ],
            "analysis_plan": state.get("analysis_plan") or {},
            "completed_agents": list(state.get("completed_agents", [])),
            "failed_agents": list(state.get("failed_agents", [])),
            "artifacts": artifact_ids_by_agent(state),
            "validation_results": validation_results[-3:],
            "semantic_validation_results": semantic_results[-3:],
            "step_summaries": list(state.get("step_summaries", []))[-5:],
            "agent_capabilities": agent_capabilities_context(),
            "pending_approval": state.get("pending_approval"),
            "terminal_state": state.get("terminal_state", ""),
        },
        max_text=70,
        max_items=8,
        depth=3,
    )


def build_execution_guard_context(state: SupervisorState) -> dict[str, Any]:
    return _bounded_context(
        {
            "requested_next_action": state.get("next_action"),
            "analysis_plan": state.get("analysis_plan") or {},
            "completed_agents": list(state.get("completed_agents", [])),
            "failed_agents": list(state.get("failed_agents", [])),
            "artifacts": artifact_ids_by_agent(state),
            "pending_approval": state.get("pending_approval"),
            "retry_counts": state.get("retry_counts", {}),
            "last_agent_result": state.get("last_agent_result") or {},
            "allowed_agent_actions": [
                "call_sql_agent",
                "call_eda_agent",
                "call_analysis_agent",
                "call_report_agent",
            ],
            "agent_capabilities": agent_capabilities_context(),
        },
        max_text=400,
        max_items=8,
        depth=4,
    )


def build_result_validation_context(state: SupervisorState) -> dict[str, Any]:
    validation_results, _, semantic_results = _validation_context_views(state)
    return _bounded_context(
        {
            "query": state.get("clarified_query") or state.get("latest_user_query", ""),
            "latest_user_query": state.get("latest_user_query", ""),
            "clarified_query": state.get("clarified_query", ""),
            "last_agent_result": state.get("last_agent_result") or {},
            "analysis_plan": state.get("analysis_plan") or {},
            "artifacts": artifact_ids_by_agent(state),
            "validation_results": validation_results[-3:],
            "recent_semantic_validation_results": semantic_results[-3:],
            "step_summaries": list(state.get("step_summaries", []))[-3:],
            "completed_agents": list(state.get("completed_agents", [])),
            "failed_agents": list(state.get("failed_agents", [])),
            "pending_approval": state.get("pending_approval"),
            "retry_counts": state.get("retry_counts", {}),
            "max_retry_per_agent": state.get("max_retry_per_agent", 1),
            "terminal_state": state.get("terminal_state", "running"),
            "agent_capabilities": agent_capabilities_context(),
        },
        max_text=500,
        max_items=8,
        depth=4,
    )


def build_step_summary_context(state: SupervisorState) -> dict[str, Any]:
    validation_results, _, semantic_results = _validation_context_views(state)
    return _bounded_context(
        {
            "current_step": state.get("current_step", ""),
            "last_agent_result": state.get("last_agent_result") or {},
            "latest_validation_result": (validation_results or [{}])[-1],
            "latest_semantic_validation_result": (semantic_results or [{}])[-1],
            "next_action": state.get("next_action", ""),
            "existing_step_summaries": list(state.get("step_summaries", []))[-3:],
        },
        max_text=500,
        max_items=8,
        depth=4,
    )


def build_finalization_context(state: SupervisorState) -> dict[str, Any]:
    validation_results, _, semantic_results = _validation_context_views(state)
    latest_validation = (validation_results or [{}])[-1]
    latest_agent = str(latest_validation.get("agent") or "")
    recent_failure_streak = (state.get("failure_streaks") or {}).get(latest_agent)
    return _bounded_context(
        {
            "query": state.get("clarified_query") or state.get("latest_user_query", ""),
            "terminal_state": state.get("terminal_state", "running"),
            "next_action": state.get("next_action", ""),
            "final_answer": state.get("final_answer", ""),
            "clarification_question": state.get("clarification_question", ""),
            "pending_approval": state.get("pending_approval"),
            "analysis_plan": state.get("analysis_plan") or {},
            "artifacts": artifact_ids_by_agent(state),
            "completed_agents": list(state.get("completed_agents", [])),
            "failed_agents": list(state.get("failed_agents", [])),
            "validation_results": validation_results[-3:],
            "semantic_validation_results": semantic_results[-3:],
            "step_summaries": list(state.get("step_summaries", []))[-5:],
            "llm_decisions": list(state.get("llm_decisions", []))[-5:],
            "decision_errors": list(state.get("decision_errors", []))[-3:],
            "recent_failure_streak": recent_failure_streak,
        },
        max_text=500,
        max_items=8,
        depth=4,
    )


def _validation_context_views(
    state: SupervisorState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    history = list(state.get("validation_history", []))
    if not history and any(
        state.get(key)
        for key in (
            "validation_results",
            "evidence_validation_results",
            "semantic_validation_results",
        )
    ):
        legacy_state = {**state, "state_schema_version": 2, "validation_history": []}
        history = list(normalize_supervisor_state(legacy_state).get("validation_history", []))

    result_views: list[dict[str, Any]] = []
    evidence_views: list[dict[str, Any]] = []
    semantic_views: list[dict[str, Any]] = []
    for record in history:
        outcome = dict(record.get("outcome") or {})
        checks = list(record.get("checks") or [])
        result_check = next((item for item in checks if item.get("name") == "result"), {})
        result_details = dict(result_check.get("details") or {})
        result_views.append(
            {
                "agent": record.get("agent"),
                "valid": bool(result_check.get("passed")),
                "hard_valid": bool(result_check.get("passed")),
                "decision": outcome.get("disposition", "reject"),
                "reason": outcome.get("reason", ""),
                "reason_code": outcome.get("reason_code", "none"),
                "terminal_state": outcome.get("terminal_state", "running"),
                "failure_reason": result_details.get("failure_reason", ""),
                "repeated_failure": bool(result_details.get("repeated_failure", False)),
                "candidate_id": record.get("candidate_id", ""),
                "validation_id": record.get("validation_id", ""),
            }
        )
        evidence_check = next((item for item in checks if item.get("name") == "evidence"), None)
        if evidence_check is not None:
            evidence_details = dict(evidence_check.get("details") or {})
            evidence_views.append(
                {
                    **evidence_details,
                    "valid": bool(evidence_check.get("passed")),
                    "findings": list(evidence_check.get("findings") or []),
                    "candidate_id": record.get("candidate_id", ""),
                }
            )
        semantic_check = next((item for item in checks if item.get("name") == "semantic"), None)
        if semantic_check is not None:
            semantic_details = dict(semantic_check.get("details") or {})
            semantic_views.append(
                {
                    **semantic_details,
                    "semantic_valid": bool(semantic_check.get("passed")),
                    "agent": record.get("agent"),
                    "source_validation_result": result_views[-1],
                }
            )
    return result_views, evidence_views, semantic_views


def _decision_messages(prompt: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _bounded_context(
    value: dict[str, Any],
    *,
    max_text: int = _SNAPSHOT_MAX_TEXT,
    max_items: int = _SNAPSHOT_MAX_ITEMS,
    depth: int = _SNAPSHOT_MAX_DEPTH,
) -> dict[str, Any]:
    return {
        str(key)[:max_text]: _bounded_value(
            item,
            max_text=max_text,
            max_items=max_items,
            depth=depth,
        )
        for key, item in value.items()
    }


def _bounded_value(
    value: Any,
    *,
    max_text: int = _SNAPSHOT_MAX_TEXT,
    max_items: int = _SNAPSHOT_MAX_ITEMS,
    depth: int = _SNAPSHOT_MAX_DEPTH,
) -> Any:
    if isinstance(value, str):
        return value[:max_text]
    if isinstance(value, BaseModel):
        return _bounded_value(
            value.model_dump(mode="json"),
            max_text=max_text,
            max_items=max_items,
            depth=depth,
        )
    if depth <= 0:
        if isinstance(value, (dict, list, tuple, set)):
            return "..."
        return value
    if isinstance(value, dict):
        return {
            str(key)[:max_text]: _bounded_value(
                item,
                max_text=max_text,
                max_items=max_items,
                depth=depth - 1,
            )
            for key, item in list(value.items())[:max_items]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _bounded_value(
                item,
                max_text=max_text,
                max_items=max_items,
                depth=depth - 1,
            )
            for item in list(value)[:max_items]
        ]
    return value


def _extract_json_object(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    raise ValueError("No JSON object found in decision text")
