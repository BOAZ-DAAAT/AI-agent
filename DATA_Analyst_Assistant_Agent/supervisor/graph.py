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
    AnalysisRuleExtractionDecision,
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
from DATA_Analyst_Assistant_Agent.supervisor.lifecycle import (
    emit_node_lifecycle_event,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import (
    ANALYSIS_RULE_EXTRACTION_PROMPT,
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
    StepSummary,
    SupervisorState,
    artifact_ids_by_agent,
    begin_or_retry_agent_node,
    fail_active_node,
    reject_pending_result,
    stage_candidate_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentContractError, AgentToolResult
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
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
    "call_insight": "insight",
}

_NODE_ONLY_AGENTS = {"insight"}

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

def _default_analysis_rule_search(query: str, **kwargs: Any) -> list[Any]:
    from DATA_Analyst_Assistant_Agent.shared.pinecone import search_company_context

    return search_company_context(query, **kwargs)


ANALYSIS_RULE_CONTEXT_LIMIT = 12
ANALYSIS_RULE_CANDIDATE_LIMIT = 24
ANALYSIS_RULE_RETRIEVAL_GROUPS = (
    ("analysis_query_rule", 5),
    ("analysis_foundation", 6),
    ("analysis_integrity_caution", 1),
)


def make_retrieve_analysis_rules_node(search: Any | None = None, model: Any | None = None):
    rule_search = search or _default_analysis_rule_search

    def retrieve_analysis_rules_node(state: SupervisorState) -> SupervisorState:
        query = str(state.get("clarified_query") or state.get("latest_user_query") or "").strip()
        base_updates: SupervisorState = {
            "analysis_rule_context": None,
            "current_step": "retrieve_analysis_rules",
            "terminal_state": "running",
        }
        if not query:
            return {
                **base_updates,
                "analysis_rule_retrieval": {"status": "skipped", "reason": "empty_query"},
            }

        try:
            hits = _retrieve_diverse_analysis_hits(rule_search, query)
        except Exception as exc:
            status = (
                "disabled"
                if exc.__class__.__name__ == "PineconeConfigurationError"
                else "failed"
            )
            limitation = f"분석 규칙 검색을 적용하지 못했습니다: {exc}"
            return {
                **base_updates,
                "analysis_rule_retrieval": {
                    "status": status,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
                "limitations": [*state.get("limitations", []), limitation],
                "run_events": [
                    *state.get("run_events", []),
                    {"type": "analysis_rule_retrieval", "status": status},
                ],
            }

        if not hits:
            return {
                **base_updates,
                "analysis_rule_retrieval": {"status": "empty"},
                "run_events": [
                    *state.get("run_events", []),
                    {"type": "analysis_rule_retrieval", "status": "empty"},
                ],
            }

        try:
            extraction = invoke_supervisor_decision(
                state,
                model,
                ANALYSIS_RULE_EXTRACTION_PROMPT,
                AnalysisRuleExtractionDecision,
                extra={
                    "user_query": query,
                    "document": _rule_document_payload(hits[0]),
                    "documents": [_rule_document_payload(hit) for hit in hits],
                },
            )
        except Exception as exc:
            limitation = f"검색된 분석 규칙 문서에서 필요한 규칙을 추출하지 못했습니다: {exc}"
            return {
                **base_updates,
                "analysis_rule_retrieval": {
                    "status": "extraction_failed",
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
                "limitations": [*state.get("limitations", []), limitation],
                "run_events": [
                    *state.get("run_events", []),
                    {"type": "analysis_rule_retrieval", "status": "extraction_failed"},
                ],
            }

        if not extraction.applicable or not extraction.rules:
            return {
                **base_updates,
                "analysis_rule_retrieval": {
                    "status": "not_applicable",
                    "document_id": str(_hit_value(hits[0], "document_id", "")),
                    "reason": extraction.reason,
                },
                "run_events": [
                    *state.get("run_events", []),
                    {"type": "analysis_rule_retrieval", "status": "not_applicable"},
                ],
            }

        context = _analysis_rule_context(hits, extraction)
        retrieved_documents = _retrieved_document_summaries(hits)
        return {
            **base_updates,
            "analysis_rule_context": context,
            "analysis_rule_retrieval": {
                "status": "success",
                "document_id": context.get("document_id", ""),
                "query_type": context.get("query_type", ""),
                "related_query_types": list(context.get("related_query_types", [])),
                "hit_count": len(hits),
                "unique_document_count": len(retrieved_documents),
                "score": float(_hit_value(hits[0], "score", 0.0) or 0.0),
                "retrieved_documents": retrieved_documents,
                "reason": extraction.reason,
            },
            "run_events": [
                *state.get("run_events", []),
                {
                    "type": "analysis_rule_retrieval",
                    "status": "success",
                    "document_id": context.get("document_id", ""),
                    "retrieved_documents": retrieved_documents,
                },
            ],
        }

    return retrieve_analysis_rules_node


def _retrieve_diverse_analysis_hits(search: Any, query: str) -> list[Any]:
    """Reserve final planning context for both intent rules and schema-grounded foundations."""
    selected: list[Any] = []
    seen_document_ids: set[str] = set()
    candidates: list[Any] = []

    for doc_type, quota in ANALYSIS_RULE_RETRIEVAL_GROUPS:
        group_hits = search(
            query,
            top_k=ANALYSIS_RULE_CANDIDATE_LIMIT,
            metadata_filter={"doc_type": doc_type},
        )
        candidates.extend(group_hits)
        for hit in _best_hit_per_document(group_hits):
            document_id = str(_hit_value(hit, "document_id", "")).strip()
            if document_id in seen_document_ids:
                continue
            selected.append(hit)
            seen_document_ids.add(document_id)
            if sum(1 for item in selected if _hit_metadata(item).get("doc_type") == doc_type) >= quota:
                break

    for hit in _best_hit_per_document(candidates):
        if len(selected) >= ANALYSIS_RULE_CONTEXT_LIMIT:
            break
        document_id = str(_hit_value(hit, "document_id", "")).strip()
        if document_id in seen_document_ids:
            continue
        selected.append(hit)
        seen_document_ids.add(document_id)

    return selected[:ANALYSIS_RULE_CONTEXT_LIMIT]


def _best_hit_per_document(hits: list[Any]) -> list[Any]:
    best_hits: dict[str, Any] = {}
    for hit in hits:
        document_id = str(_hit_value(hit, "document_id", "")).strip()
        key = document_id or str(_hit_value(hit, "record_id", ""))
        current = best_hits.get(key)
        if current is None or _planning_hit_rank(hit) > _planning_hit_rank(current):
            best_hits[key] = hit
    return sorted(best_hits.values(), key=_planning_hit_rank, reverse=True)


def _planning_hit_rank(hit: Any) -> tuple[float, int]:
    record_type = str(_hit_metadata(hit).get("record_type") or "")
    record_type_priority = {"rule_atom": 2, "section_chunk": 1, "example": 0}.get(record_type, 1)
    score = float(_hit_value(hit, "score", 0.0) or 0.0)
    # Example records can be linguistically close but are weaker planning context.
    example_penalty = 0.05 if record_type == "example" else 0.0
    return (score - example_penalty, record_type_priority)


def _analysis_rule_context(
    hits: list[Any],
    extraction: AnalysisRuleExtractionDecision,
) -> dict[str, Any]:
    first_hit = hits[0] if hits else {}
    metadata = _hit_value(first_hit, "metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    related_query_types = _ordered_unique(
        str(_hit_metadata(hit).get("query_type") or "")
        for hit in hits
        if str(_hit_metadata(hit).get("query_type") or "").strip()
    )[:3]
    source_sections = _ordered_unique(
        str(_hit_metadata(hit).get("section") or "")
        for hit in hits
        if str(_hit_metadata(hit).get("section") or "").strip()
    )[:12]
    integrity_cautions = _ordered_unique(
        str(item)
        for hit in hits
        for item in _as_list(_hit_metadata(hit).get("integrity_cautions"))
        if str(item).strip()
    )[:12]
    schema_warnings = _ordered_unique(
        str(item)
        for hit in hits
        for item in _as_list(_hit_metadata(hit).get("schema_warnings"))
        if str(item).strip()
    )[:12]
    context: dict[str, Any] = {
        "document_id": str(
            metadata.get("document_id") or _hit_value(first_hit, "document_id", "")
        ),
        "title": str(metadata.get("title") or _hit_value(first_hit, "title", "")),
        "query_type": str(metadata.get("query_type") or ""),
        "version": str(metadata.get("version") or ""),
    }
    if related_query_types:
        context["related_query_types"] = related_query_types
    if source_sections:
        context["source_sections"] = source_sections
    if integrity_cautions:
        context["integrity_cautions"] = integrity_cautions
    if schema_warnings:
        context["schema_warnings"] = schema_warnings
    context["rules"] = [
        " ".join(str(rule).split())[:500]
        for rule in extraction.rules[:12]
        if str(rule).strip()
    ]
    if extraction.clarification_needed and extraction.clarification_question.strip():
        context["clarification_question"] = " ".join(
            extraction.clarification_question.split()
        )[:500]
    return context


def _rule_document_payload(hit: Any) -> dict[str, Any]:
    return {
        "document_id": _hit_value(hit, "document_id", ""),
        "title": _hit_value(hit, "title", ""),
        "record_id": _hit_value(hit, "record_id", ""),
        "score": _hit_value(hit, "score", 0.0),
        "metadata": _hit_value(hit, "metadata", {}),
        "content": _hit_value(hit, "text", ""),
    }


def _retrieved_document_summaries(hits: list[Any]) -> list[dict[str, Any]]:
    """Keep retrieval traceable without retaining full retrieved document text in state."""
    summaries: dict[str, dict[str, Any]] = {}
    for hit in hits:
        metadata = _hit_metadata(hit)
        document_id = str(_hit_value(hit, "document_id", ""))
        key = document_id or str(_hit_value(hit, "record_id", ""))
        summary = summaries.setdefault(
            key,
            {
                "document_id": document_id,
                "doc_type": str(metadata.get("doc_type") or ""),
                "score": round(float(_hit_value(hit, "score", 0.0) or 0.0), 6),
                "sections": [],
                "record_count": 0,
            },
        )
        section = str(metadata.get("section") or "")
        if section and section not in summary["sections"]:
            summary["sections"].append(section)
        summary["record_count"] += 1
    return list(summaries.values())


def _hit_value(hit: Any, key: str, default: Any) -> Any:
    if isinstance(hit, dict):
        return hit.get(key, default)
    return getattr(hit, key, default)


def _hit_metadata(hit: Any) -> dict[str, Any]:
    metadata = _hit_value(hit, "metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


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
            "clarification_input_mode": decision.input_mode,
            "clarification_options": decision.options,
            "clarification_allow_free_text": decision.allow_free_text,
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
            "input_mode": state.get("clarification_input_mode") or "free_text",
            "options": state.get("clarification_options") or [],
            "allow_free_text": bool(state.get("clarification_allow_free_text", True)),
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
            active_node_payload = rejected.get("active_node")
            if (
                isinstance(active_node_payload, dict)
                and active_node_payload.get("status") == "waiting"
            ):
                rejected, resumed_node, resumed_event_type = begin_or_retry_agent_node(
                    rejected,
                    "analysis_agent",
                )
                emit_node_lifecycle_event(
                    backend_adapter,
                    rejected,
                    resumed_event_type,
                    resumed_node,
                    "analysis_agent 후속 분석을 재개했습니다.",
                    action="call_analysis_agent",
                    approval_id=str(pending_approval.get("approval_id") or "") or None,
                    metadata={
                        "candidate_id": pending_result.get("candidate_id"),
                        "validation_id": pending_result.get("validation_id"),
                        "reason_code": "analysis.review_followup",
                    },
                )
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
            "query_rules": dict(state.get("analysis_rule_context") or {}),
            "required_derivations": list(decision.required_derivations),
            "analysis_heuristics": list(decision.analysis_heuristics),
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
                "next_action": "call_insight",
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
        if action == "call_insight":
            return {
                "next_action": "call_insight",
                "current_step": "execute_subagent",
            }

        agent_name = SUBAGENT_ACTION_TO_AGENT.get(action)
        if agent_name is None:
            return _terminal_failure_updates(
                state,
                "execute_subagent",
                f"지원하지 않는 subagent action입니다: {action}",
            )

        working_state, active_node, activation_event_type = (
            begin_or_retry_agent_node(
                state,
                agent_name,
            )
        )
        activation_message = {
            "agent.started": f"{agent_name} 작업을 시작했습니다.",
            "agent.retrying": (
                f"{agent_name} 작업을 재시도합니다. "
                f"현재 시도: {active_node.attempt}"
            ),
            "agent.resumed": f"{agent_name} 작업을 재개했습니다.",
        }[activation_event_type]
        emit_node_lifecycle_event(
            getattr(subagent_adapter, "backend_adapter", None),
            working_state,
            activation_event_type,
            active_node,
            activation_message,
            action=action,
        )
        try:
            tool_result: AgentToolResult = subagent_adapter.call(
                agent_name,
                working_state,
            )
        except AgentContractError as exc:
            message = f"{agent_name} 결과의 에이전트 계약 검증에 실패했습니다: {exc}"
            failed_state, failed_node, failed_event_type = fail_active_node(
                working_state,
                agent_name=agent_name,
                reason=message,
                reason_code="agent_contract_mismatch",
            )
            emit_node_lifecycle_event(
                getattr(subagent_adapter, "backend_adapter", None),
                failed_state,
                failed_event_type,
                failed_node,
                message,
                action=action,
                metadata={
                    "reason_code": "agent_contract_mismatch",
                },
            )
            failed_agents = list(failed_state.get("failed_agents", []))
            if agent_name not in failed_agents:
                failed_agents.append(agent_name)
            return {
                **failed_state,
                **_terminal_failure_updates(
                    failed_state,
                    "execute_subagent",
                    message,
                    extra_updates={
                        "pending_result": None,
                        "last_agent_result": {},
                        "failed_agents": failed_agents,
                        "completed_agents": list(failed_state.get("completed_agents", [])),
                        "accepted_evidence": {
                            agent: list(items)
                            for agent, items in failed_state.get(
                                "accepted_evidence",
                                {},
                            ).items()
                        },
                        "error_state": {
                            "node": "execute_subagent",
                            "message": message,
                            "reason_code": "agent_contract_mismatch",
                            "retryable": False,
                        },
                    },
                ),
            }

        except Exception as exc:  # noqa: BLE001
            reason_code = "agent_execution_error"
            message = f"{agent_name} execution failed in execute_subagent: {type(exc).__name__}: {exc}"
            failed_state, failed_node, failed_event_type = fail_active_node(
                working_state,
                agent_name=agent_name,
                reason=message,
                reason_code=reason_code,
            )
            emit_node_lifecycle_event(
                getattr(subagent_adapter, "backend_adapter", None),
                failed_state,
                failed_event_type,
                failed_node,
                message,
                action=action,
                metadata={
                    "reason_code": reason_code,
                    "exception_type": type(exc).__name__,
                },
            )
            failed_agents = list(failed_state.get("failed_agents", []))
            if agent_name not in failed_agents:
                failed_agents.append(agent_name)
            return {
                **failed_state,
                **_terminal_failure_updates(
                    failed_state,
                    "execute_subagent",
                    message,
                    extra_updates={
                        "pending_result": None,
                        "last_agent_result": {},
                        "failed_agents": failed_agents,
                        "completed_agents": list(failed_state.get("completed_agents", [])),
                        "accepted_evidence": {
                            agent: list(items)
                            for agent, items in failed_state.get(
                                "accepted_evidence",
                                {},
                            ).items()
                        },
                        "error_state": {
                            "node": "execute_subagent",
                            "message": message,
                            "reason_code": reason_code,
                            "exception_type": type(exc).__name__,
                            "retryable": False,
                        },
                    },
                ),
            }

        updates = stage_candidate_result(
            working_state,
            tool_result.agent_result,
            tool_result.state_updates,
        )
        updates["last_agent_result"] = tool_result.agent_result.model_dump(mode="json")
        updates["current_step"] = "execute_subagent"
        updates["next_action"] = action
        _emit_run_event(
            getattr(subagent_adapter, "backend_adapter", None),
            state,
            "result.staged",
            f"{agent_name} 후보 결과를 격리했습니다.",
            node_name=agent_name,
            metadata={
                "agent_name": agent_name,
                "node_id": active_node.node_id,
                "candidate_id": (updates.get("pending_result") or {}).get("candidate_id"),
            },
        )
        return updates

    return execute_subagent_node


def make_commit_candidate_node(backend_adapter: Any | None = None):
    def commit_candidate_node(state: SupervisorState) -> SupervisorState:
        return commit_candidate(state, backend_adapter)

    return commit_candidate_node


class _InsightGeneratorAdapter:
    """subagent_adapter가 노출하는 generate_insight()를 generate_insight_node가 기대하는
    .generate() 인터페이스로 맞춰준다(호출 시그니처 통일용 얇은 어댑터)."""

    def __init__(self, subagent_adapter: Any) -> None:
        self._subagent_adapter = subagent_adapter
        self.backend_adapter = getattr(subagent_adapter, "backend_adapter", None)

    def generate(self, state: SupervisorState) -> AgentCompactResult:
        return self._subagent_adapter.generate_insight(state)


def make_generate_insight_node(insight_generator: Any):
    def generate_insight_node(state: SupervisorState) -> SupervisorState:
        working_state, active_node, activation_event_type = (
            begin_or_retry_agent_node(
                state,
                "insight",
            )
        )
        activation_message = {
            "agent.started": "insight 작업을 시작했습니다.",
            "agent.retrying": (
                "insight 작업을 재시도합니다. "
                f"현재 시도: {active_node.attempt}"
            ),
            "agent.resumed": "insight 작업을 재개했습니다.",
        }[activation_event_type]
        emit_node_lifecycle_event(
            getattr(insight_generator, "backend_adapter", None),
            working_state,
            activation_event_type,
            active_node,
            activation_message,
            action="call_insight",
        )
        evidence_ids = artifact_ids_by_agent(working_state)
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
                result = insight_generator.generate(working_state)
            except Exception as exc:
                result = _failed_insight_result(f"인사이트 생성 또는 저장에 실패했습니다: {exc}")

        if result.agent != "insight":
            result = _failed_insight_result(
                f"인사이트 생성기가 잘못된 agent 결과를 반환했습니다: {result.agent}"
            )

        updates: SupervisorState = {
            **stage_candidate_result(working_state, result, {}),
            "last_agent_result": result.model_dump(mode="json"),
            "terminal_state": "running",
            "next_action": "call_insight",
            "current_step": "generate_insight",
        }
        _emit_run_event(
            getattr(insight_generator, "backend_adapter", None),
            working_state,
            "result.staged",
            "insight 후보 결과를 격리했습니다.",
            node_name="insight",
            metadata={
                "agent_name": "insight",
                "node_id": active_node.node_id,
                "candidate_id": (updates.get("pending_result") or {}).get("candidate_id"),
            },
        )
        return updates

    return generate_insight_node


def _failed_insight_result(message: str) -> AgentCompactResult:
    return AgentCompactResult(
        agent="insight",
        status="failed",
        summary=message,
        retryable=False,
        error=message,
    )


def make_validate_candidate_node(model: Any | None):
    def validate_candidate_node(state: SupervisorState) -> SupervisorState:
        return validate_candidate(state, model)

    return validate_candidate_node


def _emit_run_event(
    backend_adapter: Any | None,
    state: SupervisorState,
    event_type: str,
    message: str,
    *,
    node_name: str = "supervisor",
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
        node_name=node_name,
        artifact_ids=artifact_ids,
        metadata=metadata,
    )


def _close_active_node_for_terminal_state(
    state: SupervisorState,
    updates: SupervisorState,
    backend_adapter: Any | None,
) -> SupervisorState:
    terminal_state = updates.get("terminal_state")
    active_node_payload = state.get("active_node")
    if not isinstance(active_node_payload, dict):
        return updates

    if terminal_state == SupervisorTerminalState.completed.value:
        message = "활성 Agent 노드가 남아 있어 Supervisor 실행을 완료할 수 없습니다."
        updates = {
            **updates,
            "terminal_state": SupervisorTerminalState.failed_terminal.value,
            "final_answer": message,
            "error_state": {
                "node": "finalize",
                "message": message,
                "reason_code": "active_node_in_completed_state",
                "retryable": False,
            },
        }
        terminal_state = SupervisorTerminalState.failed_terminal.value

    if terminal_state not in {
        SupervisorTerminalState.failed_terminal.value,
        SupervisorTerminalState.failed_with_recoverable_context.value,
    }:
        return updates

    message = str(
        updates.get("final_answer")
        or "Supervisor 실행이 완료되지 못하고 종료되었습니다."
    )
    error_state = updates.get("error_state")
    reason_code = (
        str(error_state.get("reason_code") or "supervisor_terminal_failure")
        if isinstance(error_state, dict)
        else "supervisor_terminal_failure"
    )
    agent_name = active_node_payload.get("agent_name")
    failed_state, failed_node, failed_event_type = fail_active_node(
        {**state, **updates},
        agent_name=agent_name,
        reason=message,
        reason_code=reason_code,
    )
    emit_node_lifecycle_event(
        backend_adapter,
        failed_state,
        failed_event_type,
        failed_node,
        message,
        action=next(
            (
                action
                for action, mapped_agent in ACTION_TO_AGENT.items()
                if mapped_agent == failed_node.agent_name
            ),
            "",
        ),
        metadata={
            "terminal_state": terminal_state,
            "reason_code": reason_code,
        },
    )

    failed_agents = list(state.get("failed_agents", []))
    if failed_node.agent_name not in failed_agents:
        failed_agents.append(failed_node.agent_name)
    return {
        **updates,
        "active_node": None,
        "failed_agents": failed_agents,
    }


def make_finalize_node(
    model: Any | None,
    backend_adapter: Any | None = None,
):
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
            return _close_active_node_for_terminal_state(
                state,
                _decision_failure_updates(state, "finalize", exc),
                backend_adapter,
            )

        terminal_state = decision.terminal_state
        current_terminal_state = state.get("terminal_state")
        if current_terminal_state in FINALIZE_PROTECTED_TERMINAL_STATES:
            terminal_state = current_terminal_state
        elif terminal_state == SupervisorTerminalState.needs_user_approval.value:
            # finalize_node는 실제 재개 가능한 interrupt/pending_approval을 만들 방법이
            # 없다. 이전 노드가 이미 승인 대기를 걸어둔 경우(위 protected 분기)가 아니라면
            # 재개 불가능한 "승인 대기" 라벨만 붙이고 끝나버려, 사용자가 approve해도
            # "승인 대기 상태가 아닙니다" 에러가 난다(pending_approval 없음). 그런 경우엔
            # completed로 처리하고 caveat은 final_answer에 그대로 담아 사용자가 읽게 한다.
            terminal_state = SupervisorTerminalState.completed.value

        final_answer = state.get("final_answer") or decision.final_answer
        if (
            terminal_state == SupervisorTerminalState.completed.value
            and not isinstance(state.get("active_node"), dict)
        ):
            completion = _check_completion_readiness(state)
            if completion.status != "ready":
                terminal_state = SupervisorTerminalState.failed_terminal.value
                final_answer = completion.reason
        updates: SupervisorState = {
            "terminal_state": terminal_state,
            "final_answer": final_answer,
            "next_action": "finalize",
            "current_step": "finalize",
            "llm_decisions": _append_llm_decision(state, "finalize", decision),
        }
        return _close_active_node_for_terminal_state(
            state,
            updates,
            backend_adapter,
        )

    return finalize_node


def clarify_query_node(state: SupervisorState) -> SupervisorState:
    return make_clarify_query_node(None)(state)


def create_analysis_plan_node(state: SupervisorState) -> SupervisorState:
    return make_create_analysis_plan_node(None)(state)


def collect_clarification_node(state: SupervisorState) -> SupervisorState:
    return make_collect_clarification_node()(state)


def finalize_node(state: SupervisorState) -> SupervisorState:
    return make_finalize_node(None)(state)


def completion_guard_node(state: SupervisorState) -> SupervisorState:
    return make_completion_guard_node()(state)


def build_graph(
    subagent_adapter: Any,
    model: Any | None = None,
    checkpointer: Any | None = None,
    *,
    insight_generator: Any | None = None,
    analysis_rule_search: Any | None = None,
):
    if insight_generator is None:
        if hasattr(subagent_adapter, "generate_insight"):
            insight_generator = _InsightGeneratorAdapter(subagent_adapter)
        else:
            backend_adapter = getattr(subagent_adapter, "backend_adapter", subagent_adapter)
            from DATA_Analyst_Assistant_Agent.supervisor.insighting import SupervisorInsightGenerator

            insight_generator = SupervisorInsightGenerator(backend_adapter)

    graph = StateGraph(SupervisorState)
    backend_adapter = getattr(subagent_adapter, "backend_adapter", None)
    graph.add_node(
        "retrieve_analysis_rules",
        make_retrieve_analysis_rules_node(analysis_rule_search, model),
    )
    graph.add_node("clarify_query", make_clarify_query_node(model))
    graph.add_node("collect_clarification", make_collect_clarification_node())
    graph.add_node("create_analysis_plan", make_create_analysis_plan_node(model))
    graph.add_node("decide_next_action", make_decide_next_action_node(model))
    graph.add_node("completion_guard", make_completion_guard_node())
    graph.add_node("execute_subagent", make_execute_subagent_node(subagent_adapter, model))
    graph.add_node("generate_insight", make_generate_insight_node(insight_generator))
    graph.add_node("validate_candidate", make_validate_candidate_node(model))
    graph.add_node("commit_candidate", make_commit_candidate_node(backend_adapter))
    graph.add_node("collect_analysis_review", make_collect_analysis_review_node())
    graph.add_node("resolve_analysis_review", make_resolve_analysis_review_node(backend_adapter))
    graph.add_node("finalize", make_finalize_node(model, backend_adapter))

    graph.add_edge(START, "retrieve_analysis_rules")
    graph.add_edge("retrieve_analysis_rules", "clarify_query")
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
            "generate_insight": "generate_insight",
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
            "finalize": "finalize",
        },
    )
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
    if next_action == "call_insight":
        return "generate_insight"
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
    if state.get("next_action") == "call_insight":
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
    if next_action == "call_insight":
        return "generate_insight"
    if next_action in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    if next_action in {"finalize", "fail"}:
        return "completion_guard"
    raise ValueError(f"commit_candidate 이후 지원하지 않는 next_action입니다: {next_action!r}")


def _route_after_completion_guard(state: SupervisorState) -> str:
    if state.get("next_action") == "call_insight":
        return "generate_insight"
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
