from __future__ import annotations

import json

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from DATA_Analyst_Assistant_Agent.supervisor.decision import (
    AnalysisPlanDecision,
    ClarificationDecision,
    ExecutionGuardDecision,
    FinalizationDecision,
    ResultValidationDecision,
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
    decide_next_action,
    invoke_supervisor_decision,
    parse_decision_json,
    parse_decision_json_as,
)
from DATA_Analyst_Assistant_Agent.supervisor.prompts import (
    CLARIFY_DECISION_PROMPT,
    DECIDE_NEXT_ACTION_PROMPT,
    EXECUTION_GUARD_DECISION_PROMPT,
    RESULT_VALIDATION_DECISION_PROMPT,
    SEMANTIC_VALIDATION_ADVISORY_PROMPT,
    STEP_SUMMARY_DECISION_PROMPT,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    empty_supervisor_state,
    stage_candidate_result,
)


@dataclass
class FakeMessage:
    content: str


class FakeModel:
    def invoke(self, messages):
        return FakeMessage('{"next_action":"call_eda_agent","reason":"SQL 결과를 탐색합니다."}')


class FencedJsonModel:
    def invoke(self, messages):
        return FakeMessage(
            '판단 결과입니다.\n```json\n{"next_action":"call_analysis_agent","reason":"분석 완료"}\n```'
        )


class InvalidJsonModel:
    def invoke(self, messages):
        return FakeMessage("다음 단계는 SQL입니다.")


class ExceptionModel:
    def invoke(self, messages):
        raise RuntimeError("model unavailable")


class InvalidActionModel:
    def invoke(self, messages):
        return FakeMessage('{"next_action":"call_unknown_agent","reason":"잘못된 액션"}')


class RecordingModel:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeMessage('{"next_action":"call_sql_agent","reason":"스냅샷 확인"}')


def test_parse_decision_json_extracts_next_action() -> None:
    decision = parse_decision_json('{"next_action":"call_analysis_agent","reason":"EDA 완료"}')

    assert decision.next_action == "call_analysis_agent"
    assert decision.reason == "EDA 완료"


def test_parse_decision_json_extracts_fenced_json() -> None:
    decision = parse_decision_json(
        '판단 결과입니다.\n```json\n{"next_action":"call_analysis_agent","reason":"분석 완료"}\n```'
    )

    assert decision.next_action == "call_analysis_agent"
    assert decision.reason == "분석 완료"


def test_parse_decision_json_extracts_json_from_surrounding_text() -> None:
    decision = parse_decision_json(
        '다음 JSON을 사용하세요: {"next_action":"finalize","reason":"리포트 완료"} 감사합니다.'
    )

    assert decision.next_action == "finalize"
    assert decision.reason == "리포트 완료"


def test_analysis_plan_decision_recovers_derivation_shaped_payload_without_goal() -> None:
    decision = parse_decision_json_as(
        json.dumps(
            {
                "name": "delivery_days",
                "purpose": "delivery days calculation",
                "source_columns": [
                    "orders.order_purchase_timestamp",
                    "orders.order_delivered_customer_date",
                ],
                "definition": "purchase timestamp to delivered customer date in days",
            }
        ),
        AnalysisPlanDecision,
    )

    assert decision.goal == "delivery days calculation"
    assert decision.required_derivations[0]["name"] == "delivery_days"
    assert decision.required_derivations[0]["definition"] == (
        "purchase timestamp to delivered customer date in days"
    )


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (
            SupervisorDecision,
            {"next_action": "call_sql_agent", "reason": "SQL 실행"},
        ),
        (
            ClarificationDecision,
            {
                "needs_clarification": False,
                "clarified_query": "월별 매출 분석",
                "clarification_question": "",
                "input_mode": "free_text",
                "options": [],
                "allow_free_text": True,
                "reason": "충분함",
            },
        ),
        (
            AnalysisPlanDecision,
            {
                "goal": "월별 매출 분석",
                "route_kind": "trend",
                "steps": ["SQL", "EDA"],
                "metric": "매출",
                "dimension": "월",
                "filters": [],
                "requires_mart_review": False,
                "required_derivations": [],
                "analysis_heuristics": [],
                "reason": "추이 분석",
            },
        ),
        (
            ExecutionGuardDecision,
            {"allowed": True, "next_action": "call_sql_agent", "reason": "실행 가능"},
        ),
        (
            ResultValidationDecision,
            {
                "valid": True,
                "next_action": "decide_next_action",
                "reason": "유효함",
                    "terminal_state": "running",
                    "final_answer": "",
                    "decision": "accept",
                    "reason_code": "none",
                    "failure_reason": "",
                    "repeated_failure": False,
                    "failure_streak": None,
                },
        ),
        (
            SemanticValidationAdvisoryDecision,
            {
                "semantic_valid": True,
                "severity": "info",
                "recommended_next_action": "",
                "reason": "사용자 요청과 결과가 정렬되어 있습니다.",
                "missing_evidence": [],
                "alignment_notes": ["계획의 SQL 단계가 충족되었습니다."],
            },
        ),
        (
            StepSummaryDecision,
            {
                "step": "validate_subagent_result",
                "agent": "sql_agent",
                "action": "call_sql_agent",
                "summary": "SQL 완료",
                "artifact_ids": ["artifact_sql"],
                "next_action": "decide_next_action",
                "reason": "요약",
            },
        ),
        (
            FinalizationDecision,
            {
                "terminal_state": "completed",
                "final_answer": "완료",
                "next_action": "finalize",
                "reason": "종료",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda text: text,
        lambda text: f"판단 결과입니다.\n```json\n{text}\n```",
        lambda text: f"다음 JSON을 사용하세요: {text} 감사합니다.",
    ],
)
def test_parse_decision_json_as_supports_all_decision_schemas(schema, payload, wrapper) -> None:
    decision = parse_decision_json_as(wrapper(json.dumps(payload, ensure_ascii=False)), schema)

    assert decision.model_dump(mode="json") == payload


def test_decide_next_action_uses_model_json_when_available() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=FakeModel())

    assert decision.next_action == "call_eda_agent"


def test_decide_next_action_uses_model_fenced_json_when_available() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=FencedJsonModel())

    assert decision.next_action == "call_analysis_agent"


def test_decide_next_action_requires_model() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    with pytest.raises(RuntimeError, match="Supervisor LLM decision model is required."):
        decide_next_action(state, model=None)


def test_decide_next_action_propagates_invalid_json_error() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    with pytest.raises(ValueError, match="No JSON object found in decision text"):
        decide_next_action(state, model=InvalidJsonModel())


def test_decide_next_action_propagates_model_exception() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        decide_next_action(state, model=ExceptionModel())


def test_decide_next_action_propagates_invalid_model_action() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    with pytest.raises(ValidationError):
        decide_next_action(state, model=InvalidActionModel())


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (SupervisorDecision, {"next_action": "decide_next_action"}),
        (
            ExecutionGuardDecision,
            {"allowed": True, "next_action": "decide_next_action"},
        ),
        (
            SemanticValidationAdvisoryDecision,
            {
                "semantic_valid": True,
                "recommended_next_action": "decide_next_action",
            },
        ),
    ],
)
def test_llm_selectable_decisions_reject_internal_redecision_action(schema, payload) -> None:
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


@pytest.mark.parametrize("next_action", ["clarify", "create_plan"])
@pytest.mark.parametrize("schema", [SupervisorDecision, ExecutionGuardDecision])
def test_execution_decisions_reject_initial_only_actions(schema, next_action: str) -> None:
    payload = {"next_action": next_action}
    if schema is ExecutionGuardDecision:
        payload["allowed"] = False

    with pytest.raises(ValidationError):
        schema.model_validate(payload)


@pytest.mark.parametrize("next_action", ["clarify", "create_plan"])
def test_semantic_validation_rejects_initial_only_recommendation(next_action: str) -> None:
    with pytest.raises(ValidationError):
        SemanticValidationAdvisoryDecision.model_validate(
            {
                "semantic_valid": False,
                "severity": "error",
                "recommended_next_action": next_action,
                "reason": "초기 전용 action은 권고할 수 없습니다.",
            }
        )


def test_semantic_validation_prompt_describes_missing_evidence_advisory_policy() -> None:
    recommendation_section = SEMANTIC_VALIDATION_ADVISORY_PROMPT.split(
        "허용 recommended_next_action:",
        maxsplit=1,
    )[1].split("반드시 JSON 객체만 반환하세요.", maxsplit=1)[0]

    assert "그 존재만으로 복구 또는 severity=error를 선택하지 마세요" in SEMANTIC_VALIDATION_ADVISORY_PROMPT
    assert "실제 역할 불이행이나 결과 모순" in SEMANTIC_VALIDATION_ADVISORY_PROMPT
    assert "severity=warning 또는 semantic_valid=true인 정보성 결과" in SEMANTIC_VALIDATION_ADVISORY_PROMPT
    assert "recommended_next_action으로 후속 작업을 권고" in SEMANTIC_VALIDATION_ADVISORY_PROMPT
    assert "- create_plan" not in recommendation_section


def test_semantic_validation_prompt_describes_lenient_analysis_rubric() -> None:
    assert "Partial or inspectability-only gaps are not semantic invalidity" in SEMANTIC_VALIDATION_ADVISORY_PROMPT
    assert "For analysis_agent specifically, prefer semantic_valid=true" in SEMANTIC_VALIDATION_ADVISORY_PROMPT
    assert "Do not use error for\n  missing_evidence alone" in SEMANTIC_VALIDATION_ADVISORY_PROMPT


@pytest.mark.parametrize("schema", [ResultValidationDecision, StepSummaryDecision])
def test_internal_transition_decisions_allow_redecision_action(schema) -> None:
    payload = (
        {"valid": True, "next_action": "decide_next_action"}
        if schema is ResultValidationDecision
        else {
            "step": "validate_subagent_result",
            "action": "call_sql_agent",
            "summary": "SQL 완료",
            "next_action": "decide_next_action",
        }
    )

    decision = schema.model_validate(payload)

    assert decision.next_action == "decide_next_action"


@pytest.mark.parametrize("next_action", ["clarify", "create_plan"])
@pytest.mark.parametrize("schema", [ResultValidationDecision, StepSummaryDecision])
def test_post_execution_decisions_reject_initial_only_actions(schema, next_action: str) -> None:
    payload = (
        {"valid": False, "next_action": next_action}
        if schema is ResultValidationDecision
        else {
            "step": "validate_subagent_result",
            "action": "call_sql_agent",
            "summary": "SQL 완료",
            "next_action": next_action,
        }
    )

    with pytest.raises(ValidationError):
        schema.model_validate(payload)


def test_decide_next_action_sends_compact_json_snapshot_to_model() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state["analysis_plan"] = {
        "goal": "매출 분석" * 1000,
        "steps": [{"name": f"step_{index}", "detail": "상세 설명" * 500} for index in range(20)],
    }
    state["analysis_rule_context"] = {
        "document_id": "sales_orders",
        "default_metrics": ["매출은 order_payments.payment_value 합계"],
    }
    state["clarification_answers"] = ["최근 6개월 기준"]
    state["validation_results"] = [
        {"agent": "sql_agent", "message": "검증 메시지" * 500, "nested": {"detail": "중첩" * 500}}
        for _ in range(20)
    ]
    state["semantic_validation_results"] = [
        {
            "agent": "sql_agent",
            "semantic_valid": False,
            "severity": "warning",
            "recommended_next_action": "call_eda_agent",
            "reason": "의미 검증 메시지" * 500,
            "missing_evidence": ["근거" * 200 for _ in range(20)],
            "alignment_notes": ["정렬 메모" * 200 for _ in range(20)],
        }
        for _ in range(20)
    ]
    state["step_summaries"] = [
        {"step": "execute_subagent", "summary": "요약" * 500, "items": ["항목" * 200 for _ in range(20)]}
        for _ in range(20)
    ]
    model = RecordingModel()

    decide_next_action(state, model=model)

    assert model.messages is not None
    assert len(model.messages) == 2
    assert model.messages[0]["role"] == "system"
    assert model.messages[1]["role"] == "user"
    assert len(model.messages[1]["content"]) <= 12500
    snapshot = json.loads(model.messages[1]["content"])
    assert snapshot["query"] == "월별 매출 추이를 분석해줘"
    assert snapshot["available_next_actions"] == [
        "call_sql_agent",
        "call_eda_agent",
        "call_analysis_agent",
        "finalize",
        "fail",
    ]
    assert "decide_next_action" not in snapshot["available_next_actions"]
    assert {capability["agent"] for capability in snapshot["agent_capabilities"]} == {
        "sql_agent",
        "eda_agent",
        "analysis_agent",
        "insight",
    }


def test_prompts_expose_redecision_action_only_to_internal_transition_models() -> None:
    assert "decide_next_action" not in DECIDE_NEXT_ACTION_PROMPT
    assert '"next_action":"decide_next_action"' in RESULT_VALIDATION_DECISION_PROMPT
    assert '"next_action":"decide_next_action"' in STEP_SUMMARY_DECISION_PROMPT


def test_clarification_prompt_limits_questions_to_user_owned_ambiguity() -> None:
    assert "사용자 답변 없이는 실행 방향이나 분석 범위를 확정할 수 없을 때만" in CLARIFY_DECISION_PROMPT
    assert "데이터/스키마 탐색" in CLARIFY_DECISION_PROMPT
    assert "하위 에이전트의 휴리스틱" in CLARIFY_DECISION_PROMPT
    assert "사용할 데이터소스, 테이블, 컬럼, 조인 경로" in CLARIFY_DECISION_PROMPT
    assert "결과에 기준·가정·한계를 명시" in CLARIFY_DECISION_PROMPT
    assert '"needs_clarification":false' in CLARIFY_DECISION_PROMPT
    assert '"needs_clarification":true' in CLARIFY_DECISION_PROMPT
    assert "RFM과 평균 리뷰점수" in CLARIFY_DECISION_PROMPT
    assert "최근 매출 추이" in CLARIFY_DECISION_PROMPT


@pytest.mark.parametrize(
    "prompt",
    [
        DECIDE_NEXT_ACTION_PROMPT,
        EXECUTION_GUARD_DECISION_PROMPT,
        RESULT_VALIDATION_DECISION_PROMPT,
        SEMANTIC_VALIDATION_ADVISORY_PROMPT,
        STEP_SUMMARY_DECISION_PROMPT,
    ],
)
def test_post_initial_prompts_do_not_expose_initial_only_actions(prompt: str) -> None:
    assert "- clarify" not in prompt
    assert "- create_plan" not in prompt


def test_invoke_supervisor_decision_requires_model() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    with pytest.raises(RuntimeError, match="Supervisor LLM decision model is required."):
        invoke_supervisor_decision(state, None, "prompt", SupervisorDecision, extra={})


def test_node_context_builders_are_bounded_and_include_required_keys() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="datasource_001",
        catalog_summary={"tables": [{"name": "sales", "detail": "상세" * 1000}] * 20},
    )
    state["analysis_plan"] = {
        "goal": "매출 분석" * 1000,
        "steps": [{"name": f"step_{index}", "detail": "상세 설명" * 500} for index in range(20)],
    }
    state["analysis_rule_context"] = {
        "document_id": "sales_orders",
        "default_metrics": ["매출은 order_payments.payment_value 합계"],
    }
    state["clarification_answers"] = ["최근 6개월 기준"]
    state["last_agent_result"] = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료" * 500,
        artifact_ids=["artifact_sql"],
    ).model_dump(mode="json")
    state["semantic_validation_results"] = [
        {
            "agent": "sql_agent",
            "semantic_valid": True,
            "severity": "info",
            "recommended_next_action": "",
            "reason": "정렬됨",
            "missing_evidence": [],
            "alignment_notes": ["월별 매출 계획과 SQL 결과가 맞습니다."],
        }
    ]

    contexts = [
        build_clarification_context(state),
        build_plan_context(state),
        build_next_action_context(state),
        build_execution_guard_context(state),
        build_result_validation_context(state),
        build_step_summary_context(state),
        build_finalization_context(state),
    ]

    assert "latest_user_query" in contexts[0]
    assert contexts[0]["clarification_answers"] == ["최근 6개월 기준"]
    assert contexts[0]["analysis_rule_context"]["document_id"] == "sales_orders"
    assert "query" in contexts[1]
    assert contexts[1]["analysis_rule_context"]["document_id"] == "sales_orders"
    assert "available_next_actions" in contexts[2]
    assert "agent_capabilities" in contexts[2]
    assert "requested_next_action" in contexts[3]
    assert "agent_capabilities" in contexts[3]
    assert "last_agent_result" in contexts[4]
    assert "pending_result" in contexts[4]
    assert "query" in contexts[4]
    assert "agent_capabilities" in contexts[4]
    assert "recent_semantic_validation_results" not in contexts[4]
    assert "latest_validation_result" in contexts[5]
    assert "latest_semantic_validation_result" in contexts[5]
    assert "terminal_state" in contexts[6]
    assert "semantic_validation_results" in contexts[2]
    assert "semantic_validation_results" in contexts[6]
    next_action_capabilities = contexts[2]["agent_capabilities"]
    guard_capabilities = contexts[3]["agent_capabilities"]
    validation_capabilities = contexts[4]["agent_capabilities"]
    assert next_action_capabilities[0]["agent"] == "sql_agent"
    assert next_action_capabilities[0]["action"] == "call_sql_agent"
    assert guard_capabilities[2]["requires_any_artifacts_from"] == ["sql_agent", "eda_agent"]
    assert validation_capabilities[3]["requires_any_artifacts_from"] == [
        "sql_agent",
        "eda_agent",
        "analysis_agent",
    ]
    assert all(len(json.dumps(context, ensure_ascii=False)) <= 12500 for context in contexts)


def test_result_validation_context_is_scoped_to_current_candidate() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="analyze sales",
        datasource_id=None,
    )
    old_record = {
        "candidate_id": "candidate_old",
        "validation_id": "validation_old",
        "agent": "eda_agent",
        "outcome": {
            "disposition": "recover",
            "reason": "stale missing evidence reason",
            "reason_code": "semantic_validation_failed",
            "retry_target": None,
            "terminal_state": "running",
        },
        "checks": [
            {"name": "result", "passed": True, "findings": [], "details": {}},
            {
                "name": "semantic",
                "passed": False,
                "findings": [],
                "details": {
                    "semantic_valid": False,
                    "reason": "stale missing evidence reason",
                    "missing_evidence": ["old artifact"],
                },
            },
        ],
    }
    result = AgentCompactResult(
        agent="eda_agent",
        status="success",
        summary="current candidate summary",
        artifact_ids=["eda_current"],
    )
    state = stage_candidate_result(state, result, {})
    state["last_agent_result"] = result.model_dump(mode="json")
    pending = state["pending_result"]
    state["validation_history"] = [
        old_record,
        {
            "candidate_id": pending["candidate_id"],
            "validation_id": pending["validation_id"],
            "agent": "eda_agent",
            "outcome": {
                "disposition": "accept",
                "reason": "current candidate hard checks passed",
                "reason_code": "none",
                "retry_target": None,
                "terminal_state": "running",
            },
            "checks": [
                {"name": "result", "passed": True, "findings": [], "details": {}},
            ],
        },
    ]

    context = build_result_validation_context(state)

    assert "recent_semantic_validation_results" not in context
    assert context["pending_result"]["candidate_id"] == pending["candidate_id"]
    assert len(context["validation_results"]) == 1
    assert context["validation_results"][0]["agent"] == "eda_agent"
    assert context["validation_results"][0]["decision"] == "accept"
    assert context["validation_results"][0]["reason"] == "current candidate hard checks passed"
    assert "stale missing evidence reason" not in json.dumps(context, ensure_ascii=False)


def test_next_action_context_preserves_capability_role_boundary_text() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    context = build_next_action_context(state)
    capabilities = {item["agent"]: item for item in context["agent_capabilities"]}

    assert "관찰 가능한 신호" in capabilities["eda_agent"]["description"]
    assert "분석 판단의 재료" in capabilities["eda_agent"]["when_to_use"]
    assert "기준값" in " ".join(capabilities["sql_agent"]["avoid_when"])
    assert "결론의 강도와 한계" in capabilities["analysis_agent"]["description"]


def test_finalization_context_includes_latest_validation_agent_failure_streak() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    analysis_streak = {
        "reason_code": "method_review_failed",
        "failure_reason": "wrong method",
        "signature": '["method_review_failed", "wrong method"]',
        "consecutive_count": 2,
    }
    state["failure_streaks"] = {
        "analysis_agent": analysis_streak,
        "sql_agent": {
            "reason_code": "timeout",
            "failure_reason": "timeout",
            "signature": '["timeout", "timeout"]',
            "consecutive_count": 1,
        },
    }
    state["validation_results"] = [{"agent": "analysis_agent", "valid": False}]

    context = build_finalization_context(state)

    assert context["recent_failure_streak"] == analysis_streak


def test_context_derives_legacy_payload_keys_from_validation_history() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state["validation_history"] = [
        {
            "candidate_id": "candidate_001",
            "validation_id": "validation_001",
            "agent": "analysis_agent",
            "outcome": {
                "disposition": "accept",
                "reason": "검증 통과",
                "reason_code": "none",
                "retry_target": None,
                "terminal_state": "running",
            },
            "checks": [
                {"name": "result", "passed": True, "findings": [], "details": {}},
                {
                    "name": "semantic",
                    "passed": True,
                    "findings": [],
                    "details": {
                        "semantic_valid": True,
                        "recommended_next_action": "finalize",
                    },
                },
            ],
        }
    ]

    context = build_next_action_context(state)

    assert context["validation_results"][0]["agent"] == "analysis_agent"
    assert context["validation_results"][0]["decision"] == "accept"
    assert context["semantic_validation_results"][0]["semantic_valid"] is True
    assert "validation_history" not in context
