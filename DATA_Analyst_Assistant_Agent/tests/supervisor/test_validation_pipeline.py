from __future__ import annotations

import json
from types import SimpleNamespace

from DATA_Analyst_Assistant_Agent.shared.contracts import ApprovalRequirement, ValidationFinding
from DATA_Analyst_Assistant_Agent.supervisor.candidate import commit_candidate
from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    make_validate_candidate_node,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    begin_or_retry_agent_node,
    empty_supervisor_state,
    stage_candidate_result,
)


class SemanticModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


class FailingSemanticModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        raise RuntimeError("semantic model unavailable")


class RecoveringSemanticModel(SemanticModel):
    def invoke(self, messages):
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError("temporary semantic failure")
        return super().invoke(messages)


def _state(result: AgentCompactResult):
    state = empty_supervisor_state(
        thread_id="thread_001",
        run_id="run_001",
        user_query="매출을 분석해줘",
        datasource_id="ds_001",
    )
    staged = stage_candidate_result(state, result, {})
    staged["last_agent_result"] = result.model_dump(mode="json")
    return staged


def _semantic_success(next_action: str = "") -> dict[str, object]:
    return {
        "semantic_valid": True,
        "severity": "info",
        "recommended_next_action": next_action,
        "reason": "의미 검증 통과",
        "missing_evidence": [],
        "alignment_notes": [],
    }


def test_validate_candidate_runs_all_checks_in_order_and_accepts() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )
    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "semantic"
    ]
    assert record["outcome"]["disposition"] == "accept"
    assert model.calls == 1
    assert "validation_history" not in updates


def test_validate_candidate_short_circuits_after_retryable_result_failure() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
        error="구문 오류",
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == ["contract", "result"]
    assert record["outcome"]["disposition"] == "retry"
    assert model.calls == 0


def test_approval_contract_mismatch_stops_before_result_validation() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="잘못된 승인 계약",
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == ["contract"]
    assert record["checks"][0]["passed"] is False
    assert record["outcome"]["reason_code"] == "approval_contract_mismatch"
    assert model.calls == 0


def test_approval_is_decided_only_after_result_and_semantic_checks() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="승인 필요",
        artifact_ids=["analysis_001"],
        approval=ApprovalRequirement(required=True, reason="검토가 필요합니다."),
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "semantic"
    ]
    assert record["outcome"]["disposition"] == "await_approval"


def test_required_approval_takes_priority_over_analysis_semantic_limitation() -> None:
    model = SemanticModel(
        {
            **_semantic_success(),
            "semantic_valid": False,
            "severity": "warning",
            "reason": "analysis answer needs human review before promotion",
            "missing_evidence": ["monthly_metric_evidence"],
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="approval required",
        artifact_ids=["analysis_001"],
        approval=ApprovalRequirement(required=True, reason="review before using this analysis"),
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["name"] == "semantic"
    assert record["checks"][-1]["findings"][0]["disposition"] == "limitation"
    assert record["outcome"]["disposition"] == "await_approval"


def test_limitation_finding_becomes_accept_with_limitations() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="warning",
        summary="제한이 있는 분석",
        artifact_ids=["analysis_001"],
        findings=[
            ValidationFinding(
                code="small_sample",
                source="analysis_agent",
                severity="warning",
                disposition="limitation",
                message="표본이 작습니다.",
            )
        ],
    )

    updates = make_validate_candidate_node(SemanticModel(_semantic_success()))(_state(result))

    assert updates["pending_validation"]["outcome"]["disposition"] == "accept_with_limitations"


def test_success_without_artifacts_reaches_semantic_validation() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="근거 없는 분석",
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "semantic"
    ]
    assert record["outcome"]["disposition"] == "accept"
    assert model.calls == 1


def test_semantic_info_without_reason_or_missing_evidence_is_accepted() -> None:
    model = SemanticModel({**_semantic_success(), "reason": ""})
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["passed"] is True
    assert record["checks"][-1]["findings"] == []
    assert record["checks"][-1]["details"]["missing_evidence"] == []
    assert record["outcome"]["disposition"] == "accept"


def test_semantic_info_with_missing_evidence_is_accepted_with_limitations() -> None:
    model = SemanticModel({**_semantic_success("call_sql_agent"), "missing_evidence": ["analysis_table"]})
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "semantic"
    ]
    semantic_check = record["checks"][-1]
    assert semantic_check["passed"] is True
    assert semantic_check["findings"][0]["disposition"] == "limitation"
    assert semantic_check["details"]["missing_evidence"] == ["analysis_table"]
    assert record["outcome"]["disposition"] == "accept_with_limitations"
    assert record["outcome"]["recovery_action"] is None


def test_semantic_missing_evidence_without_reason_generates_limitation_message() -> None:
    model = SemanticModel(
        {
            **_semantic_success(),
            "reason": "",
            "missing_evidence": ["analysis_table", "segment_summary"],
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    expected = "누락 근거: analysis_table, segment_summary"
    assert record["checks"][-1]["findings"][0]["message"] == expected
    assert record["outcome"]["reason"] == expected


def test_semantic_warning_with_invalid_flag_is_accepted_with_limitations() -> None:
    model = SemanticModel(
        {
            **_semantic_success(),
            "semantic_valid": False,
            "severity": "warning",
            "reason": "질문과 분석이 일치하지 않습니다.",
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )
    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["name"] == "semantic"
    assert record["checks"][-1]["passed"] is True
    assert record["checks"][-1]["findings"][0]["disposition"] == "limitation"
    assert record["outcome"]["disposition"] == "accept_with_limitations"


def test_semantic_warning_without_missing_evidence_is_accepted_with_limitations() -> None:
    model = SemanticModel(
        {
            **_semantic_success(),
            "severity": "warning",
            "reason": "일부 기간 데이터가 희소합니다.",
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["passed"] is True
    assert record["checks"][-1]["findings"][0]["message"] == "일부 기간 데이터가 희소합니다."
    assert record["outcome"]["disposition"] == "accept_with_limitations"


def test_semantic_warning_with_missing_evidence_is_accepted_with_limitations() -> None:
    model = SemanticModel(
        {
            **_semantic_success(),
            "severity": "warning",
            "reason": "집계 근거가 일부 누락되었습니다.",
            "missing_evidence": ["monthly_sales"],
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["passed"] is True
    assert record["checks"][-1]["findings"][0]["disposition"] == "limitation"
    assert record["checks"][-1]["details"]["missing_evidence"] == ["monthly_sales"]
    assert record["outcome"]["disposition"] == "accept_with_limitations"


def test_analysis_semantic_info_with_invalid_flag_is_accepted_with_limitations() -> None:
    model = SemanticModel(
        {
            **_semantic_success("call_sql_agent"),
            "semantic_valid": False,
            "reason": "응답의 유효성 표시가 모순됩니다.",
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["passed"] is True
    assert record["checks"][-1]["findings"][0]["disposition"] == "limitation"
    assert record["outcome"]["disposition"] == "accept_with_limitations"
    assert record["outcome"]["recovery_action"] is None


def test_analysis_semantic_error_is_accepted_with_limitations() -> None:
    model = SemanticModel(
        {
            **_semantic_success("call_eda_agent"),
            "severity": "error",
            "reason": "필수 분석이 누락되었습니다.",
            "missing_evidence": ["segment_summary"],
        }
    )
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )
    state = _state(result)

    updates = make_validate_candidate_node(model)(state)

    record = updates["pending_validation"]
    assert record["checks"][-1]["passed"] is True
    assert record["checks"][-1]["findings"][0]["severity"] == "warning"
    assert record["checks"][-1]["findings"][0]["disposition"] == "limitation"
    assert record["checks"][-1]["details"]["missing_evidence"] == ["segment_summary"]
    assert record["outcome"]["disposition"] == "accept_with_limitations"
    assert record["outcome"]["recovery_action"] is None
    committed = commit_candidate({**state, **updates}, None)
    assert len(committed["agent_results"]) == 1
    assert committed["rejected_results"] == []


def test_same_analysis_recommendation_retries_on_the_existing_node() -> None:
    first_result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="검정 근거가 부족한 분석",
        artifact_ids=["analysis_attempt_1"],
    )
    state, first_node, activation = begin_or_retry_agent_node(
        empty_supervisor_state(
            thread_id="thread_001",
            run_id="run_001",
            user_query="배송 지연과 리뷰 평점 관계를 분석해줘",
            datasource_id="ds_001",
        ),
        "analysis_agent",
    )
    assert activation == "agent.started"

    first_staged = stage_candidate_result(state, first_result, {})
    first_updates = make_validate_candidate_node(
        SemanticModel(
            {
                **_semantic_success("call_analysis_agent"),
                "semantic_valid": False,
                "severity": "error",
                "reason": "가설 검정 결과가 부족합니다.",
            }
        )
    )(first_staged)

    first_record = first_updates["pending_validation"]
    assert first_record["outcome"]["disposition"] == "retry"
    assert first_record["outcome"]["reason_code"] == "semantic_same_agent_retry"
    assert first_record["checks"][-1]["passed"] is False

    retry_state = commit_candidate({**first_staged, **first_updates}, None)
    assert retry_state["agent_results"] == []
    assert retry_state["completed_agents"] == []
    assert retry_state["retry_counts"] == {"analysis_agent": 1}
    assert retry_state["active_node"]["node_id"] == first_node.node_id

    retry_state, retried_node, retry_activation = begin_or_retry_agent_node(
        retry_state,
        "analysis_agent",
    )
    assert retry_activation == "agent.retrying"
    assert retried_node.node_id == first_node.node_id
    assert retried_node.node_sequence == first_node.node_sequence
    assert retried_node.attempt == 2

    final_result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="검정 근거를 보완한 분석",
        artifact_ids=["analysis_attempt_2"],
    )
    final_staged = stage_candidate_result(retry_state, final_result, {})
    final_updates = make_validate_candidate_node(
        SemanticModel(_semantic_success())
    )(final_staged)
    completed = commit_candidate({**final_staged, **final_updates}, None)

    assert completed["active_node"] is None
    assert completed["last_completed_node_id"] == first_node.node_id
    assert completed["node_sequence"] == first_node.node_sequence
    assert completed["completed_agents"] == ["analysis_agent"]
    assert len(completed["agent_results"]) == 1
    assert completed["agent_results"][0]["summary"] == "검정 근거를 보완한 분석"


def test_same_analysis_recommendation_does_not_loop_after_retry_budget() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="재시도 한도에 도달한 분석",
        artifact_ids=["analysis_final_attempt"],
    )
    state, node, _ = begin_or_retry_agent_node(
        empty_supervisor_state(
            thread_id="thread_001",
            run_id="run_001",
            user_query="배송 지연과 리뷰 평점 관계를 분석해줘",
            datasource_id="ds_001",
        ),
        "analysis_agent",
    )
    state["retry_counts"] = {"analysis_agent": state["max_retry_per_agent"]}
    staged = stage_candidate_result(state, result, {})
    updates = make_validate_candidate_node(
        SemanticModel(
            {
                **_semantic_success("call_analysis_agent"),
                "semantic_valid": False,
                "severity": "error",
                "reason": "추가 검정 근거가 필요합니다.",
            }
        )
    )(staged)

    assert updates["pending_validation"]["outcome"]["disposition"] == "accept_with_limitations"
    completed = commit_candidate({**staged, **updates}, None)

    assert completed["next_action"] == "decide_next_action"
    assert completed["last_completed_node_id"] == node.node_id
    assert completed["completed_agents"] == ["analysis_agent"]
    assert len(completed["agent_results"]) == 1


def test_analysis_semantic_model_failure_retries_once_then_accepts_with_limitations() -> None:
    model = FailingSemanticModel()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )
    state = _state(result)

    updates = make_validate_candidate_node(model)(state)

    assert model.calls == 2
    assert updates["pending_validation"]["outcome"] == {
        "disposition": "accept_with_limitations",
        "reason": "semantic validation 모델 호출에 반복 실패했습니다: semantic model unavailable",
        "reason_code": "semantic_model_failed",
        "retry_target": None,
        "recovery_action": None,
        "terminal_state": "running",
    }
    semantic_check = updates["pending_validation"]["checks"][-1]
    assert semantic_check["passed"] is True
    assert semantic_check["findings"][0]["disposition"] == "limitation"
    assert semantic_check["details"] == {
        "attempts": 2,
        "failure_reason": "semantic model unavailable",
    }
    committed = commit_candidate({**state, **updates}, None)
    assert len(committed["agent_results"]) == 1
    assert committed["rejected_results"] == []


def test_analysis_semantic_contract_violation_retries_then_accepts_with_limitations() -> None:
    model = SemanticModel(_semantic_success("create_plan"))
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    assert model.calls == 2
    assert updates["pending_validation"]["outcome"]["disposition"] == "accept_with_limitations"
    assert updates["pending_validation"]["outcome"]["reason_code"] == "semantic_model_failed"


def test_other_agent_semantic_error_still_recovers() -> None:
    model = SemanticModel(
        {
            **_semantic_success("call_eda_agent"),
            "severity": "error",
            "reason": "필수 근거가 누락되었습니다.",
        }
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["sql_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["passed"] is False
    assert record["outcome"]["disposition"] == "recover"
    assert record["outcome"]["recovery_action"] == "call_eda_agent"


def test_other_agent_semantic_model_failure_still_rejects() -> None:
    model = FailingSemanticModel()
    result = AgentCompactResult(
        agent="eda_agent",
        status="success",
        summary="EDA 완료",
        artifact_ids=["eda_001"],
    )

    updates = make_validate_candidate_node(model)(_state(result))

    assert model.calls == 2
    assert updates["pending_validation"]["outcome"]["disposition"] == "reject"
    assert updates["pending_validation"]["checks"][-1]["passed"] is False


def test_semantic_model_first_failure_retries_once_and_accepts() -> None:
    model = RecoveringSemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )
    state = _state(result)

    updates = make_validate_candidate_node(model)(state)

    candidate_id = state["pending_result"]["candidate_id"]
    assert model.calls == 2
    assert updates["pending_validation"]["outcome"]["disposition"] == "accept"
    assert updates["semantic_retry_counts"] == {candidate_id: 1}


def test_resolve_validation_records_and_rejects_retry_candidate_once() -> None:
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
        error="구문 오류",
    )
    state = _state(result)
    validated = make_validate_candidate_node(SemanticModel(_semantic_success()))(state)

    resolved = commit_candidate({**state, **validated}, None)

    assert len(resolved["validation_history"]) == 1
    assert len(resolved["rejected_results"]) == 1
    assert resolved["pending_result"] is None
    assert resolved["retry_counts"] == {"sql_agent": 1}
    assert resolved["next_action"] == "call_sql_agent"
    assert resolved["pending_validation"] is None


def test_validation_preserves_staged_content_hashes() -> None:
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="승인 필요",
        artifacts=[
            {
                "artifact_id": "analysis_001",
                "type": "analysis",
                "content_hash": "sha256:original",
            }
        ],
        approval=ApprovalRequirement(required=True, reason="검토가 필요합니다."),
    )
    state = _state(result)

    updates = make_validate_candidate_node(SemanticModel(_semantic_success()))(state)

    assert state["pending_result"]["content_hashes"] == {
        "analysis_001": "sha256:original"
    }
    assert updates["pending_result"]["content_hashes"] == {
        "analysis_001": "sha256:original"
    }
