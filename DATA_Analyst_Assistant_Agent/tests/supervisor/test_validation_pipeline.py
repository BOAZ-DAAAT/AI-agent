from __future__ import annotations

import json
from types import SimpleNamespace

from DATA_Analyst_Assistant_Agent.shared.contracts import ApprovalRequirement, ValidationFinding
from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    make_resolve_validation_node,
    make_validate_candidate_node,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
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


class ForbiddenBackend:
    def get_artifact(self, _artifact_id: str):
        raise AssertionError("결정론 검증 실패 뒤에는 Evidence 검증을 호출하면 안 됩니다.")


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


def _semantic_success(next_action: str = "call_report_agent") -> dict[str, object]:
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

    updates = make_validate_candidate_node(model, None)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "evidence", "semantic"
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

    updates = make_validate_candidate_node(model, ForbiddenBackend())(_state(result))

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

    updates = make_validate_candidate_node(model, ForbiddenBackend())(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == ["contract"]
    assert record["checks"][0]["passed"] is False
    assert record["outcome"]["reason_code"] == "approval_contract_mismatch"
    assert model.calls == 0


def test_approval_is_decided_only_after_evidence_and_semantic_checks() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="approval_required",
        summary="승인 필요",
        artifact_ids=["analysis_001"],
        approval=ApprovalRequirement(required=True, reason="검토가 필요합니다."),
    )

    updates = make_validate_candidate_node(model, None)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "evidence", "semantic"
    ]
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

    updates = make_validate_candidate_node(
        SemanticModel(_semantic_success()), None
    )(_state(result))

    assert updates["pending_validation"]["outcome"]["disposition"] == "accept_with_limitations"


def test_missing_evidence_is_recorded_and_semantic_validation_is_skipped() -> None:
    model = SemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="근거 없는 분석",
    )

    updates = make_validate_candidate_node(model, None)(_state(result))

    record = updates["pending_validation"]
    assert [check["name"] for check in record["checks"]] == [
        "contract", "result", "evidence"
    ]
    assert record["checks"][-1]["passed"] is False
    assert record["outcome"]["disposition"] == "reject"
    assert model.calls == 0


def test_semantic_rejection_is_recorded_in_single_validation_record() -> None:
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

    updates = make_validate_candidate_node(model, None)(_state(result))

    record = updates["pending_validation"]
    assert record["checks"][-1]["name"] == "semantic"
    assert record["checks"][-1]["passed"] is False
    assert record["outcome"]["disposition"] == "reject"


def test_semantic_model_failure_retries_once_then_rejects() -> None:
    model = FailingSemanticModel()
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )

    updates = make_validate_candidate_node(model, None)(_state(result))

    assert model.calls == 2
    assert updates["pending_validation"]["outcome"] == {
        "disposition": "reject",
        "reason": "semantic validation 모델 호출에 반복 실패했습니다: semantic model unavailable",
        "reason_code": "semantic_model_failed",
        "retry_target": None,
        "terminal_state": "failed_with_recoverable_context",
    }


def test_semantic_model_first_failure_retries_once_and_accepts() -> None:
    model = RecoveringSemanticModel(_semantic_success())
    result = AgentCompactResult(
        agent="analysis_agent",
        status="success",
        summary="분석 완료",
        artifact_ids=["analysis_001"],
    )
    state = _state(result)

    updates = make_validate_candidate_node(model, None)(state)

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
    validated = make_validate_candidate_node(SemanticModel(_semantic_success()), None)(state)

    resolved = make_resolve_validation_node(None)({**state, **validated})

    assert len(resolved["validation_history"]) == 1
    assert len(resolved["rejected_results"]) == 1
    assert resolved["pending_result"] is None
    assert resolved["retry_counts"] == {"sql_agent": 1}
    assert resolved["next_action"] == "call_sql_agent"
    assert resolved["pending_validation"] is None
