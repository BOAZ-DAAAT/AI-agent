from __future__ import annotations

import json

from dataclasses import dataclass

from DATA_Analyst_Assistant_Agent.supervisor.decision import decide_next_action, parse_decision_json
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, empty_supervisor_state, merge_agent_result
from DATA_Analyst_Assistant_Agent.supervisor.summarizer import summarize_agent_step


@dataclass
class FakeMessage:
    content: str


class FakeModel:
    def invoke(self, messages):
        return FakeMessage('{"next_action":"call_eda_agent","reason":"SQL 결과를 탐색합니다."}')


class FencedJsonModel:
    def invoke(self, messages):
        return FakeMessage(
            '판단 결과입니다.\n```json\n{"next_action":"call_report_agent","reason":"분석 완료"}\n```'
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
        '판단 결과입니다.\n```json\n{"next_action":"call_report_agent","reason":"분석 완료"}\n```'
    )

    assert decision.next_action == "call_report_agent"
    assert decision.reason == "분석 완료"


def test_parse_decision_json_extracts_json_from_surrounding_text() -> None:
    decision = parse_decision_json(
        '다음 JSON을 사용하세요: {"next_action":"finalize","reason":"리포트 완료"} 감사합니다.'
    )

    assert decision.next_action == "finalize"
    assert decision.reason == "리포트 완료"


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

    assert decision.next_action == "call_report_agent"


def test_decide_next_action_fallback_runs_sql_only_without_evidence() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=None)

    assert decision.next_action == "call_sql_agent"
    assert "fallback" in decision.reason


def test_decide_next_action_fallback_handles_invalid_json() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=InvalidJsonModel())

    assert decision.next_action == "call_sql_agent"
    assert "fallback" in decision.reason


def test_decide_next_action_fallback_handles_model_exception() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=ExceptionModel())

    assert decision.next_action == "call_sql_agent"
    assert "fallback" in decision.reason


def test_decide_next_action_fallback_ignores_completed_agent_without_artifact_evidence() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=[],
        ),
    )

    decision = decide_next_action(state, model=None)

    assert decision.next_action == "call_sql_agent"
    assert "fallback" in decision.reason


def test_decide_next_action_fallback_fails_after_evidence_exists() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql_result"],
        ),
    )

    decision = decide_next_action(state, model=None)

    assert decision.next_action == "fail"
    assert "fallback" in decision.reason


def test_decide_next_action_fallback_rejects_invalid_model_action_after_evidence_exists() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql_result"],
        ),
    )

    decision = decide_next_action(state, model=InvalidActionModel())

    assert decision.next_action == "fail"
    assert "fallback" in decision.reason


def test_decide_next_action_fallback_finalizes_after_report_agent_completed() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="report_agent",
            status="success",
            summary="리포트 완료",
            artifact_ids=["artifact_report"],
        ),
    )

    decision = decide_next_action(state, model=InvalidJsonModel())

    assert decision.next_action == "finalize"
    assert "fallback" in decision.reason


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
    state["validation_results"] = [
        {"agent": "sql_agent", "message": "검증 메시지" * 500, "nested": {"detail": "중첩" * 500}}
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
    assert len(model.messages[1]["content"]) <= 6000
    snapshot = json.loads(model.messages[1]["content"])
    assert snapshot["query"] == "월별 매출 추이를 분석해줘"


def test_summarize_agent_step_is_compact() -> None:
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["artifact_sql_result"],
    )
    summary = summarize_agent_step("execute_subagent", result, next_action="call_eda_agent")

    assert summary.step == "execute_subagent"
    assert summary.agent == "sql_agent"
    assert summary.action == "call_sql_agent"
    assert summary.artifact_ids == ["artifact_sql_result"]


def test_summarize_agent_step_truncates_summary_to_1000_chars() -> None:
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="가" * 1001,
        artifact_ids=[],
    )

    summary = summarize_agent_step("execute_subagent", result, next_action="call_eda_agent")

    assert len(summary.summary) == 1000
