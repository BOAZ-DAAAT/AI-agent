from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.report import ReportAgent
from DATA_Analyst_Assistant_Agent.agents.report.builder import build_report
from DATA_Analyst_Assistant_Agent.agents.report.service import generate_report_envelope
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, LocalCheck, OrchestrationState
from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph, make_generate_report_node
from DATA_Analyst_Assistant_Agent.supervisor.reporting import SupervisorReportGenerator
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
    merge_agent_result,
)
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentToolResult, default_agents


@pytest.fixture()
def adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _register_sql(adapter: BackendAdapter, run_id: str, content: str = "value\r\n1\r\n") -> str:
    return adapter.register_artifact(
        run_id,
        ArtifactType.sql_result,
        content_text=content,
        filename="result.csv",
        created_by_tool="test.sql",
        metadata={"kind": "sql_result"},
        preview={"row_count": 1, "columns": ["value"]},
    ).artifact_id


def _orchestration_state(run_id: str, sql_id: str, *, query: str = "매출을 요약해줘") -> OrchestrationState:
    return OrchestrationState(
        run_id=run_id,
        user_query=query,
        goal=query,
        artifact_ids={"sql_agent": [sql_id]},
        plan=AnalysisPlan(goal=query, route_kind="simple"),
        route_kind="simple",
    )


def _supervisor_state(run_id: str, sql_id: str | None = None) -> dict[str, Any]:
    state = empty_supervisor_state(
        thread_id="thread_report_001",
        run_id=run_id,
        user_query="매출을 요약해줘",
        datasource_id=None,
    )
    if sql_id is None:
        return state
    return merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=[sql_id],
            artifacts=[
                ArtifactSummary(
                    artifact_id=sql_id,
                    type="sql_result",
                    kind="sql_result",
                )
            ],
        ),
    )


def test_generate_report_service_registers_expected_boundary(adapter: BackendAdapter) -> None:
    run = adapter.create_run(thread_id="thread_report_001")
    sql_id = _register_sql(adapter, run.run_id)

    envelope = generate_report_envelope(
        _orchestration_state(run.run_id, sql_id),
        AgentRuntime(adapter),
        node_name="generate_report",
    )

    artifact = adapter.get_artifact(envelope.artifact_ids()[0])
    assert envelope.agent_name == "report_agent"
    assert artifact.type == ArtifactType.report
    assert artifact.created_by_tool == "report_agent.final_report"
    assert artifact.created_by_node == "generate_report"
    assert artifact.metadata["kind"] == "final_report"
    assert artifact.parent_ids == [sql_id]


def test_report_omits_analysis_decision_section_without_history() -> None:
    state = OrchestrationState(run_id="run_001", user_query="매출", goal="매출")

    report = build_report(state)

    assert "## 사용자 분석 결정" not in report


def test_report_renders_option_and_free_text_analysis_decisions() -> None:
    state = OrchestrationState(
        run_id="run_001",
        user_query="매출",
        goal="매출",
        analysis_review_decisions=[
            {
                "approval_id": "approval_option",
                "candidate_id": "candidate_001",
                "review_request": {"requires_followup_analysis": True},
                "selection_response": {"selected_option_id": "median", "free_text": None},
                "selected_option": {
                    "id": "median",
                    "label": "중앙값",
                    "method": "50% 분위수",
                },
            },
            {
                "approval_id": "approval_text",
                "candidate_id": "candidate_002",
                "review_request": {"requires_followup_analysis": False},
                "selection_response": {
                    "selected_option_id": None,
                    "free_text": "중앙값을 사용하고\n### 임의 제목은 만들지 마세요.",
                },
                "selected_option": None,
            },
        ],
    )

    report = build_report(state)

    assert "## 사용자 분석 결정" in report
    assert "approval_option" in report
    assert "median" in report
    assert "중앙값" in report
    assert "50% 분위수" in report
    assert "후속 분석: 필요" in report
    assert "approval_text" in report
    assert "> 중앙값을 사용하고" in report
    assert "> ### 임의 제목은 만들지 마세요." in report
    assert "후속 분석: 불필요" in report


def test_report_agent_remains_compatibility_wrapper(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)

    envelope = ReportAgent().run(_orchestration_state(run.run_id, sql_id), AgentRuntime(adapter))

    artifact = adapter.get_artifact(envelope.artifact_ids()[0])
    assert artifact.created_by_node == "report_agent"


def test_generate_report_reuses_same_content_artifact(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    state = _orchestration_state(run.run_id, sql_id)
    runtime = AgentRuntime(adapter)

    first = generate_report_envelope(state, runtime, node_name="generate_report")
    second = generate_report_envelope(state, runtime, node_name="generate_report")

    reports = adapter.list_artifacts(run_id=run.run_id, artifact_type=ArtifactType.report)
    assert first.artifact_ids() == second.artifact_ids()
    assert len(reports) == 1


def test_generate_report_creates_new_artifact_when_content_or_evidence_changes(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    first_sql_id = _register_sql(adapter, run.run_id)
    runtime = AgentRuntime(adapter)
    first_state = _orchestration_state(run.run_id, first_sql_id)
    first = generate_report_envelope(first_state, runtime, node_name="generate_report")

    changed_content = generate_report_envelope(
        _orchestration_state(run.run_id, first_sql_id, query="매출과 비용을 요약해줘"),
        runtime,
        node_name="generate_report",
    )
    second_sql_id = _register_sql(adapter, run.run_id, "value\r\n2\r\n")
    changed_evidence_state = _orchestration_state(run.run_id, first_sql_id)
    changed_evidence_state.artifact_ids["sql_agent"].append(second_sql_id)
    changed_evidence_state.artifact_ids["report_agent"] = first.artifact_ids()
    changed_evidence = generate_report_envelope(
        changed_evidence_state,
        runtime,
        node_name="generate_report",
    )

    reports = adapter.list_artifacts(run_id=run.run_id, artifact_type="report")
    assert len({first.artifact_ids()[0], changed_content.artifact_ids()[0], changed_evidence.artifact_ids()[0]}) == 3
    assert len(reports) == 3
    assert first.artifact_ids()[0] not in adapter.get_artifact(changed_evidence.artifact_ids()[0]).parent_ids


def test_generate_report_self_check_failure_returns_failed_envelope(
    adapter: BackendAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.report.service.run_report_self_check",
        lambda markdown: [LocalCheck(name="summary_present", passed=False, detail="요약 누락")],
    )

    envelope = generate_report_envelope(
        _orchestration_state(run.run_id, sql_id),
        AgentRuntime(adapter),
        node_name="generate_report",
    )

    assert envelope.status == "failed"
    assert envelope.artifact_ids() == []
    assert adapter.list_artifacts(run_id=run.run_id, artifact_type="report") == []


def test_supervisor_report_generator_compacts_report_envelope(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)

    result = SupervisorReportGenerator(adapter).generate(_supervisor_state(run.run_id, sql_id))

    assert result.agent == "report_agent"
    assert result.status == "success"
    assert result.artifact_ids
    assert result.artifacts[0].kind == "final_report"


def test_limited_report_uses_only_accepted_evidence_and_exposes_limitations(
    adapter: BackendAdapter,
) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    quarantined_id = _register_sql(adapter, run.run_id, "value\r\n999\r\n")
    state = _supervisor_state(run.run_id, sql_id)
    state["quarantined_artifacts"] = [{"artifact_id": quarantined_id}]
    state["limitations"] = [
        "필수 분석 근거가 누락되었습니다.",
        "필수 분석 근거가 누락되었습니다.",
        "승인된 기존 근거만 사용해 제한적 Report fallback을 생성합니다.",
    ]

    result = SupervisorReportGenerator(adapter).generate(state)

    artifact = adapter.get_artifact(result.artifact_ids[0])
    markdown = adapter.read_artifact_text(result.artifact_ids[0])
    assert artifact.parent_ids == [sql_id]
    assert quarantined_id not in artifact.parent_ids
    assert "## Limitations" in markdown
    assert markdown.count("필수 분석 근거가 누락되었습니다.") == 1
    assert "승인된 기존 근거만 사용해 제한적 Report fallback을 생성합니다." in markdown


@dataclass
class StubReportGenerator:
    result: AgentCompactResult | None = None
    error: Exception | None = None
    calls: int = 0

    def generate(self, state: dict[str, Any]) -> AgentCompactResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_generate_report_node_preserves_report_state_contract(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)

    updates = make_generate_report_node(SupervisorReportGenerator(adapter))(
        _supervisor_state(run.run_id, sql_id)
    )

    assert updates["next_action"] == "call_report_agent"
    assert updates["terminal_state"] == "running"
    assert updates["completed_agents"] == ["sql_agent"]
    assert updates["failed_agents"] == []
    assert updates["agent_results"][-1]["agent"] == "sql_agent"
    assert "report_agent" not in updates["artifacts"]
    assert updates["last_agent_result"]["agent"] == "report_agent"
    assert updates["pending_result"]["result"]["agent"] == "report_agent"


def test_generate_report_node_stages_self_check_failure_without_polluting_state(
    adapter: BackendAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.report.service.run_report_self_check",
        lambda markdown: [LocalCheck(name="summary_present", passed=False, detail="요약 누락")],
    )

    updates = make_generate_report_node(SupervisorReportGenerator(adapter))(
        _supervisor_state(run.run_id, sql_id)
    )

    assert updates["terminal_state"] == "running"
    assert updates["next_action"] == "call_report_agent"
    assert updates["pending_result"]["result"]["status"] == "failed"
    assert "자체 검사" in updates["pending_result"]["result"]["error"]
    assert "report_agent" not in updates["failed_agents"]


@pytest.mark.parametrize(
    "state_factory,generator,expected_message",
    [
        (lambda run_id, sql_id: _supervisor_state(run_id), StubReportGenerator(), "근거"),
        (
            lambda run_id, sql_id: _supervisor_state(run_id, sql_id),
            StubReportGenerator(
                result=AgentCompactResult(
                    agent="report_agent",
                    status="success",
                    summary="리포트 생성 완료",
                )
            ),
            "산출물 ID",
        ),
        (
            lambda run_id, sql_id: _supervisor_state(run_id, sql_id),
            StubReportGenerator(error=RuntimeError("저장 실패")),
            "저장 실패",
        ),
    ],
)
def test_generate_report_node_stages_failures_for_common_validation_pipeline(
    adapter: BackendAdapter,
    state_factory,
    generator: StubReportGenerator,
    expected_message: str,
) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)

    updates = make_generate_report_node(generator)(state_factory(run.run_id, sql_id))

    assert updates["terminal_state"] == "running"
    assert updates["next_action"] == "call_report_agent"
    assert "report_agent" not in updates["failed_agents"]
    assert "report_agent" not in updates["completed_agents"]
    pending_result = updates["pending_result"]["result"]
    if expected_message == "산출물 ID":
        assert pending_result["status"] == "success"
        assert pending_result["artifact_ids"] == []
    else:
        assert expected_message in pending_result["error"]


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class SequencedDecisionModel:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)

    def invoke(self, messages: list[dict[str, str]]) -> FakeMessage:
        import json

        return FakeMessage(json.dumps(self.decisions.pop(0), ensure_ascii=False))


class RecordingSubAgentAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, agent_name: str, state: dict[str, Any]) -> AgentToolResult:
        self.calls.append(agent_name)
        raise AssertionError("report는 SubAgentAdapter를 호출하면 안 됩니다.")


def test_call_report_agent_routes_directly_to_generate_report(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    subagents = RecordingSubAgentAdapter()
    report_generator = StubReportGenerator(
        result=AgentCompactResult(
            agent="report_agent",
            status="success",
            summary="리포트 완료",
            artifact_ids=["artifact_report"],
            artifacts=[ArtifactSummary(artifact_id="artifact_report", type="report", kind="final_report")],
        )
    )
    model = SequencedDecisionModel(
        [
            {
                "needs_clarification": False,
                "clarified_query": "매출을 요약해줘",
                "clarification_question": "",
                "reason": "충분함",
            },
            {
                "goal": "매출 요약",
                "route_kind": "simple",
                "steps": ["보고서 생성"],
                "metric": "매출",
                "dimension": None,
                "filters": [],
                "requires_mart_review": False,
                "reason": "계획 완료",
            },
                {"next_action": "call_report_agent", "reason": "근거 준비 완료"},
                {
                    "semantic_valid": True,
                    "severity": "info",
                    "recommended_next_action": "",
                    "reason": "근거와 리포트가 일치함",
                    "missing_evidence": [],
                    "alignment_notes": [],
                },
                {
                "terminal_state": "completed",
                "final_answer": "리포트 완료",
                "next_action": "finalize",
                "reason": "완료",
            },
        ]
    )
    graph = build_graph(subagents, report_generator=report_generator, model=model)

    result = graph.invoke(
        _supervisor_state(run.run_id, sql_id),
        {"configurable": {"thread_id": "thread_report_001"}},
    )

    assert subagents.calls == []
    assert report_generator.calls == 1
    assert result["completed_agents"] == ["sql_agent", "report_agent"]
    semantic_check = next(
        check
        for check in result["validation_history"][0]["checks"]
        if check["name"] == "semantic"
    )
    assert semantic_check["passed"] is True
    assert result["step_summaries"][-1]["step"] == "generate_report"


def test_execution_guard_redirect_to_report_skips_subagent_call(adapter: BackendAdapter) -> None:
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    subagents = RecordingSubAgentAdapter()
    report_generator = StubReportGenerator(
        result=AgentCompactResult(
            agent="report_agent",
            status="success",
            summary="리포트 완료",
            artifact_ids=["artifact_report"],
        )
    )
    model = SequencedDecisionModel(
        [
            {
                "needs_clarification": False,
                "clarified_query": "매출을 요약해줘",
                "clarification_question": "",
                "reason": "충분함",
            },
            {
                "goal": "매출 요약",
                "route_kind": "simple",
                "steps": ["보고서 생성"],
                "metric": "매출",
                "dimension": None,
                "filters": [],
                "requires_mart_review": False,
                "reason": "계획 완료",
            },
                {"next_action": "call_report_agent", "reason": "기존 근거로 보고서 생성"},
                {
                    "semantic_valid": True,
                    "severity": "info",
                    "recommended_next_action": "",
                    "reason": "근거와 리포트가 일치함",
                    "missing_evidence": [],
                    "alignment_notes": [],
                },
                {
                "terminal_state": "completed",
                "final_answer": "리포트 완료",
                "next_action": "finalize",
                "reason": "완료",
            },
        ]
    )
    graph = build_graph(subagents, report_generator=report_generator, model=model)

    result = graph.invoke(
        _supervisor_state(run.run_id, sql_id),
        {"configurable": {"thread_id": "thread_report_001"}},
    )

    assert subagents.calls == []
    assert report_generator.calls == 1
    assert result["completed_agents"] == ["sql_agent", "report_agent"]


def test_default_agents_excludes_report_agent() -> None:
    assert set(default_agents()) == {"sql_agent", "eda_agent", "analysis_agent"}
