from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.agent_runs import service
from backend.main import create_app
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.report.schemas import FindingSection, ReportResult
from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from data_agent_backend.services.factory import create_backend_services

TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _services(tmp_path):
    return create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _report() -> ReportResult:
    return ReportResult(
        title="월별 주문 분석 리포트",
        executive_summary="월별 주문 흐름을 확인했습니다.",
        background_and_question="사용자는 월별 주문 추이를 질문했습니다.",
        methodology_narrative="주문 데이터를 집계하고 분석했습니다.",
        key_findings=[FindingSection(heading="주요 흐름", body="주문 추이가 확인됐습니다.")],
        limitations=["표본 기간이 제한적입니다."],
        conclusion_and_recommendations="추가 기간을 포함해 확인해야 합니다.",
        key_finding="월별 주문 추이가 확인됐습니다.",
        included_stages=["sql", "analysis", "insight"],
    )


def test_generate_node_report_uses_selected_insight_lineage(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    adapter = BackendAdapter(services=services)
    run = services.run_service.create_run(thread_id="thread_report", project_id="session_1")

    sql = adapter.register_artifact(
        run.run_id,
        ArtifactType.sql_result,
        content_text="month,count\n2026-01,10\n",
        filename="result.csv",
        created_by_tool="test",
        metadata={"kind": "sql_result"},
    )
    analysis = adapter.register_artifact(
        run.run_id,
        ArtifactType.file,
        content_text='{"title":"analysis"}',
        filename="analysis.json",
        created_by_tool="test",
        parent_ids=[sql.artifact_id],
        metadata={"kind": "analysis_result"},
    )
    insight = adapter.register_artifact(
        run.run_id,
        ArtifactType.file,
        content_text='{"answer":"insight"}',
        filename="insight.json",
        created_by_tool="test",
        parent_ids=[analysis.artifact_id],
        metadata={"kind": "insight_payload"},
    )
    final_report = adapter.register_artifact(
        run.run_id,
        ArtifactType.report,
        content_text="# Existing insight report",
        filename="final_report.md",
        created_by_tool="test",
        parent_ids=[analysis.artifact_id],
        metadata={"kind": "final_report"},
    )
    services.run_service.append_event(
        run.run_id,
        "agent.completed",
        "insight 작업을 완료했습니다.",
        node_name="insight",
        artifact_ids=[final_report.artifact_id, insight.artifact_id],
        metadata={"node_id": "node_insight", "agent_name": "insight"},
    )

    captured: list[str] = []

    def fake_generate_report(source_ids, runtime):
        captured.extend(source_ids)
        payload = _report()
        return runtime.adapter.register_artifact(
            run.run_id,
            ArtifactType.file,
            content_text=json.dumps(payload.model_dump(), ensure_ascii=False),
            filename="report.json",
            created_by_tool="test.report",
            parent_ids=source_ids,
            metadata={"kind": "report"},
        )

    monkeypatch.setattr(service, "generate_report", fake_generate_report)

    result = service.generate_node_report(
        services=services,
        run_id=run.run_id,
        node_id="node_insight",
    )

    assert captured[0] == insight.artifact_id
    assert set(captured) == {
        insight.artifact_id,
        final_report.artifact_id,
        analysis.artifact_id,
        sql.artifact_id,
    }
    assert result.report.title == "월별 주문 분석 리포트"
    assert result.report_artifact_id

    stored = service.list_session_reports(services=services, session_id="session_1")
    assert [item.report_artifact_id for item in stored] == [result.report_artifact_id]
    assert stored[0].report.key_finding == "월별 주문 추이가 확인됐습니다."


def test_generate_node_report_rejects_non_insight_node(tmp_path) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(thread_id="thread_report", project_id="session_1")
    services.run_service.append_event(
        run.run_id,
        "agent.completed",
        "analysis 작업을 완료했습니다.",
        node_name="analysis_agent",
        artifact_ids=[],
        metadata={"node_id": "node_analysis", "agent_name": "analysis_agent"},
    )

    with pytest.raises(service.NodeReportGenerationError, match="Insight"):
        service.generate_node_report(
            services=services,
            run_id=run.run_id,
            node_id="node_analysis",
        )


def test_report_create_and_list_routes_return_report_contract(tmp_path, monkeypatch) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(
        thread_id="thread_report_api",
        project_id="session_1",
        metadata={"session_id": "session_1"},
    )
    report = _report()
    stored = service.StoredReportLookup(
        run_id=run.run_id,
        report_artifact_id="artifact_report",
        created_at="2026-07-22T00:00:00+00:00",
        report=report,
    )

    monkeypatch.setattr("backend.auth.deps.Auth.ENABLED", False)
    monkeypatch.setattr(
        "backend.agent_runs.routes.get_owned_session",
        lambda session_id, username: SimpleNamespace(id=session_id),
    )
    monkeypatch.setattr(
        "backend.agent_runs.routes.generate_node_report",
        lambda **kwargs: service.NodeReportLookup(
            node_id="node_insight",
            report_artifact_id=stored.report_artifact_id,
            created_at=stored.created_at,
            report=stored.report,
        ),
    )
    monkeypatch.setattr(
        "backend.agent_runs.routes.list_session_reports",
        lambda **kwargs: [stored],
    )

    client = TestClient(create_app(services=services))
    headers = {"Authorization": "Bearer ignored"}
    create_response = client.post(
        f"/agent-runs/{run.run_id}/nodes/node_insight/report",
        headers=headers,
    )
    list_response = client.get(
        "/agent-runs/reports",
        params={"session_id": "session_1"},
        headers=headers,
    )

    assert create_response.status_code == 200
    assert create_response.json()["report"]["title"] == report.title
    assert list_response.status_code == 200
    assert list_response.json()["reports"][0]["report_artifact_id"] == "artifact_report"
