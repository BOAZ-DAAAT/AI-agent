from __future__ import annotations

from data_agent_backend.agent.report.tools import (
    SaveReportArtifactRequest,
    get_artifact_preview_payload,
    get_run_summary_payload,
    save_report_artifact_payload,
)
from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactRegisterRequest, ArtifactType
from data_agent_backend.services.factory import create_core_services


def _services(tmp_path):
    return create_core_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_report_internal_tools_use_backend_services(tmp_path) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run(metadata={"source": "report-test"})
    artifact = services.artifact_registry.register_artifact(
        ArtifactRegisterRequest(
            run_id=run.run_id,
            type=ArtifactType.sql_result,
            content_text="category,total_orders\nA,10\n",
            filename="result.csv",
            created_by_tool="sql_run_query",
        )
    )

    summary = get_run_summary_payload(services, run.run_id)
    preview = get_artifact_preview_payload(services, artifact.artifact_id)

    assert summary["run"]["run_id"] == run.run_id
    assert preview["artifact"]["artifact_id"] == artifact.artifact_id
    assert preview["preview"]["columns"] == ["category", "total_orders"]


def test_save_report_artifact_creates_report_without_sensitive_metadata(tmp_path) -> None:
    services = _services(tmp_path)
    run = services.run_service.create_run()

    saved = save_report_artifact_payload(
        services,
        SaveReportArtifactRequest(
            run_id=run.run_id,
            title="카테고리 성과 리포트",
            report_markdown="# 카테고리 성과 리포트\n\n요약입니다.",
            artifact_refs=[{"artifact_id": "art_input", "type": "sql_result"}],
            metadata={"password": "secret", "credential": {"password": "secret"}, "safe": "ok"},
        ),
    )

    artifact = services.artifact_registry.get_artifact(saved["artifact_id"])

    assert artifact.type == ArtifactType.report
    assert artifact.created_by_tool == "create_report"
    assert artifact.metadata["title"] == "카테고리 성과 리포트"
    assert artifact.metadata["safe"] == "ok"
    assert "password" not in str(artifact.metadata)
    assert "credential" not in str(artifact.metadata)
