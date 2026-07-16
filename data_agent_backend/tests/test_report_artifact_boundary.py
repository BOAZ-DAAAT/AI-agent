from __future__ import annotations

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from data_agent_backend.models.common import BackendError
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.report import ReportGenerator
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState


def _adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def _register_sql(adapter: BackendAdapter, run_id: str) -> str:
    ref = adapter.register_artifact(
        run_id,
        ArtifactType.sql_result,
        content_text="sample_value\r\n1\r\n",
        filename="result.csv",
        created_by_tool="test.sql",
        preview={"row_count": 1, "columns": ["sample_value"], "sample_rows": [{"sample_value": 1}]},
    )
    return ref.artifact_id


def test_save_workspace_file_is_not_part_of_backend_core(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    run = adapter.create_run()

    with pytest.raises(BackendError) as exc_info:
        adapter.save_workspace_file(run.run_id, "# report")

    assert exc_info.value.code == "UNSUPPORTED_OPERATION"


def test_report_agent_registers_report_artifact_directly(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    run = adapter.create_run()
    sql_id = _register_sql(adapter, run.run_id)
    state = OrchestrationState(
        run_id=run.run_id,
        user_query="간단한 요약",
        artifact_ids={"sql_agent": [sql_id]},
        plan=AnalysisPlan(goal="summary", route_kind="simple"),
        route_kind="simple",
    )

    envelope = ReportGenerator().run(state, AgentRuntime(adapter))
    artifact = adapter.get_artifact(envelope.artifact_ids()[0])

    assert artifact.type == ArtifactType.report
    assert artifact.metadata["kind"] == "final_report"
    assert sql_id in (artifact.metadata.get("source_artifact_ids") or [])
