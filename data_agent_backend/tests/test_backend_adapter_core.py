from __future__ import annotations

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType
from data_agent_backend.models.common import BackendError
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter


def _adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_run_sql_preview_registers_result_without_backend_sql_executor(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    run = adapter.create_run()

    ref = adapter.run_sql_preview(run.run_id, "SELECT 42 AS answer")

    assert ref.type == ArtifactType.sql_result
    assert ref.preview["row_count"] == 1
    assert ref.preview["columns"] == ["answer"]
    artifact = adapter.get_artifact(ref.artifact_id)
    assert artifact.metadata["execution_owner"] == "agent_runtime_preview"


def test_run_sql_preview_blocks_write_sql(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    run = adapter.create_run()

    with pytest.raises(BackendError) as exc_info:
        adapter.run_sql_preview(run.run_id, "DROP TABLE users")

    assert exc_info.value.code == "POLICY_BLOCKED"


def test_run_sql_preview_rejects_backend_datasource_execution(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    run = adapter.create_run()

    with pytest.raises(BackendError) as exc_info:
        adapter.run_sql_preview(run.run_id, "SELECT 1", datasource_id="ds_legacy")

    assert exc_info.value.code == "UNSUPPORTED_OPERATION"
