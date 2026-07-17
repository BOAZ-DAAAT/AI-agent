from __future__ import annotations

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.common import BackendError
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter


def _adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


def test_save_workspace_file_is_not_part_of_backend_core(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    run = adapter.create_run()

    with pytest.raises(BackendError) as exc_info:
        adapter.save_workspace_file(run.run_id, "# report")

    assert exc_info.value.code == "UNSUPPORTED_OPERATION"
