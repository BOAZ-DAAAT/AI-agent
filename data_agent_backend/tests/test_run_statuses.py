from __future__ import annotations

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.runs import TERMINAL_RUN_STATUSES, RunStatus
from data_agent_backend.services.factory import create_backend_services


def test_waiting_input_status_is_valid_and_not_terminal(tmp_path) -> None:
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id="thread_waiting_001", metadata={"source": "test"})

    waiting = services.run_service.update_status(
        run.run_id,
        RunStatus.waiting_input,
        metadata={"reason": "clarification"},
    )

    assert waiting.status == RunStatus.waiting_input
    assert RunStatus.waiting_input not in TERMINAL_RUN_STATUSES
    assert services.run_service.get_run(run.run_id).status == RunStatus.waiting_input
    assert [item.run_id for item in services.run_service.list_runs(status=RunStatus.waiting_input)] == [run.run_id]
    assert services.run_service.get_summary(run.run_id).run.status == RunStatus.waiting_input

    running = services.run_service.update_status(run.run_id, RunStatus.running)
    succeeded = services.run_service.update_status(run.run_id, RunStatus.succeeded)

    assert running.status == RunStatus.running
    assert succeeded.status == RunStatus.succeeded
