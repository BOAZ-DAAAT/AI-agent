from __future__ import annotations

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.common import BackendError
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


def test_claim_waiting_input_is_atomic(tmp_path) -> None:
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id="thread_claim")
    services.run_service.update_status(run.run_id, RunStatus.running)
    services.run_service.update_status(run.run_id, RunStatus.waiting_input)

    claimed = services.run_service.claim_waiting_input(
        run.run_id,
        metadata={"resumed_from": "clarification"},
    )

    assert claimed.status == RunStatus.running
    assert claimed.metadata["resumed_from"] == "clarification"
    with pytest.raises(BackendError, match="not waiting") as exc_info:
        services.run_service.claim_waiting_input(run.run_id)
    assert exc_info.value.code == "RUN_NOT_WAITING_INPUT"


def test_claim_waiting_approval_is_atomic(tmp_path) -> None:
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id="thread_claim_approval")
    services.run_service.update_status(run.run_id, RunStatus.running)
    services.run_service.update_status(run.run_id, RunStatus.waiting_approval)

    claimed = services.run_service.claim_waiting_approval(
        run.run_id,
        metadata={"resumed_from": "approval"},
    )

    assert claimed.status == RunStatus.running
    assert claimed.metadata["resumed_from"] == "approval"
    with pytest.raises(BackendError, match="not waiting") as exc_info:
        services.run_service.claim_waiting_approval(run.run_id)
    assert exc_info.value.code == "RUN_NOT_WAITING_APPROVAL"


def test_claim_waiting_approval_does_not_claim_waiting_input(tmp_path) -> None:
    """waiting_approval 전용 claim이 waiting_input 상태의 run은 못 건드리는지 — 두 상태를 섞어 쓰지 않는지 확인."""
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id="thread_claim_mismatch")
    services.run_service.update_status(run.run_id, RunStatus.running)
    services.run_service.update_status(run.run_id, RunStatus.waiting_input)

    with pytest.raises(BackendError) as exc_info:
        services.run_service.claim_waiting_approval(run.run_id)
    assert exc_info.value.code == "RUN_NOT_WAITING_APPROVAL"
    assert services.run_service.get_run(run.run_id).status == RunStatus.waiting_input


def test_claim_created_does_not_restart_cancelled_run(tmp_path) -> None:
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id="thread_cancel_before_start")
    services.run_service.cancel_run(run.run_id, metadata={"cancel_reason": "user_requested"})

    with pytest.raises(BackendError) as exc_info:
        services.run_service.claim_created(run.run_id)

    assert exc_info.value.code == "RUN_NOT_CREATED"
    assert services.run_service.get_run(run.run_id).status == RunStatus.cancelled


@pytest.mark.parametrize(
    "status",
    [
        RunStatus.created,
        RunStatus.running,
        RunStatus.waiting_input,
        RunStatus.waiting_approval,
    ],
)
def test_cancel_run_atomically_cancels_active_statuses(tmp_path, status) -> None:
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id=f"thread_cancel_{status.value}")
    if status != RunStatus.created:
        services.run_service.update_status(run.run_id, status)

    cancelled = services.run_service.cancel_run(
        run.run_id,
        metadata={"cancel_reason": "user_requested"},
    )
    repeated = services.run_service.cancel_run(run.run_id)

    assert cancelled.status == RunStatus.cancelled
    assert cancelled.metadata["cancel_reason"] == "user_requested"
    assert repeated.status == RunStatus.cancelled


@pytest.mark.parametrize("status", [RunStatus.succeeded, RunStatus.failed])
def test_cancel_run_rejects_finished_statuses(tmp_path, status) -> None:
    services = create_backend_services(BackendConfig(base_data_dir=tmp_path / ".data_agent"))
    run = services.run_service.create_run(thread_id=f"thread_finished_{status.value}")
    services.run_service.update_status(run.run_id, status)

    with pytest.raises(BackendError) as exc_info:
        services.run_service.cancel_run(run.run_id)

    assert exc_info.value.code == "RUN_NOT_CANCELLABLE"
