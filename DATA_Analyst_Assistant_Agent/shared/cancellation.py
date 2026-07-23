from __future__ import annotations

from typing import Any


class RunCancellationRequested(Exception):
    """Raised when an active workflow reaches a safe cancellation boundary."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Agent run {run_id} was cancelled.")
        self.run_id = run_id


def raise_if_run_cancelled(backend_adapter: Any | None, run_id: str) -> None:
    if backend_adapter is None or not run_id:
        return
    checker = getattr(backend_adapter, "is_run_cancelled", None)
    if checker is not None and checker(run_id):
        raise RunCancellationRequested(run_id)
