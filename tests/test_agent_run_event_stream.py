from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend.agent_runs.event_stream import stream_run_events


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


class RunningRunService:
    def list_events(self, run_id: str):
        return []

    def get_run(self, run_id: str):
        return SimpleNamespace(status="running")


def test_non_terminal_stream_rotates_without_waiting_for_run_completion() -> None:
    async def collect() -> list[str]:
        return [
            message
            async for message in stream_run_events(
                ConnectedRequest(),
                RunningRunService(),
                "run_waiting",
                None,
                poll_interval=0,
                max_connection_seconds=0,
            )
        ]

    assert asyncio.run(collect()) == []
