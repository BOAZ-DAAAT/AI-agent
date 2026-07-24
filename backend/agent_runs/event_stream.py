from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from data_agent_backend.models.runs import TERMINAL_RUN_STATUSES, RunEvent


EVENT_STREAM_POLL_INTERVAL_SECONDS = 0.5
EVENT_STREAM_HEARTBEAT_SECONDS = 15.0
EVENT_STREAM_MAX_CONNECTION_SECONDS = 10.0


def sse_message(
    event: str,
    data: dict[str, Any],
    *,
    event_id: str | None = None,
) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}"
    )
    return "\n".join(lines) + "\n\n"


def seen_event_ids(
    events: list[RunEvent],
    last_event_id: str | None,
) -> set[str]:
    if not last_event_id:
        return set()

    seen: set[str] = set()
    for event in events:
        seen.add(event.event_id)
        if event.event_id == last_event_id:
            return seen

    # An unknown cursor is replayed from the beginning to avoid missing events.
    return set()


async def stream_run_events(
    request: Request,
    run_service: Any,
    run_id: str,
    last_event_id: str | None,
    *,
    poll_interval: float,
    max_connection_seconds: float = EVENT_STREAM_MAX_CONNECTION_SECONDS,
) -> AsyncIterator[str]:
    initial_events = await asyncio.to_thread(run_service.list_events, run_id)
    seen_ids = seen_event_ids(initial_events, last_event_id)
    terminal_idle_polls = 0
    connection_started = time.monotonic()
    last_heartbeat = connection_started

    while True:
        if await request.is_disconnected():
            return

        events = await asyncio.to_thread(run_service.list_events, run_id)
        new_events = [event for event in events if event.event_id not in seen_ids]
        for event in new_events:
            seen_ids.add(event.event_id)
            yield sse_message(
                "run.event",
                event.model_dump(mode="json"),
                event_id=event.event_id,
            )

        run = await asyncio.to_thread(run_service.get_run, run_id)
        if run.status in TERMINAL_RUN_STATUSES:
            terminal_idle_polls = 0 if new_events else terminal_idle_polls + 1
            if terminal_idle_polls >= 2:
                yield sse_message(
                    "run.closed",
                    {"run_id": run_id, "status": run.status.value},
                )
                return
        else:
            terminal_idle_polls = 0

        now = time.monotonic()
        # Uvicorn reload가 장기 SSE 종료를 무한히 기다리지 않도록 연결을 주기적으로 교체한다.
        # 클라이언트는 마지막 event_id를 커서로 자동 재연결하므로 이벤트는 유실되지 않는다.
        if now - connection_started >= max_connection_seconds:
            return
        if now - last_heartbeat >= EVENT_STREAM_HEARTBEAT_SECONDS:
            yield ": heartbeat\n\n"
            last_heartbeat = now

        await asyncio.sleep(poll_interval)
