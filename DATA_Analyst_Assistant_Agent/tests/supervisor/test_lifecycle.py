from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.supervisor.lifecycle import (
    emit_node_lifecycle_event,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import ActiveNodeExecution


class RecordingBackend:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append_run_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                **kwargs,
            }
        )


def test_lifecycle_event_key_changes_only_for_event_type_or_attempt() -> None:
    backend = RecordingBackend()
    state = {"current_run_id": "run_001"}
    first_attempt = ActiveNodeExecution(
        node_id="node_001",
        agent_name="sql_agent",
        node_sequence=1,
        attempt=1,
    )
    second_attempt = first_attempt.model_copy(update={"attempt": 2})

    emit_node_lifecycle_event(
        backend,
        state,
        "agent.started",
        first_attempt,
        "started",
    )
    emit_node_lifecycle_event(
        backend,
        state,
        "agent.started",
        first_attempt,
        "duplicate delivery",
    )
    emit_node_lifecycle_event(
        backend,
        state,
        "agent.retrying",
        second_attempt,
        "retrying",
    )

    assert [event["event_key"] for event in backend.events] == [
        "agent-node:node_001:agent.started:attempt:1",
        "agent-node:node_001:agent.started:attempt:1",
        "agent-node:node_001:agent.retrying:attempt:2",
    ]
