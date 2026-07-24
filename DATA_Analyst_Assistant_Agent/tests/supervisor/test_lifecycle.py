from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.supervisor.lifecycle import (
    emit_node_lifecycle_event,
    emit_supervisor_selection_started,
)
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    ActiveNodeExecution,
    begin_or_retry_agent_node,
    empty_supervisor_state,
)


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


def test_supervisor_selection_reserves_the_next_agent_node_id() -> None:
    backend = RecordingBackend()

    node_id = emit_supervisor_selection_started(
        backend,
        run_id="run_001",
        node_sequence=2,
        parent_node_id="run_001:node:1",
        selection_reason="next_action",
    )

    assert node_id == "run_001:node:2"
    assert backend.events == [
        {
            "run_id": "run_001",
            "event_type": "supervisor.selection.started",
            "message": "다음 Agent를 선택하고 있습니다.",
            "event_key": "agent-selection:run_001:node:2:started",
            "node_name": "supervisor",
            "metadata": {
                "node_id": "run_001:node:2",
                "parent_node_id": "run_001:node:1",
                "node_sequence": 2,
                "agent_name": "supervisor",
                "status": "selecting",
                "selection_reason": "next_action",
            },
        }
    ]


def test_reserved_selection_id_matches_the_started_agent_node_id() -> None:
    state = empty_supervisor_state(
        thread_id="thread_001",
        run_id="run_001",
        user_query="매출을 분석해줘",
        datasource_id=None,
    )
    reserved_node_id = emit_supervisor_selection_started(
        None,
        run_id="run_001",
        node_sequence=1,
        parent_node_id=None,
        selection_reason="initial",
    )

    _, active_node, event_type = begin_or_retry_agent_node(state, "sql_agent")

    assert event_type == "agent.started"
    assert active_node.node_id == reserved_node_id
