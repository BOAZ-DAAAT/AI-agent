from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.supervisor.state import (
    ActiveNodeExecution,
    CompletedNodeExecution,
    FailedNodeExecution,
    NodeLifecycleEventType,
    SupervisorState,
)


LifecycleNode = ActiveNodeExecution | CompletedNodeExecution | FailedNodeExecution


def emit_node_lifecycle_event(
    backend_adapter: Any | None,
    state: SupervisorState,
    event_type: NodeLifecycleEventType,
    node: LifecycleNode,
    message: str,
    *,
    action: str = "",
    approval_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    append_event = getattr(backend_adapter, "append_run_event", None)
    run_id = str(state.get("current_run_id") or "")

    if append_event is None or not run_id:
        return

    event_metadata = {
        **node.model_dump(mode="json"),
        "action": action,
        **dict(metadata or {}),
    }

    artifact_ids = (
        list(node.summary.artifact_ids)
        if isinstance(node, CompletedNodeExecution)
        else []
    )

    append_event(
        run_id,
        event_type,
        message,
        node_name=node.agent_name,
        artifact_ids=artifact_ids,
        approval_id=approval_id,
        metadata=event_metadata,
    )
