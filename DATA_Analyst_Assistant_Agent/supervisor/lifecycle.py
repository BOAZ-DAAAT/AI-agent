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


def supervisor_selection_node_id(run_id: str, node_sequence: int) -> str:
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("Supervisor 선택 이벤트에는 run_id가 필요합니다.")
    if node_sequence < 1:
        raise ValueError("Supervisor 선택 이벤트의 node_sequence는 1 이상이어야 합니다.")
    return f"{normalized_run_id}:node:{node_sequence}"


def emit_supervisor_selection_started(
    backend_adapter: Any | None,
    *,
    run_id: str,
    node_sequence: int,
    parent_node_id: str | None,
    selection_reason: str,
) -> str:
    node_id = supervisor_selection_node_id(run_id, node_sequence)
    append_event = getattr(backend_adapter, "append_run_event", None)
    if append_event is None:
        return node_id

    append_event(
        run_id,
        "supervisor.selection.started",
        "다음 Agent를 선택하고 있습니다.",
        event_key=f"agent-selection:{node_id}:started",
        node_name="supervisor",
        metadata={
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "node_sequence": node_sequence,
            "agent_name": "supervisor",
            "status": "selecting",
            "selection_reason": selection_reason,
        },
    )
    return node_id


def _lifecycle_event_key(
    event_type: NodeLifecycleEventType,
    node: LifecycleNode,
) -> str:
    return f"agent-node:{node.node_id}:{event_type}:attempt:{node.attempt}"


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
        event_key=_lifecycle_event_key(event_type, node),
        node_name=node.agent_name,
        artifact_ids=artifact_ids,
        approval_id=approval_id,
        metadata=event_metadata,
    )
