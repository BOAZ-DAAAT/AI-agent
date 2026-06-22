from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, dump_result
from data_agent_backend.mcp.tools_runs import run_append_event, run_create, run_get, run_list, run_list_events, run_summary, run_update_status
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/runs", tags=["runs"])


class RunCreateRequest(BackendModel):
    run_id: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    metadata: dict[str, Any] | None = None
    context: ContextPayload = None


class RunGetRequest(BackendModel):
    run_id: str
    context: ContextPayload = None


class RunListRequest(BackendModel):
    thread_id: str | None = None
    project_id: str | None = None
    status: str | None = None
    context: ContextPayload = None


class RunUpdateStatusRequest(BackendModel):
    run_id: str
    status: str
    metadata: dict[str, Any] | None = None
    context: ContextPayload = None


class RunAppendEventRequest(BackendModel):
    run_id: str
    event_type: str
    message: str
    node_name: str | None = None
    tool_name: str | None = None
    artifact_ids: list[str] | None = None
    approval_id: str | None = None
    memory_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None
    context: ContextPayload = None


@router.post("/create")
def create_run(payload: RunCreateRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        run_create(
            run_id=payload.run_id,
            thread_id=payload.thread_id,
            project_id=payload.project_id,
            metadata=payload.metadata,
            context=payload.context,
            services=services,
        )
    )


@router.post("/get")
def get_run(payload: RunGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(run_get(run_id=payload.run_id, context=payload.context, services=services))


@router.post("/list")
def list_runs(payload: RunListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        run_list(
            thread_id=payload.thread_id,
            project_id=payload.project_id,
            status=payload.status,
            context=payload.context,
            services=services,
        )
    )


@router.post("/update-status")
def update_run_status(payload: RunUpdateStatusRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        run_update_status(
            run_id=payload.run_id,
            status=payload.status,
            metadata=payload.metadata,
            context=payload.context,
            services=services,
        )
    )


@router.post("/events/append")
def append_run_event(payload: RunAppendEventRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        run_append_event(
            run_id=payload.run_id,
            event_type=payload.event_type,
            message=payload.message,
            node_name=payload.node_name,
            tool_name=payload.tool_name,
            artifact_ids=payload.artifact_ids,
            approval_id=payload.approval_id,
            memory_ids=payload.memory_ids,
            metadata=payload.metadata,
            context=payload.context,
            services=services,
        )
    )


@router.post("/events/list")
def list_run_events(payload: RunGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(run_list_events(run_id=payload.run_id, context=payload.context, services=services))


@router.post("/summary")
def summarize_run(payload: RunGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(run_summary(run_id=payload.run_id, context=payload.context, services=services))
