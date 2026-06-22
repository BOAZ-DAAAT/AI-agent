from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
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
        result_wrap(
            lambda: services.run_service.create_run(
                payload.run_id,
                payload.thread_id,
                payload.project_id,
                payload.metadata,
                context_from(payload.context, "run_create"),
            )
        )
    )


@router.post("/get")
def get_run(payload: RunGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.run_service.get_run(payload.run_id, context_from(payload.context, "run_get"))))


@router.post("/list")
def list_runs(payload: RunListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(
            lambda: services.run_service.list_runs(
                payload.thread_id,
                payload.project_id,
                payload.status,
                context_from(payload.context, "run_list"),
            )
        )
    )


@router.post("/update-status")
def update_run_status(payload: RunUpdateStatusRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(
            lambda: services.run_service.update_status(
                payload.run_id,
                payload.status,
                payload.metadata,
                context_from(payload.context, "run_update_status"),
            )
        )
    )


@router.post("/events/append")
def append_run_event(payload: RunAppendEventRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(
            lambda: services.run_service.append_event(
                payload.run_id,
                payload.event_type,
                payload.message,
                payload.node_name,
                payload.tool_name,
                payload.artifact_ids,
                payload.approval_id,
                payload.memory_ids,
                payload.metadata,
                context_from(payload.context, "run_append_event"),
            )
        )
    )


@router.post("/events/list")
def list_run_events(payload: RunGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.run_service.list_events(payload.run_id, context_from(payload.context, "run_list_events"))))


@router.post("/summary")
def summarize_run(payload: RunGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.run_service.get_summary(payload.run_id, context_from(payload.context, "run_summary"))))
