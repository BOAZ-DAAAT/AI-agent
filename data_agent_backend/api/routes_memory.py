from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryProposeRequest(BackendModel):
    namespace: list[str]
    type: str
    content: Any
    source: dict[str, Any]
    metadata: dict[str, Any] | None = None
    context: ContextPayload = None


class MemoryListRequest(BackendModel):
    namespace: list[str]
    type: str | None = None
    context: ContextPayload = None


class MemoryGetRequest(BackendModel):
    memory_id: str
    context: ContextPayload = None


class MemorySearchRequest(BackendModel):
    namespace: list[str]
    query: str
    type: str | None = None
    context: ContextPayload = None


@router.post("/propose")
def propose_memory(payload: MemoryProposeRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(
            lambda: services.memory_store.propose_memory(
                payload.namespace,
                payload.type,
                payload.content,
                payload.source,
                payload.metadata,
                context_from(payload.context, "memory_propose"),
            )
        )
    )


@router.post("/list")
def list_memory(payload: MemoryListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.memory_store.list_memory(payload.namespace, payload.type)))


@router.post("/get")
def get_memory(payload: MemoryGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.memory_store.get_memory(payload.memory_id)))


@router.post("/search")
def search_memory(payload: MemorySearchRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.memory_store.search_memory(payload.query, payload.namespace, payload.type)))

