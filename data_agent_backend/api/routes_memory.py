from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, dump_result
from data_agent_backend.mcp.tools_memory import memory_get, memory_list, memory_propose, memory_search
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
        memory_propose(
            namespace=payload.namespace,
            type=payload.type,
            content=payload.content,
            source=payload.source,
            metadata=payload.metadata,
            context=payload.context,
            services=services,
        )
    )


@router.post("/list")
def list_memory(payload: MemoryListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(memory_list(namespace=payload.namespace, type=payload.type, context=payload.context, services=services))


@router.post("/get")
def get_memory(payload: MemoryGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(memory_get(memory_id=payload.memory_id, context=payload.context, services=services))


@router.post("/search")
def search_memory(payload: MemorySearchRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(memory_search(namespace=payload.namespace, query=payload.query, type=payload.type, context=payload.context, services=services))

