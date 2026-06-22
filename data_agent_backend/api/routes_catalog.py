from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, EmptyRequest, dump_result
from data_agent_backend.mcp.tools_catalog import catalog_get, catalog_list
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/catalog", tags=["catalog"])


class CatalogGetRequest(BackendModel):
    path_or_name: str
    context: ContextPayload = None


@router.post("/list")
def list_catalog(payload: EmptyRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(catalog_list(context=payload.context, services=services))


@router.post("/get")
def get_catalog(payload: CatalogGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(catalog_get(path_or_name=payload.path_or_name, context=payload.context, services=services))

