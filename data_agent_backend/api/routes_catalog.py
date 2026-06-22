from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, EmptyRequest, context_from, dump_result, result_wrap
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/catalog", tags=["catalog"])


class CatalogGetRequest(BackendModel):
    path_or_name: str
    context: ContextPayload = None


@router.post("/list")
def list_catalog(payload: EmptyRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.catalog_store.list(context_from(payload.context, "catalog_list"))))


@router.post("/get")
def get_catalog(payload: CatalogGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.catalog_store.read_text(payload.path_or_name, context_from(payload.context, "catalog_get"))))

