from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import dump_result
from data_agent_backend.models.common import BackendModel
from data_agent_backend.models.datasource import DatasourceCreateRequest, DatasourceCredential
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/datasources", tags=["datasources"])


class CredentialRequest(BackendModel):
    credential: DatasourceCredential


class SampleRowsRequest(BackendModel):
    credential: DatasourceCredential
    limit: int | None = None


@router.post("")
def create_datasource(payload: DatasourceCreateRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    record = services.datasource_service.register(payload)
    safe = next(item for item in services.datasource_service.list_agent_datasources() if item.datasource_id == record.datasource_id)
    return dump_result(ToolResult.success(safe))


@router.get("")
def list_datasources(services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(ToolResult.success(services.datasource_service.list_agent_datasources()))


@router.get("/{datasource_id}")
def get_datasource(datasource_id: str, services: BackendServices = Depends(get_backend_services)) -> dict:
    services.datasource_service.get(datasource_id)
    safe = next(item for item in services.datasource_service.list_agent_datasources() if item.datasource_id == datasource_id)
    return dump_result(ToolResult.success(safe))


@router.post("/{datasource_id}/test")
def test_datasource(datasource_id: str, payload: CredentialRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(ToolResult.success({"connected": services.datasource_service.test_connection(datasource_id, payload.credential)}))


@router.post("/{datasource_id}/schema")
def get_schema(datasource_id: str, payload: CredentialRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(ToolResult.success(services.datasource_service.get_schema(datasource_id, payload.credential)))


@router.post("/{datasource_id}/tables/{table_name}/sample")
def sample_rows(
    datasource_id: str,
    table_name: str,
    payload: SampleRowsRequest,
    services: BackendServices = Depends(get_backend_services),
) -> dict:
    return dump_result(ToolResult.success(services.datasource_service.sample_rows(datasource_id, table_name, payload.limit, payload.credential)))
