from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, dump_result
from data_agent_backend.mcp.tools_exports import export_create
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/exports", tags=["exports"])


class ExportCreateRequest(BackendModel):
    artifact_id: str
    format: str
    destination: str | None = None
    context: ContextPayload = None


@router.post("/create")
def create_export(payload: ExportCreateRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        export_create(
            artifact_id=payload.artifact_id,
            format=payload.format,
            destination=payload.destination,
            context=payload.context,
            services=services,
        )
    )

