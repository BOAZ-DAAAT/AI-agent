from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
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
        result_wrap(
            lambda: services.export_service.create_export(
                payload.artifact_id,
                payload.format,
                payload.destination,
                context_from(payload.context, "export_create"),
            )
        )
    )

