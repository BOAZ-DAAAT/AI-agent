from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, dump_result
from data_agent_backend.mcp.tools_artifacts import artifact_get, artifact_lineage, artifact_list, artifact_preview, artifact_register
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/artifacts", tags=["artifacts"])


class ArtifactRegisterRequestBody(BackendModel):
    payload: dict[str, Any]
    context: ContextPayload = None


class ArtifactGetRequest(BackendModel):
    artifact_id: str


class ArtifactListRequest(BackendModel):
    run_id: str | None = None
    type: str | None = None


@router.post("/register")
def register_artifact(payload: ArtifactRegisterRequestBody, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(artifact_register(payload=payload.payload, context=payload.context, services=services))


@router.post("/get")
def get_artifact(payload: ArtifactGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(artifact_get(artifact_id=payload.artifact_id, services=services))


@router.post("/list")
def list_artifacts(payload: ArtifactListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(artifact_list(run_id=payload.run_id, type=payload.type, services=services))


@router.post("/preview")
def preview_artifact(payload: ArtifactGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(artifact_preview(artifact_id=payload.artifact_id, services=services))


@router.post("/lineage")
def lineage_artifact(payload: ArtifactGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(artifact_lineage(artifact_id=payload.artifact_id, services=services))

