from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
from data_agent_backend.models.artifacts import ArtifactRegisterRequest
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
    return dump_result(
        result_wrap(
            lambda: services.artifact_registry.register_artifact(
                ArtifactRegisterRequest(**payload.payload),
                context_from(payload.context, "artifact_register"),
            )
        )
    )


@router.post("/get")
def get_artifact(payload: ArtifactGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.artifact_registry.get_artifact(payload.artifact_id)))


@router.post("/list")
def list_artifacts(payload: ArtifactListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.artifact_registry.list_artifacts(run_id=payload.run_id, type=payload.type)))


@router.post("/preview")
def preview_artifact(payload: ArtifactGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.artifact_registry.get_preview(payload.artifact_id)))


@router.post("/lineage")
def lineage_artifact(payload: ArtifactGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.artifact_registry.get_lineage(payload.artifact_id)))

