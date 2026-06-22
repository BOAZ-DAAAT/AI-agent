from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/workspace", tags=["workspace"])


class WorkspaceListRequest(BackendModel):
    path: str = "/"
    context: ContextPayload = None


class WorkspaceReadTextRequest(BackendModel):
    path: str
    context: ContextPayload = None


class WorkspaceWriteTextRequest(BackendModel):
    path: str
    content: str
    context: ContextPayload = None


class WorkspacePreviewRequest(BackendModel):
    path_or_artifact_id: str
    context: ContextPayload = None


@router.post("/list")
def list_workspace(payload: WorkspaceListRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.workspace_backend.list(payload.path, context_from(payload.context, "workspace_list"))))


@router.post("/read-text")
def read_workspace_text(payload: WorkspaceReadTextRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.workspace_backend.read_text(payload.path, context_from(payload.context, "workspace_read_text"))))


@router.post("/write-text")
def write_workspace_text(payload: WorkspaceWriteTextRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(lambda: services.workspace_backend.write_text(payload.path, payload.content, context_from(payload.context, "workspace_write_text")))
    )


@router.post("/preview")
def preview_workspace(payload: WorkspacePreviewRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(lambda: services.workspace_backend.preview(payload.path_or_artifact_id, context_from(payload.context, "workspace_preview")))
    )

