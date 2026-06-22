from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, dump_result
from data_agent_backend.mcp.tools_workspace import workspace_list, workspace_preview, workspace_read_text, workspace_write_text
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
    return dump_result(workspace_list(path=payload.path, context=payload.context, services=services))


@router.post("/read-text")
def read_workspace_text(payload: WorkspaceReadTextRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(workspace_read_text(path=payload.path, context=payload.context, services=services))


@router.post("/write-text")
def write_workspace_text(payload: WorkspaceWriteTextRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(workspace_write_text(path=payload.path, content=payload.content, context=payload.context, services=services))


@router.post("/preview")
def preview_workspace(payload: WorkspacePreviewRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(workspace_preview(path_or_artifact_id=payload.path_or_artifact_id, context=payload.context, services=services))

