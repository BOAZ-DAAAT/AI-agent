from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, EmptyRequest, dump_result
from data_agent_backend.mcp.tools_approvals import approval_get, approval_list_pending, approval_resolve
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalGetRequest(BackendModel):
    approval_id: str
    context: ContextPayload = None


class ApprovalResolveRequest(BackendModel):
    approval_id: str
    decision: str
    edited_payload: dict[str, Any] | None = None
    context: ContextPayload = None


@router.post("/pending")
def pending_approvals(payload: EmptyRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(approval_list_pending(context=payload.context, services=services))


@router.post("/get")
def get_approval(payload: ApprovalGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(approval_get(approval_id=payload.approval_id, context=payload.context, services=services))


@router.post("/resolve")
def resolve_approval(payload: ApprovalResolveRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        approval_resolve(
            approval_id=payload.approval_id,
            decision=payload.decision,
            edited_payload=payload.edited_payload,
            context=payload.context,
            services=services,
        )
    )

