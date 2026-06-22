from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, EmptyRequest, context_from, dump_result, result_wrap
from data_agent_backend.models.approvals import ApprovalDecision
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
    return dump_result(result_wrap(lambda: services.approval_store.list_pending_approvals(context_from(payload.context, "approval_list_pending"))))


@router.post("/get")
def get_approval(payload: ApprovalGetRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(result_wrap(lambda: services.approval_store.get_approval_request(payload.approval_id)))


@router.post("/resolve")
def resolve_approval(payload: ApprovalResolveRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(
            lambda: services.approval_store.resolve_approval_request(
                payload.approval_id,
                ApprovalDecision(payload.decision),
                payload.edited_payload,
                context_from(payload.context, "approval_resolve").user_id,
            )
        )
    )

