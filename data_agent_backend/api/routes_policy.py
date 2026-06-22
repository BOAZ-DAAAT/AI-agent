from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, dump_result
from data_agent_backend.mcp.tools_policy import policy_evaluate
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/policy", tags=["policy"])


class PolicyEvaluateRequest(BackendModel):
    action: str
    resource: str
    payload: dict[str, Any] | None = None
    context: ContextPayload = None


@router.post("/evaluate")
def evaluate_policy(payload: PolicyEvaluateRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        policy_evaluate(action=payload.action, resource=payload.resource, payload=payload.payload, context=payload.context, services=services)
    )

