from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType
from data_agent_backend.models.common import BackendModel
from data_agent_backend.models.execution import ExecutionLimits
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/execution", tags=["execution"])


class PythonRunRequest(BackendModel):
    code: str
    run_id: str
    input_artifact_ids: list[str] | None = None
    context: ContextPayload = None


@router.post("/python")
def run_python(payload: PythonRunRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    context = context_from(payload.context, "sandbox_run_python")
    context = context.model_copy(update={"run_id": context.run_id or payload.run_id})
    inputs = [ArtifactRef(artifact_id=item, type=ArtifactType.dataset) for item in (payload.input_artifact_ids or [])]
    return dump_result(
        result_wrap(lambda: services.sandbox_executor.run_python(payload.code, inputs, ExecutionLimits(), context))
    )
