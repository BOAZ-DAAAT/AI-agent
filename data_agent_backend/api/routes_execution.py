from __future__ import annotations

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import ContextPayload, context_from, dump_result, result_wrap
from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType
from data_agent_backend.models.common import BackendModel
from data_agent_backend.models.datasource import DatasourceCredential
from data_agent_backend.models.execution import ExecutionLimits
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/execution", tags=["execution"])


class SqlRunRequest(BackendModel):
    query: str
    run_id: str
    datasource_id: str
    credential: DatasourceCredential
    row_limit: int | None = None
    context: ContextPayload = None


class PythonRunRequest(BackendModel):
    code: str
    run_id: str
    input_artifact_ids: list[str] | None = None
    context: ContextPayload = None


@router.post("/sql")
def run_sql(payload: SqlRunRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        result_wrap(
            lambda: services.sql_executor.run_sql_query(
                query=payload.query,
                run_id=payload.run_id,
                datasource_id=payload.datasource_id,
                credential=payload.credential,
                context=context_from(payload.context, "sql_run_query"),
                row_limit=payload.row_limit,
            )
        )
    )


@router.post("/python")
def run_python(payload: PythonRunRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    context = context_from(payload.context, "sandbox_run_python")
    context = context.model_copy(update={"run_id": context.run_id or payload.run_id})
    inputs = [ArtifactRef(artifact_id=item, type=ArtifactType.dataset) for item in (payload.input_artifact_ids or [])]
    return dump_result(
        result_wrap(lambda: services.sandbox_executor.run_python(payload.code, inputs, ExecutionLimits(), context))
    )

