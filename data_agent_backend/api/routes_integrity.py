from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from data_agent_backend.api.common import dump_result
from data_agent_backend.models.common import BackendModel
from data_agent_backend.services.factory import BackendServices

from .deps import get_backend_services


router = APIRouter(prefix="/integrity", tags=["integrity"])


class IntegrityNotifyRequest(BackendModel):
    dataset_name: str
    tables: list[str] | dict[str, Any] | None = None
    source_version: str | None = None
    metadata: dict[str, Any] | None = None


class IntegrityReadyRequest(BackendModel):
    dataset_name: str
    tables: list[str]
    wait_timeout_s: float = 0.0
    metadata: dict[str, Any] | None = None


class IntegritySummaryRequest(BackendModel):
    dataset_name: str
    tables: list[str] | None = None
    include_pass: bool = False


@router.post("/notify-dataset-update")
def notify_dataset_update(payload: IntegrityNotifyRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        services.integrity_service.notify_dataset_update(
            payload.dataset_name,
            payload.tables,
            payload.source_version,
            payload.metadata,
        )
    )


@router.post("/ensure-tables-ready")
def ensure_tables_ready(payload: IntegrityReadyRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(
        services.integrity_service.ensure_tables_ready(
            payload.dataset_name,
            payload.tables,
            payload.wait_timeout_s,
            payload.metadata,
        )
    )


@router.post("/summary")
def get_integrity_summary(payload: IntegritySummaryRequest, services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(services.integrity_service.get_summary(payload.dataset_name, payload.tables, payload.include_pass))


@router.post("/process-next")
def process_next_integrity_job(services: BackendServices = Depends(get_backend_services)) -> dict:
    return dump_result(services.integrity_service.process_next_pending())
