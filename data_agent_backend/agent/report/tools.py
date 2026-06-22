from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.tools import tool
from pydantic import Field

from data_agent_backend.models.artifacts import ArtifactRegisterRequest, ArtifactType
from data_agent_backend.models.common import BackendModel, JsonDict, to_jsonable
from data_agent_backend.models.contexts import PolicyContext

if TYPE_CHECKING:
    from data_agent_backend.services.factory import BackendServices


SENSITIVE_METADATA_KEYS = {"password", "credential", "credentials", "secret", "token", "api_key"}


class SaveReportArtifactRequest(BackendModel):
    run_id: str
    title: str
    report_markdown: str
    thread_id: str | None = None
    project_id: str | None = None
    artifact_refs: list[JsonDict] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_METADATA_KEYS:
                continue
            sanitized[str(key)] = _sanitize_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    return value


def get_run_summary_payload(services: "BackendServices", run_id: str) -> JsonDict:
    summary = services.run_service.get_summary(run_id, PolicyContext(run_id=run_id, tool_name="report_get_run_summary"))
    return to_jsonable(summary)


def get_artifact_preview_payload(services: "BackendServices", artifact_id: str) -> JsonDict:
    artifact = services.artifact_registry.get_artifact(artifact_id)
    return {"artifact": to_jsonable(artifact.ref()), "preview": services.artifact_registry.get_preview(artifact_id)}


def save_report_artifact_payload(services: "BackendServices", request: SaveReportArtifactRequest | dict[str, Any]) -> JsonDict:
    payload = request if isinstance(request, SaveReportArtifactRequest) else SaveReportArtifactRequest(**request)
    parent_ids = [str(item["artifact_id"]) for item in payload.artifact_refs if item.get("artifact_id")]
    metadata = {
        "title": payload.title,
        "source": "report_agent",
        **_sanitize_metadata(payload.metadata),
    }
    artifact = services.artifact_registry.register_artifact(
        ArtifactRegisterRequest(
            run_id=payload.run_id,
            thread_id=payload.thread_id,
            project_id=payload.project_id,
            type=ArtifactType.report,
            content_text=payload.report_markdown,
            filename="report.md",
            created_by_tool="create_report",
            parent_ids=parent_ids,
            lineage_edge_type="report_of",
            metadata=metadata,
            preview={"title": payload.title, "snippet": payload.report_markdown[:500]},
        ),
        PolicyContext(run_id=payload.run_id, thread_id=payload.thread_id, project_id=payload.project_id, tool_name="create_report"),
    )
    return to_jsonable(artifact.ref())


def get_report_tools(services: "BackendServices") -> list[Any]:
    @tool("get_run_summary")
    def get_run_summary(run_id: str) -> JsonDict:
        """Return the run summary, including run metadata, events, artifacts, and pending approvals."""
        return get_run_summary_payload(services, run_id)

    @tool("get_artifact_preview")
    def get_artifact_preview(artifact_id: str) -> JsonDict:
        """Return a safe artifact reference and its preview for report evidence."""
        return get_artifact_preview_payload(services, artifact_id)

    @tool("save_report_artifact")
    def save_report_artifact(payload: dict[str, Any]) -> JsonDict:
        """Save a completed Markdown report as a report artifact."""
        return save_report_artifact_payload(services, payload)

    return [get_run_summary, get_artifact_preview, save_report_artifact]
