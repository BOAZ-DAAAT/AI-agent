from __future__ import annotations

from typing import Any

from .common import BackendModel, JsonDict, StrEnum


class ArtifactType(StrEnum):
    dataset = "dataset"
    sql_query = "sql_query"
    sql_result = "sql_result"
    python_code = "python_code"
    execution_log = "execution_log"
    data_profile = "data_profile"
    chart = "chart"
    report = "report"
    file = "file"
    export = "export"


class ArtifactRef(BackendModel):
    artifact_id: str
    type: ArtifactType
    uri: str | None = None
    content_hash: str | None = None
    preview: JsonDict = {}


class ArtifactRecord(BackendModel):
    artifact_id: str
    run_id: str
    thread_id: str | None = None
    project_id: str | None = None
    type: ArtifactType
    uri: str
    local_path: str | None = None
    content_hash: str | None = None
    created_at: str
    created_by_tool: str
    created_by_node: str | None = None
    parent_ids: list[str] = []
    metadata: JsonDict = {}
    preview: JsonDict = {}
    permissions: JsonDict | None = None
    approval_id: str | None = None
    policy_decision_id: str | None = None

    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.artifact_id,
            type=self.type,
            uri=self.uri,
            content_hash=self.content_hash,
            preview=self.preview,
        )


class ArtifactLineageEdge(BackendModel):
    parent_id: str
    child_id: str
    edge_type: str = "derived_from"
    metadata: JsonDict = {}


class ClaimEvidence(BackendModel):
    claim_id: str
    artifact_ids: list[str]
    metadata: JsonDict = {}


class ArtifactRegisterRequest(BackendModel):
    run_id: str
    type: ArtifactType
    content_text: str | None = None
    content_bytes: bytes | None = None
    filename: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    created_by_tool: str
    created_by_node: str | None = None
    parent_ids: list[str] = []
    lineage_edge_type: str = "derived_from"
    metadata: JsonDict = {}
    preview: JsonDict | None = None
    permissions: JsonDict | None = None
    approval_id: str | None = None
    policy_decision_id: str | None = None
