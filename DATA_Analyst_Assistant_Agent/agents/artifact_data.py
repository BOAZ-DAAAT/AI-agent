from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from data_agent_backend.models.artifacts import ArtifactRecord, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


@dataclass
class CsvArtifactData:
    artifact_id: str
    text: str
    dataframe: pd.DataFrame
    error: str | None = None


def sql_result_artifact_ids(state: OrchestrationState, runtime: AgentRuntime) -> list[str]:
    result_ids: list[str] = []
    for artifact_id in state.artifact_ids.get("sql_agent", []):
        try:
            artifact = runtime.adapter.get_artifact(artifact_id)
        except Exception:
            continue
        if artifact.type == ArtifactType.sql_result:
            result_ids.append(artifact_id)
    return result_ids


def sql_plan_artifacts(state: OrchestrationState, runtime: AgentRuntime) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    for artifact_id in state.artifact_ids.get("sql_agent", []):
        try:
            artifact = runtime.adapter.get_artifact(artifact_id)
        except Exception:
            continue
        if artifact.metadata.get("kind") == "sql_plan":
            artifacts.append(artifact)
    return artifacts


def read_sql_result_csvs(state: OrchestrationState, runtime: AgentRuntime) -> list[CsvArtifactData]:
    csvs: list[CsvArtifactData] = []
    for artifact_id in sql_result_artifact_ids(state, runtime):
        try:
            text = runtime.adapter.read_artifact_text(artifact_id)
            df = pd.read_csv(io.StringIO(text))
            csvs.append(CsvArtifactData(artifact_id=artifact_id, text=text, dataframe=df))
        except EmptyDataError:
            csvs.append(CsvArtifactData(artifact_id=artifact_id, text="", dataframe=pd.DataFrame(), error="empty_csv"))
        except Exception as exc:
            csvs.append(CsvArtifactData(artifact_id=artifact_id, text="", dataframe=pd.DataFrame(), error=str(exc)))
    return csvs


def first_dataframe(csvs: list[CsvArtifactData]) -> pd.DataFrame:
    for csv in csvs:
        if csv.error is None:
            return csv.dataframe
    return pd.DataFrame()


def read_json_artifact(runtime: AgentRuntime, artifact_id: str) -> dict[str, Any]:
    text = runtime.adapter.read_artifact_text(artifact_id)
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def generated_sql_from_artifacts(state: OrchestrationState, runtime: AgentRuntime) -> str:
    if getattr(state, "generated_sql", ""):
        return state.generated_sql
    for artifact in sql_plan_artifacts(state, runtime):
        try:
            payload = json.loads(runtime.adapter.read_artifact_text(artifact.artifact_id))
        except Exception:
            continue
        generated_sql = payload.get("generated_sql")
        if isinstance(generated_sql, str) and generated_sql.strip():
            return generated_sql
    if state.plan and state.plan.source_sql:
        return state.plan.source_sql
    return ""
