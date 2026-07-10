from __future__ import annotations

import io
import json
import os
import re
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


# analytics 스키마 안의 `schema.table` 형태만 허용(식별자 문자만) — SQL 인젝션·범위이탈 차단.
_ALLOWED_MART_SCHEMA = os.getenv("ALLOWED_MART_SCHEMA", "analytics")
_MART_TABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def _is_safe_mart_table(target_table: str) -> bool:
    """target_table 이 허용된 마트 스키마 안의 안전한 `schema.table` 식별자인지 검사."""
    name = (target_table or "").strip().replace("`", "")
    if not _MART_TABLE_PATTERN.match(name):
        return False
    schema = name.split(".", 1)[0]
    return schema.lower() == _ALLOWED_MART_SCHEMA.lower()


def read_mart_dataframe(target_table: str) -> pd.DataFrame | None:
    """analytics 스키마의 마트 테이블을 DataFrame 으로 직접 조회(읽기 전용).

    SQL 에이전트가 적재한 마트를 하류(EDA/분석)가 CSV 아티팩트 대신 DB 에서 바로 읽는 경로.
    안전하지 않은 이름·DB 접속 불가·조회 실패 시 None 을 반환해 호출부가 CSV 로 폴백하게 한다.
    """
    if not _is_safe_mart_table(target_table):
        return None
    from sqlalchemy import text

    from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine

    name = target_table.strip().replace("`", "")
    schema, table = name.split(".", 1)
    quoted = f"`{schema}`.`{table}`"
    try:
        engine = get_db_engine()
        if engine is None:
            return None
        with engine.connect() as conn:
            return pd.read_sql(text(f"SELECT * FROM {quoted}"), conn)
    except Exception:
        return None


def load_analysis_inputs(state: OrchestrationState, runtime: AgentRuntime) -> list[CsvArtifactData]:
    """하류(EDA/분석) 공용 데이터 진입점.

    comprehensive(마트) 경로면 plan.target_table 로 DB 에서 마트를 직접 조회하고,
    실패하거나 simple 경로면 기존 sql_result CSV 아티팩트로 폴백한다. 반환 형식은
    read_sql_result_csvs 와 동일(list[CsvArtifactData])이라 호출부의 나머지 로직은 불변.
    """
    target_table = state.plan.target_table if state.plan else None
    if target_table:
        df = read_mart_dataframe(target_table)
        if df is not None and not df.empty:
            # 계보(lineage)용 앵커: 마트를 만든 SQL 실행의 아티팩트를 부모로 유지.
            anchor_ids = [i for i in sql_result_artifact_ids(state, runtime) if i]
            if not anchor_ids:
                anchor_ids = [i for i in state.artifact_ids.get("sql_agent", []) if i]
            anchor_id = anchor_ids[0] if anchor_ids else ""
            return [CsvArtifactData(artifact_id=anchor_id, text="", dataframe=df)]
    return read_sql_result_csvs(state, runtime)


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


def sql_validation_artifacts(state: OrchestrationState, runtime: AgentRuntime) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    for artifact_id in state.artifact_ids.get("sql_agent", []):
        try:
            artifact = runtime.adapter.get_artifact(artifact_id)
        except Exception:
            continue
        if artifact.metadata.get("kind") in {"ge_table_validation_json", "sql_validation_summary"}:
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


def latest_sql_validation_summary(state: OrchestrationState, runtime: AgentRuntime) -> dict[str, Any]:
    for artifact in reversed(sql_validation_artifacts(state, runtime)):
        try:
            payload = json.loads(runtime.adapter.read_artifact_text(artifact.artifact_id))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}
