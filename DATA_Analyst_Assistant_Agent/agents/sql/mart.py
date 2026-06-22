"""Mart candidate detection and proposal for the SQL-Agent.

SQL-Agent proposes mart candidates. Supervisor owns the approval gate.
"""

from __future__ import annotations

import json
import re

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.sql.planner import SQLPlan
from DATA_Analyst_Assistant_Agent.mart.metadata import MartCandidate
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState

_MART_KEYWORDS = re.compile(
    r"(반복|재사용|데이터마트|마트|저장|repeat|reusable|datamart|mart|save)",
    re.IGNORECASE,
)


def needs_mart_candidate(user_query: str) -> bool:
    """Does the user query indicate mart/reusable storage intent?"""
    return _MART_KEYWORDS.search(user_query) is not None


def build_mart_candidate(
    sql_plan: SQLPlan,
    row_count: int | None = None,
    columns: list[str] | None = None,
) -> MartCandidate:
    """Build a MartCandidate from SQL plan and preview results."""
    dimension_cols = []
    measure_cols = []
    for col in (columns or sql_plan.selected_columns):
        if col == sql_plan.dimension or col in ("month", "day", "date"):
            dimension_cols.append(col)
        else:
            measure_cols.append(col)

    schema_info = {col: "unknown" for col in (columns or sql_plan.selected_columns)}

    return MartCandidate(
        source_sql=sql_plan.generated_sql,
        grain=sql_plan.dimension or "row",
        key_columns=[sql_plan.dimension] if sql_plan.dimension else [],
        dimension_columns=dimension_cols,
        measure_columns=measure_cols,
        row_count=row_count,
        schema_info=schema_info,
        storage_recommendation="manual",
    )


def register_mart_candidate_artifact(
    state: OrchestrationState,
    runtime: AgentRuntime,
    candidate: MartCandidate,
    parent_artifact_id: str,
) -> str:
    """Register mart candidate metadata as a file artifact. Returns artifact_id."""
    from data_agent_backend.models.artifacts import ArtifactType

    context = runtime.context(state, node_name="sql_agent", tool_name="sql_agent.mart_candidate")
    ref = runtime.adapter.register_artifact(
        state.run_id,
        ArtifactType.file,
        content_text=json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False, indent=2),
        filename=f"mart_candidate_{state.run_id}.json",
        created_by_tool="DATA_Analyst_Assistant_Agent.sql_agent.mart_candidate",
        context=context,
        parent_ids=[parent_artifact_id],
        lineage_edge_type="proposes_mart",
        metadata={"kind": "mart_candidate", "grain": candidate.grain},
        preview={"grain": candidate.grain, "storage_recommendation": candidate.storage_recommendation},
    )
    return ref.artifact_id
