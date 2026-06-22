"""Mart persistence helpers.

Permanent materialization is gated by Supervisor approval.
This module only provides helper utilities; it does NOT write marts autonomously.
"""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.mart.metadata import MartCandidate


def is_approved_for_persistence(approval_status: str) -> bool:
    """Check whether an approval status allows permanent mart write."""
    return approval_status == "approved"


def build_schema_json(candidate: MartCandidate) -> dict:
    """Build the schema_json payload expected by BackendAdapter.materialize_mart_metadata()."""
    columns = []
    for col in candidate.key_columns:
        columns.append({"name": col, "role": "key", "type": candidate.schema_info.get(col, "unknown")})
    for col in candidate.dimension_columns:
        columns.append({"name": col, "role": "dimension", "type": candidate.schema_info.get(col, "unknown")})
    for col in candidate.measure_columns:
        columns.append({"name": col, "role": "measure", "type": candidate.schema_info.get(col, "unknown")})
    return {"columns": columns, "grain": candidate.grain}
