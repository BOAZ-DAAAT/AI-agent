"""Mart candidate metadata model.

MartCandidate captures everything needed to evaluate whether a query result
should be persisted as a reusable data mart. The actual materialization
happens only after Supervisor approval gate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MartCandidate(BaseModel):
    """Proposed mart candidate produced by SQL-Agent."""
    source_sql: str
    grain: str = ""
    key_columns: list[str] = Field(default_factory=list)
    dimension_columns: list[str] = Field(default_factory=list)
    measure_columns: list[str] = Field(default_factory=list)
    row_count: int | None = None
    schema_info: dict[str, str] = Field(default_factory=dict)
    storage_recommendation: str = "manual"
