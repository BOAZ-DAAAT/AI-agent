"""load_context 노드: 스키마/정합성 메타데이터 로드."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_all_metadata


def load_context(state: AgentState):
    metadata = load_all_metadata()
    return {
        "schema_text": metadata["schema_text"],
        "integrity_text": metadata["integrity_text"],
    }
