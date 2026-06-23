from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import load_all_metadata


def load_context(state: AgentState):
    metadata = load_all_metadata()
    schema_text = (state.get("required_db_schema") or state.get("schema_text") or "").strip()
    integrity_text = (state.get("integrity_text") or "").strip()
    return {
        "schema_text": schema_text or metadata["schema_text"],
        "integrity_text": integrity_text or metadata["integrity_text"],
    }
