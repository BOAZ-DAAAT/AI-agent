"""Shared support logic for analysis tools."""

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.catalog import CAPABILITIES, AnalysisCapability, tool_catalog_text
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.insight import build_hypotheses, evidence_from_payload
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.modeling import build_preprocessor, top_coefficients
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.statistics import add_benjamini_hochberg_adjustment

__all__ = [
    "AnalysisCapability",
    "CAPABILITIES",
    "add_benjamini_hochberg_adjustment",
    "build_preprocessor",
    "build_hypotheses",
    "evidence_from_payload",
    "frame_from_records",
    "top_coefficients",
    "tool_catalog_text",
]
