"""clustering node: K-means clustering with LangSmith-visible tool and LLM runs."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import (
    append_errors,
    safe_json_parse,
    split_marked_json,
)
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import tool_runner
from DATA_Analyst_Assistant_Agent.agents.eda.prompts import clustering_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.analysis import ANALYSIS_FACTS_MARKER
from DATA_Analyst_Assistant_Agent.agents.eda.state import EDAState
from DATA_Analyst_Assistant_Agent.agents.eda.tools import run_clustering


def clustering_node(state: EDAState) -> dict:
    raw_result, tool_err = tool_runner.run_node_with_retry(
        lambda: run_clustering.invoke({}),
        "clustering.tool",
        fallback='{"skip": true, "reason": "clustering tool failed; analysis skipped"}',
    )
    result = safe_json_parse(
        raw_result,
        {"skip": True, "reason": "clustering result was not valid JSON"},
    )

    summary, summary_err = tool_runner.run_node_with_retry(
        lambda: tool_runner.get_llm()
        .invoke(clustering_prompt(state["user_question"], state.get("inspect_result", ""), raw_result))
        .content.strip(),
        "clustering",
        fallback="clustering summary failed",
    )
    prose, parsed = split_marked_json(summary, ANALYSIS_FACTS_MARKER)
    facts = tool_runner._normalize_analysis_facts(parsed, prose)

    return {
        "clustering_result": result,
        "clustering_summary": prose,
        "clustering_facts": facts,
        "error_log": append_errors(state, tool_err, summary_err),
    }
