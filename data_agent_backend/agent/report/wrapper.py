from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool

from data_agent_backend.agent.schemas import ReportAgentInput, ReportAgentOutput
from data_agent_backend.models.common import BackendError, to_jsonable


def _coerce_input(payload: ReportAgentInput | dict[str, Any]) -> ReportAgentInput:
    return payload if isinstance(payload, ReportAgentInput) else ReportAgentInput(**payload)


def _extract_report_output(result: Any) -> ReportAgentOutput:
    if isinstance(result, ReportAgentOutput):
        return result
    if isinstance(result, dict):
        if "structured_response" in result:
            structured = result["structured_response"]
            return structured if isinstance(structured, ReportAgentOutput) else ReportAgentOutput(**structured)
        messages = result.get("messages")
        if messages:
            last = messages[-1]
            content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
            if isinstance(content, str):
                return ReportAgentOutput(**json.loads(content))
    structured = getattr(result, "structured_response", None)
    if structured is not None:
        return structured if isinstance(structured, ReportAgentOutput) else ReportAgentOutput(**structured)
    raise BackendError("REPORT_AGENT_OUTPUT_INVALID", "Report Agent did not return a valid structured response.")


def call_create_report(report_agent: Any, payload: ReportAgentInput | dict[str, Any]) -> ReportAgentOutput:
    request = _coerce_input(payload)
    result = report_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.model_dump_json(),
                }
            ]
        }
    )
    return _extract_report_output(result)


def make_create_report_tool(report_agent: Any):
    @tool("create_report")
    def create_report(payload: dict[str, Any]) -> dict[str, Any]:
        """Create a final analysis report from SQL, EDA, Analysis outputs, chart refs, and artifact refs."""
        return to_jsonable(call_create_report(report_agent, payload))

    return create_report
