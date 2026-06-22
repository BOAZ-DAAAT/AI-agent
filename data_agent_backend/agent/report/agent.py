from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from data_agent_backend.agent.report.prompts import REPORT_AGENT_SYSTEM_PROMPT
from data_agent_backend.agent.report.tools import get_report_tools
from data_agent_backend.agent.schemas import ReportAgentOutput
from data_agent_backend.models.common import BackendError

if TYPE_CHECKING:
    from data_agent_backend.services.factory import BackendServices


def build_report_agent(model: Any, services: "BackendServices", tools: list[Any] | None = None):
    if model is None:
        raise BackendError("VALIDATION_ERROR", "A LangChain chat model is required to build the Report Agent.")

    return create_agent(
        model=model,
        tools=tools if tools is not None else get_report_tools(services),
        system_prompt=REPORT_AGENT_SYSTEM_PROMPT,
        response_format=ReportAgentOutput,
    )
