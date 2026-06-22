from __future__ import annotations

from .common import BackendModel


class RunContext(BackendModel):
    run_id: str
    thread_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None


class PolicyContext(BackendModel):
    run_id: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    tool_name: str | None = None
    node_name: str | None = None
    approval_id: str | None = None
    dev_mode: bool = False

    def with_tool(self, tool_name: str) -> "PolicyContext":
        return self.model_copy(update={"tool_name": tool_name})

