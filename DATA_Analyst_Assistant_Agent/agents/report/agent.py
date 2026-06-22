from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.artifact_data import generated_sql_from_artifacts, read_json_artifact
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.report.builder import build_report
from DATA_Analyst_Assistant_Agent.agents.report.self_check import run_report_self_check
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, OrchestrationState, ValidationBlock


class ReportAgent:
    name = "report_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="report_agent.final_report")
        parent_ids = [artifact_id for ids in state.artifact_ids.values() for artifact_id in ids]
        eda_profile = None
        if state.artifact_ids.get("eda_agent"):
            eda_payload = read_json_artifact(runtime, state.artifact_ids["eda_agent"][0])
            eda_profile = eda_payload.get("profile", eda_payload)
        analysis_result = None
        if state.artifact_ids.get("analysis_agent"):
            analysis_result = read_json_artifact(runtime, state.artifact_ids["analysis_agent"][0])
        visualization_result = None
        if state.artifact_ids.get("visualization_agent"):
            visualization_result = read_json_artifact(runtime, state.artifact_ids["visualization_agent"][0])
        report = build_report(
            state,
            generated_sql=generated_sql_from_artifacts(state, runtime),
            eda_profile=eda_profile,
            analysis_result=analysis_result,
            visualization_result=visualization_result,
        )
        ref = runtime.adapter.save_workspace_file(
            state.run_id,
            report,
            context=context,
            filename="final_report.md",
            metadata={
                "kind": "final_report",
                "source_artifact_count": len(parent_ids),
                "source_artifact_ids": parent_ids,
            },
        )
        return AgentEnvelope(
            agent_name=self.name,
            summary="Final report generated from registered evidence artifacts.",
            artifact_refs=[ref],
            validation=ValidationBlock(local_checks=run_report_self_check(report)),
            next_handoff="validation_agent",
        )
