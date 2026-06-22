from __future__ import annotations

import json

from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import first_dataframe, read_json_artifact, read_sql_result_csvs
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.visualization.chart_selector import build_chart_config
from DATA_Analyst_Assistant_Agent.agents.visualization.self_check import run_visualization_self_check
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, OrchestrationState, ValidationBlock


class VisualizationAgent:
    name = "visualization_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="visualization_agent.chart_spec")
        parent_ids = state.artifact_ids.get("analysis_agent", []) + state.artifact_ids.get("sql_agent", [])
        analysis_result = {}
        if state.artifact_ids.get("analysis_agent"):
            analysis_result = read_json_artifact(runtime, state.artifact_ids["analysis_agent"][0])
        csvs = read_sql_result_csvs(state, runtime)
        config = build_chart_config(state, dataframe=first_dataframe(csvs), analysis_result=analysis_result)
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.chart,
            content_text=json.dumps(config, ensure_ascii=False, indent=2),
            filename="vega_lite_spec.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.visualization",
            context=context,
            parent_ids=parent_ids,
            metadata={"kind": "vega_lite_spec", "chart_type": config["chart_type"], "source_artifact_count": len(parent_ids)},
            preview={
                "chart_type": config["chart_type"],
                "title": config["title"],
                "encoding": config["encoding"],
                "data_reference": config["data_reference"],
            },
        )
        return AgentEnvelope(
            agent_name=self.name,
            summary=f"{config['chart_type']} Vega-Lite specification generated.",
            artifact_refs=[ref],
            validation=ValidationBlock(local_checks=run_visualization_self_check(config)),
            next_handoff="validation_agent",
        )
