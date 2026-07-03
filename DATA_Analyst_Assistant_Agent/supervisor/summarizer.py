from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, StepSummary


MAX_STEP_SUMMARY_LENGTH = 1000


def summarize_agent_step(step: str, result: AgentCompactResult, *, next_action: str) -> StepSummary:
    return StepSummary(
        step=step,
        agent=result.agent,
        action=f"call_{result.agent}",
        summary=result.summary[:MAX_STEP_SUMMARY_LENGTH],
        artifact_ids=list(result.artifact_ids),
        next_action=next_action,
    )
