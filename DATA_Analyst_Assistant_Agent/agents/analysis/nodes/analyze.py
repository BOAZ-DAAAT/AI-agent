"""Analysis orchestration: classify's intent -> generate -> execute -> critic.

This is the analysis itself, not a side "codegen" step. It runs the
evaluator-optimizer loop from the LangGraph playbook. Two failure modes both
feed feedback back into the next generate attempt:

- execution failure (import/runtime error)  -> reflect on the traceback
- method failure (critic verdict == "fail")  -> reflect on the critique feedback

The loop is bounded. With no benchmark available, the critic is the safety net,
so a run that never passes the critic is reported as ``failed`` rather than
silently shipping an unreviewed result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.critic import critique_analysis_code
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.generate import (
    AnalysisCodeError,
    execute_generated_code,
    generate_analysis_code,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    CodeCritique,
    GeneratedAnalysisCode,
)

DEFAULT_MAX_ATTEMPTS = 3


@dataclass
class AnalysisOutcome:
    status: str  # "passed" | "failed"
    attempts: int
    code: GeneratedAnalysisCode | None = None
    result: dict[str, Any] | None = None
    critique: CodeCritique | None = None
    error_history: list[dict[str, str]] = field(default_factory=list)


def run_analysis(
    intent: AnalysisIntent,
    context: AnalysisContext,
    dataframe: pd.DataFrame,
    *,
    code_generator_model: Any | None = None,
    critic_model: Any | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> AnalysisOutcome:
    """Generate/execute/critique in a bounded reflect loop."""

    feedback: str | None = None
    history: list[dict[str, str]] = []
    last_code: GeneratedAnalysisCode | None = None
    last_result: dict[str, Any] | None = None
    last_critique: CodeCritique | None = None

    for attempt in range(1, max_attempts + 1):
        code = generate_analysis_code(
            intent, context, model=code_generator_model, feedback=feedback
        )
        last_code = code

        try:
            result = execute_generated_code(code, dataframe)
        except AnalysisCodeError as exc:
            feedback = f"The previous code failed to run: {exc}. Fix it and regenerate."
            history.append({"stage": "execute", "code": code.code, "error": str(exc)})
            continue

        critique = critique_analysis_code(
            intent, code, result, model=critic_model
        )
        last_result = result
        last_critique = critique

        if critique.verdict == "pass":
            return AnalysisOutcome(
                status="passed",
                attempts=attempt,
                code=code,
                result=result,
                critique=critique,
                error_history=history,
            )

        feedback = critique.feedback or "; ".join(critique.method_issues)
        history.append({"stage": "critic", "code": code.code, "error": feedback})

    return AnalysisOutcome(
        status="failed",
        attempts=max_attempts,
        code=last_code,
        result=last_result,
        critique=last_critique,
        error_history=history,
    )
