"""Critic node: judge whether generated code applied a sound statistical method.

Scope is deliberately narrow (the "A" decision): it checks *method validity*,
not fit to user intent — intent adequacy is the supervisor's job. Because this
project has no benchmark to compare against, the critic is the primary safety
net for the codegen path: it is what catches a plausible-but-wrong analysis
(wrong test, wrong grain, leakage, unstated assumptions) before it ships.

On ``fail`` the returned ``feedback`` is fed back into the generate node.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisIntent,
    CodeCritique,
    GeneratedAnalysisCode,
)
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


CRITIC_SYSTEM_PROMPT = """You are a skeptical senior statistician reviewing one
generated analysis script and its computed result. Judge ONLY methodological
validity, not whether it matches the user's request (a separate agent owns intent).

Return the CodeCritique schema.

Fail (verdict="fail") if any of these hold:
- The statistical method is wrong for the data (e.g. a t-test on paired/temporal
  data, correlation reported as causation, classification metrics on a regression).
- Time handling is wrong: no resample to the stated grain, or a trend fit on too
  few periods presented as reliable.
- Data leakage, train/test contamination, or evaluation on training data.
- The result asserts findings not supported by the computed statistics.
- A required assumption or caveat is missing for the method used.
- The code silently drops most of the data or divides by zero-prone quantities
  without guarding.

Otherwise verdict="pass". Be strict but concrete: list each problem in
method_issues, and put actionable fix instructions in feedback (used to
regenerate). Default to "fail" when genuinely unsure the method is sound.
"""


def critique_analysis_code(
    intent: AnalysisIntent,
    code: GeneratedAnalysisCode,
    result: dict[str, Any],
    *,
    model: Any | None = None,
) -> CodeCritique:
    """Adversarially review the code + its result for method validity."""

    chat_model = model or get_chat_model(model_env="CODE_GENERATOR_MODEL", temperature=0)
    structured_model = chat_model.with_structured_output(CodeCritique)
    human = (
        f"Objective: {intent.objective}\n"
        f"Domain: {intent.domain}; time_grain: {intent.time_grain}; "
        f"is_time_based: {intent.is_time_based}\n\n"
        f"Generated code:\n{code.imports}\n{code.code}\n\n"
        f"Computed result:\n"
        f"- summary: {result.get('summary')}\n"
        f"- findings: {result.get('findings')}\n"
        f"- statistics: {result.get('statistics')}\n"
        f"- limitations: {result.get('limitations')}\n"
    )
    verdict = structured_model.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ])
    return verdict if isinstance(verdict, CodeCritique) else CodeCritique.model_validate(verdict)
