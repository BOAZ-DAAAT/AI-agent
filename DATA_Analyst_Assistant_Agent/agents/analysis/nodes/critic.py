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
    AnalysisContext,
    AnalysisIntent,
    CodeCritique,
    GeneratedAnalysisCode,
)
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.coverage import build_answer_coverage


CRITIC_SYSTEM_PROMPT = """You are a skeptical senior statistician reviewing one
generated analysis script and its computed result. Judge ONLY methodological
validity, not whether it matches the user's request (a separate agent owns intent).

Return the CodeCritique schema.

Return verdict="review_required" only when the computation is valid and useful
AND the result includes a concrete review_request for a human analysis decision
whose answer can change the next analysis path or definition. Examples: selecting
a segment/group threshold, approving a proxy label, choosing an operational
metric definition, selecting a cohort observation window, or choosing an
exploratory substitute when prediction/causal analysis is not supported.

Heuristic definitions, proxy labels, operational metrics, and rule-based
segments are not failures by themselves. For those cases, expect hypothesis
tests, effect sizes, and limitations. If the code supplies those and avoids
overclaiming, return pass unless the result explicitly proposes a human-choice
review_request.

Do NOT return review_required for non-actionable method/data cautions such as
small sample size, group imbalance, missing values, skewed numeric distributions,
short observation windows, weak or missing effect sizes, missing validation sets,
missing true labels, or observational/non-causal data. Those should appear in
limitations or method_notes, and the verdict should usually be pass.

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
- A heuristic/proxy segmentation is evaluated only with descriptive shares while
  strong conclusions are stated, with no hypothesis tests or effect sizes.

Otherwise verdict="pass". Be strict but concrete: list each problem in
method_issues, and put actionable fix instructions in feedback (used to
regenerate). Default to "fail" when genuinely unsure the method is sound.
"""


def critique_analysis_code(
    intent: AnalysisIntent,
    code: GeneratedAnalysisCode,
    result: dict[str, Any],
    *,
    context: AnalysisContext | None = None,
    model: Any | None = None,
) -> CodeCritique:
    """Adversarially review the code + its result for method validity."""

    chat_model = model or get_chat_model(model_env="ANALYSIS_CRITIC_MODEL", temperature=0)
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
        f"- method_notes: {result.get('method_notes')}\n"
        f"- review_request: {result.get('review_request')}\n"
    )
    # 상류 SQL 원천 테이블의 알려진 정합성 이슈(#130) — 이를 감안 안 한 결론을 잡아내는 검토 근거.
    if context is not None and context.known_data_quality_issues:
        joined = "\n".join(f"- {issue}" for issue in context.known_data_quality_issues)
        human += (
            "\nKnown data quality issues (from SQL integrity check):\n"
            f"{joined}\n"
            "Fail if the analysis draws conclusions these issues would undermine "
            "without acknowledging them.\n"
        )
    verdict = structured_model.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ])
    return verdict if isinstance(verdict, CodeCritique) else CodeCritique.model_validate(verdict)


def deterministic_precheck(
    intent: AnalysisIntent,
    context: AnalysisContext,
    code: GeneratedAnalysisCode,
    result: dict[str, Any],
) -> CodeCritique | None:
    """Cheap structural checks before spending an LLM critic call.

    The LLM critic remains method-only. These checks catch deterministic
    codegen failures such as empty statistics or generated code that never
    touches the metric/dimension/time signals the classifier identified.
    """

    issues: list[str] = []
    if not str(result.get("summary") or "").strip():
        issues.append("result.summary is empty")
    if not result.get("findings"):
        issues.append("result.findings is empty")
    statistics = result.get("statistics")
    if not isinstance(statistics, dict) or not statistics:
        issues.append("result.statistics must contain computed values")

    coverage = build_answer_coverage(intent, context, code, result)
    if coverage.coverage_status == "missing":
        issues.append(
            "generated code did not use requested analysis signals: "
            + ", ".join(coverage.missing_requirements)
        )
    elif coverage.coverage_status == "partial" and intent.is_time_based:
        issues.append(
            "time-based analysis only partially covered requested signals: "
            + ", ".join(coverage.missing_requirements)
        )

    if not issues:
        return None
    feedback = (
        "Fix deterministic pre-check failures before method review: "
        + "; ".join(issues)
    )
    return CodeCritique(verdict="fail", method_issues=issues, feedback=feedback)
