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
from typing import Any, Callable

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.critic import (
    critique_analysis_code,
    deterministic_precheck,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.generate import (
    AnalysisCodeError,
    execute_generated_code,
    generate_analysis_code,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.result_contract import fatal_result_contract_errors
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    CodeCritique,
    GeneratedAnalysisCode,
)

DEFAULT_MAX_ATTEMPTS = 2


@dataclass
class AnalysisOutcome:
    status: str  # "passed" | "review_required" | "failed"
    attempts: int
    code: GeneratedAnalysisCode | None = None
    result: dict[str, Any] | None = None
    critique: CodeCritique | None = None
    error_history: list[dict[str, str]] = field(default_factory=list)
    early_stop_reason: str = ""


def run_analysis(
    intent: AnalysisIntent,
    context: AnalysisContext,
    dataframe: pd.DataFrame,
    *,
    code_generator_model: Any | None = None,
    critic_model: Any | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    progress_callback: Callable[[str, str, int], None] | None = None,
) -> AnalysisOutcome:
    """Generate/execute/critique in a bounded reflect loop."""

    feedback: str | None = None
    history: list[dict[str, str]] = []
    last_code: GeneratedAnalysisCode | None = None
    last_result: dict[str, Any] | None = None
    last_critique: CodeCritique | None = None
    previous_failure_signature: tuple[str, str] | None = None

    for attempt in range(1, max_attempts + 1):
        _notify_progress(progress_callback, "generate", "started", attempt)
        try:
            code = generate_analysis_code(
                intent,
                context,
                model=code_generator_model,
                feedback=feedback,
            )
        except Exception:
            _notify_progress(progress_callback, "generate", "failed", attempt)
            raise
        _notify_progress(progress_callback, "generate", "completed", attempt)
        last_code = code

        _notify_progress(progress_callback, "execute", "started", attempt)
        try:
            result = execute_generated_code(code, dataframe)
        except AnalysisCodeError as exc:
            _notify_progress(progress_callback, "execute", "failed", attempt)
            feedback = f"The previous code failed to run: {exc}. Fix it and regenerate."
            error = str(exc)
            history.append({"stage": "execute", "code": code.code, "error": error})
            signature = _failure_signature("execute", error)
            if signature == previous_failure_signature:
                return AnalysisOutcome(
                    status="failed",
                    attempts=attempt,
                    code=last_code,
                    result=last_result,
                    critique=last_critique,
                    error_history=history,
                    early_stop_reason="same execution failure repeated after regeneration",
                )
            previous_failure_signature = signature
            continue
        contract_errors = fatal_result_contract_errors(result)
        if contract_errors:
            _notify_progress(progress_callback, "execute", "failed", attempt)
            last_result = result
            feedback = (
                "The previous code produced a result dict that does not match the "
                "AnalysisResult contract. Fix these result payload issues and regenerate: "
                + " ".join(contract_errors)
            )
            history.append({"stage": "result_contract", "code": code.code, "error": feedback})
            signature = _failure_signature("result_contract", feedback)
            if signature == previous_failure_signature:
                return AnalysisOutcome(
                    status="failed",
                    attempts=attempt,
                    code=last_code,
                    result=last_result,
                    critique=last_critique,
                    error_history=history,
                    early_stop_reason="same result contract failure repeated after regeneration",
                )
            previous_failure_signature = signature
            continue
        _notify_progress(progress_callback, "execute", "completed", attempt)

        critique = deterministic_precheck(intent, context, code, result)
        if critique is None:
            _notify_progress(progress_callback, "critic", "started", attempt)
            try:
                critique = critique_analysis_code(
                    intent,
                    code,
                    result,
                    context=context,
                    model=critic_model,
                )
            except Exception:
                _notify_progress(progress_callback, "critic", "failed", attempt)
                raise
            _notify_progress(progress_callback, "critic", "completed", attempt)
        else:
            _notify_progress(progress_callback, "critic", "skipped_precheck", attempt)
        last_result = result
        last_critique = critique

        if critique.verdict == "pass" and _has_actionable_review_request(result):
            return AnalysisOutcome(
                status="review_required",
                attempts=attempt,
                code=code,
                result=result,
                critique=CodeCritique(
                    verdict="review_required",
                    method_issues=["analysis decision review requested"],
                    feedback=str(result["review_request"].get("question") or "Review the proposed analysis decision."),
                ),
                error_history=history,
            )

        if critique.verdict == "pass":
            return AnalysisOutcome(
                status="passed",
                attempts=attempt,
                code=code,
                result=result,
                critique=critique,
                error_history=history,
            )
        if critique.verdict == "review_required":
            if not _has_actionable_review_request(result):
                _append_method_note(
                    result,
                    critique.feedback or "; ".join(critique.method_issues)
                    or "Method review noted non-actionable interpretation cautions.",
                )
                return AnalysisOutcome(
                    status="passed",
                    attempts=attempt,
                    code=code,
                    result=result,
                    critique=CodeCritique(
                        verdict="pass",
                        method_issues=critique.method_issues,
                        feedback=critique.feedback,
                    ),
                    error_history=history,
                )
            return AnalysisOutcome(
                status="review_required",
                attempts=attempt,
                code=code,
                result=result,
                critique=critique,
                error_history=history,
            )

        feedback = critique.feedback or "; ".join(critique.method_issues)
        history.append({"stage": "critic", "code": code.code, "error": feedback})
        signature = _failure_signature("critic", feedback)
        if signature == previous_failure_signature:
            return AnalysisOutcome(
                status="failed",
                attempts=attempt,
                code=last_code,
                result=last_result,
                critique=last_critique,
                error_history=history,
                early_stop_reason="same critic failure repeated after regeneration",
            )
        previous_failure_signature = signature

    return AnalysisOutcome(
        status="failed",
        attempts=max_attempts,
        code=last_code,
        result=last_result,
        critique=last_critique,
        error_history=history,
    )


def _notify_progress(
    callback: Callable[[str, str, int], None] | None,
    stage: str,
    status: str,
    attempt: int,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, status, attempt)
    except Exception:
        # Progress telemetry must never interfere with analysis execution.
        return


def _failure_signature(stage: str, error: str) -> tuple[str, str]:
    """Compact a failure so repeated unresolved problems can stop early."""

    return stage, " ".join(str(error or "").split()).lower()


def _has_actionable_review_request(result: dict[str, Any]) -> bool:
    request = result.get("review_request")
    if not isinstance(request, dict):
        return False
    required_text = (
        request.get("question"),
        request.get("proposal"),
        request.get("recommended_option_id"),
    )
    if not all(str(value or "").strip() for value in required_text):
        return False
    options = request.get("options")
    return isinstance(options, list) and any(
        isinstance(option, dict) and str(option.get("id") or "").strip()
        for option in options
    )


def _append_method_note(result: dict[str, Any], note: str) -> None:
    normalized = str(note or "").strip()
    if not normalized:
        return
    notes = result.get("method_notes")
    if not isinstance(notes, list):
        notes = []
    if normalized not in [str(item) for item in notes]:
        notes.append(normalized)
    result["method_notes"] = notes
