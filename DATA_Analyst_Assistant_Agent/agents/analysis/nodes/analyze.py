"""Analysis orchestration: classify's intent -> generate -> execute -> critic.

This is the analysis itself, not a side "codegen" step. It runs the
evaluator-optimizer loop from the LangGraph playbook. Three failure modes all
feed feedback back into the next generate attempt:

- generation failure (unparseable structured output) -> reflect on the parse error
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
    AnalysisGenerationError,
    CodeExecutionPlan,
    execute_generated_code,
    generate_analysis_code,
    inspect_generated_code,
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
    progress_callback: Callable[..., None] | None = None,
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
        except AnalysisGenerationError as exc:
            _notify_progress(progress_callback, "generate", "failed", attempt)
            error = str(exc)
            feedback = (
                f"The previous response was not valid JSON: {error}. Return ONLY the "
                "JSON object with no markdown fences or extra text, and keep the code "
                "focused on what the objective needs rather than every possible statistic."
            )
            history.append({"stage": "generate", "code": "", "error": error})
            signature = _failure_signature("generate", error)
            if signature == previous_failure_signature:
                return AnalysisOutcome(
                    status="failed",
                    attempts=attempt,
                    code=last_code,
                    result=last_result,
                    critique=last_critique,
                    error_history=history,
                    early_stop_reason="same generation failure repeated after regeneration",
                )
            previous_failure_signature = signature
            continue
        except Exception:
            _notify_progress(progress_callback, "generate", "failed", attempt)
            raise
        _notify_progress(
            progress_callback,
            "generate",
            "completed",
            attempt,
            {
                "generated_code": code.code,
                "generated_imports": code.imports,
                "generated_rationale": code.rationale,
            },
        )
        last_code = code

        _notify_progress(progress_callback, "execute.preflight", "started", attempt)
        execution_plan = inspect_generated_code(code, dataframe)
        _notify_progress(
            progress_callback,
            "execute.preflight",
            "completed",
            attempt,
            execution_plan.model_dump(),
        )
        if execution_plan.decision == "blocked_unsafe":
            _notify_progress(progress_callback, "execute", "failed", attempt, execution_plan.model_dump())
            error = "; ".join(execution_plan.reasons) or "generated code failed preflight"
            feedback = f"The previous code failed preflight: {error}. Generate safe code without unsafe access."
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
                    early_stop_reason="same execution preflight failure repeated after regeneration",
                )
            previous_failure_signature = signature
            continue
        if execution_plan.decision == "manual_run_recommended":
            _notify_progress(progress_callback, "execute", "skipped_manual_run", attempt, execution_plan.model_dump())
            result = _manual_run_result(code, execution_plan)
            return AnalysisOutcome(
                status="passed",
                attempts=attempt,
                code=code,
                result=result,
                critique=CodeCritique(
                    verdict="pass",
                    method_issues=[],
                    feedback="Generated code was returned for manual execution because preflight marked it as long-running.",
                ),
                error_history=history,
                early_stop_reason="manual_run_recommended",
            )

        _notify_progress(progress_callback, "execute", "started", attempt)
        try:
            result = execute_generated_code(code, dataframe, isolated=True)
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
        _notify_progress(progress_callback, "contract_check", "started", attempt)
        contract_errors = fatal_result_contract_errors(result)
        if contract_errors:
            _append_method_note(
                result,
                "Analysis result payload had schema issues and was normalized with limitations: "
                + " ".join(contract_errors),
            )
        _notify_progress(
            progress_callback,
            "contract_check",
            "completed",
            attempt,
            {"fatal_error_count": len(contract_errors)},
        )
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
        if _is_contract_coverage_failure(critique):
            history.append({"stage": "critic", "code": code.code, "error": feedback})
            if attempt < max_attempts:
                previous_failure_signature = _failure_signature("critic", feedback)
                continue
            return AnalysisOutcome(
                status="failed",
                attempts=attempt,
                code=code,
                result=result,
                critique=critique,
                error_history=history,
                early_stop_reason="analysis_data_contract_coverage_failed",
            )
        _append_method_note(
            result,
            "Analysis critic warning: " + (feedback or "method review raised a caution"),
        )
        return AnalysisOutcome(
            status="passed",
            attempts=attempt,
            code=code,
            result=result,
            critique=CodeCritique(
                verdict="pass",
                method_issues=critique.method_issues,
                feedback=feedback,
            ),
            error_history=history,
        )

    return AnalysisOutcome(
        status="failed",
        attempts=max_attempts,
        code=last_code,
        result=last_result,
        critique=last_critique,
        error_history=history,
    )


def _is_contract_coverage_failure(critique: CodeCritique) -> bool:
    return any(
        "contract_metric_support:" in str(issue)
        for issue in critique.method_issues
    )


def _notify_progress(
    callback: Callable[..., None] | None,
    stage: str,
    status: str,
    attempt: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    try:
        if metadata is None:
            callback(stage, status, attempt)
        else:
            callback(stage, status, attempt, metadata)
    except TypeError:
        try:
            callback(stage, status, attempt)
        except Exception:
            return
    except Exception:
        # Progress telemetry must never interfere with analysis execution.
        return


def _manual_run_result(
    code: GeneratedAnalysisCode,
    execution_plan: CodeExecutionPlan,
) -> dict[str, Any]:
    plan_payload = execution_plan.model_dump()
    source = f"{code.imports}\n{code.code}" if code.imports else code.code
    reasons = "; ".join(execution_plan.reasons)
    return {
        "summary": "이 분석은 장시간 실행 가능성이 있어 자동 실행하지 않았습니다.",
        "findings": [
            "생성된 분석 코드에 장시간 실행될 수 있는 모델 학습/반복 계산 패턴이 감지되었습니다.",
            "아래 generated_code를 별도 Python 환경에서 실행해 결과를 확인하세요.",
        ],
        "statistics": {
            "execution_decision": execution_plan.decision,
            "risk_level": execution_plan.risk_level,
            "risk_reasons": list(execution_plan.reasons),
            "row_count": execution_plan.row_count,
            "column_count": execution_plan.column_count,
        },
        "limitations": [
            "자동 실행 결과가 아니므로 통계값과 결론은 아직 계산되지 않았습니다.",
            f"자동 실행을 건너뛴 이유: {reasons}" if reasons else "자동 실행을 건너뛴 이유가 기록되지 않았습니다.",
        ],
        "method_notes": [
            "장시간 실행 가능성이 있어 자동 실행하지 않았습니다. generated_code를 별도 Python 환경에서 실행하세요.",
        ],
        "method_decision": {
            "selected_method": "manual_generated_code_execution",
            "rationale": "Preflight detected potentially long-running generated analysis code.",
            "assumptions_checked": [f"{key}={value}" for key, value in plan_payload.items() if key in {"decision", "risk_level", "row_count", "column_count"}],
            "fallbacks_considered": ["Automatic sandbox execution was skipped to avoid blocking the agent process."],
        },
        "generated_code": source,
        "execution_plan": plan_payload,
    }


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
