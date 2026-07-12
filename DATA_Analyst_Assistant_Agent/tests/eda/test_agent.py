from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.agent import _build_retry_hint


def test_retryable_validation_result_sets_retry_hint() -> None:
    hint = _build_retry_hint({
        "retryable": True,
        "failure_code": "insight_fallback",
        "reason": "검증 미통과(재시도 소진): insight가 생성되지 않음(빈 값/실패)",
    })

    assert hint.retryable is True
    assert hint.reason_code == "eda_self_validation_failed_retryable"
    assert hint.suggested_action == "rerun_eda_agent"
    assert hint.details["failure_code"] == "insight_fallback"


def test_non_retryable_validation_result_keeps_default_hint() -> None:
    hint = _build_retry_hint({
        "retryable": False,
        "failure_code": "no_completed_analyses",
        "reason": "검증 미통과(재시도 소진): 분석이 하나도 실행되지 않음",
    })

    assert hint.retryable is False


def test_missing_validation_result_keeps_default_hint() -> None:
    hint = _build_retry_hint({})

    assert hint.retryable is False
