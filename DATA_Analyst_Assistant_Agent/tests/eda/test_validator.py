from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.agent import run_eda_self_check
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.validator import (
    MAX_VALIDATION_RETRIES,
    validator_node,
)


def _capped_fail_state() -> dict:
    return {
        "user_question": "질문",
        "validation_retries": MAX_VALIDATION_RETRIES,
        "insight_result": "",  # _FALLBACK_TEXTS 매치 -> 결정론적 실패
        "hypotheses": "가설 텍스트",
        "controller_log": [{"choice": "planner"}],
        "statistical_metadata": {"some": "stat"},
        "cautions": [],
    }


def test_capped_deterministic_failure_surfaces_as_caution() -> None:
    update = validator_node(_capped_fail_state())

    assert update["validation_result"]["status"] == "pass"
    codes = [c["code"] for c in update["cautions"]]
    assert "EDA_SELF_VALIDATION_FAILED" in codes


def test_insight_fallback_is_classified_retryable() -> None:
    update = validator_node(_capped_fail_state())

    verdict = update["validation_result"]
    assert verdict["failure_code"] == "insight_fallback"
    assert verdict["retryable"] is True
    caution = next(c for c in update["cautions"] if c["code"] == "EDA_SELF_VALIDATION_FAILED")
    assert caution["details"] == {"failure_code": "insight_fallback", "retryable": True}


def test_hypothesis_fallback_is_classified_retryable() -> None:
    state = _capped_fail_state()
    state["insight_result"] = "정상 인사이트"
    state["hypotheses"] = "가설 생성 실패"

    verdict = validator_node(state)["validation_result"]

    assert verdict["failure_code"] == "hypothesis_fallback"
    assert verdict["retryable"] is True


def test_no_completed_analyses_is_classified_non_retryable() -> None:
    state = _capped_fail_state()
    state["insight_result"] = "정상 인사이트"
    state["hypotheses"] = "정상 가설"
    state["controller_log"] = []

    verdict = validator_node(state)["validation_result"]

    assert verdict["failure_code"] == "no_completed_analyses"
    assert verdict["retryable"] is False


def test_missing_statistical_metadata_is_classified_non_retryable() -> None:
    state = _capped_fail_state()
    state["insight_result"] = "정상 인사이트"
    state["hypotheses"] = "정상 가설"
    state["statistical_metadata"] = {}

    verdict = validator_node(state)["validation_result"]

    assert verdict["failure_code"] == "missing_statistical_metadata"
    assert verdict["retryable"] is False


def test_capped_failure_caution_preserves_existing_cautions() -> None:
    state = _capped_fail_state()
    state["cautions"] = [{"code": "OTHER", "source": "x"}]

    update = validator_node(state)

    codes = [c["code"] for c in update["cautions"]]
    assert codes == ["OTHER", "EDA_SELF_VALIDATION_FAILED"]


def test_non_capped_deterministic_failure_still_retries() -> None:
    state = _capped_fail_state()
    state["validation_retries"] = 0

    update = validator_node(state)

    assert update["validation_result"]["status"] == "retry"
    assert "cautions" not in update


def test_run_eda_self_check_flags_validator_failure() -> None:
    cautions = [{
        "code": "EDA_SELF_VALIDATION_FAILED",
        "source": "eda_validator",
        "message_ko": "실패 사유",
    }]

    checks = run_eda_self_check(["artifact-1"], {"columns": ["a"]}, cautions)
    validation_check = next(c for c in checks if c.name == "eda_self_validation")

    assert validation_check.passed is False
    assert validation_check.severity == "error"
    assert validation_check.detail == "실패 사유"


def test_run_eda_self_check_passes_without_validator_failure() -> None:
    checks = run_eda_self_check(["artifact-1"], {"columns": ["a"]}, [])
    validation_check = next(c for c in checks if c.name == "eda_self_validation")

    assert validation_check.passed is True
    assert validation_check.severity == "info"
