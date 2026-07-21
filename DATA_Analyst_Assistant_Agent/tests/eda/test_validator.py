from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.agent import (
    run_eda_self_check,
    run_eda_validation_findings,
)
from DATA_Analyst_Assistant_Agent.agents.eda.nodes import validator as validator_mod
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.validator import (
    MAX_VALIDATION_RETRIES,
    validator_node,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def invoke(self, prompt: str):
        return _FakeResponse(self.reply)


def _llm_audit_state() -> dict:
    """결정론 체크를 통과하는(=LLM 감사로 진입하는) 정상형 state."""
    return {
        "user_question": "카테고리별 가격대별 리뷰점수 차이 검정",
        "validation_retries": MAX_VALIDATION_RETRIES,
        "insight_result": "가격대가 높을수록 평점이 3.575로 가장 높다.",
        "hypotheses": "가설 텍스트",
        "final_summary": "요약",
        "controller_log": [{"choice": "comparison"}],
        "statistical_metadata": {"group_comparison": {"price": {}}},
        "cautions": [],
    }


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


def test_non_capped_retryable_failure_still_retries() -> None:
    state = _capped_fail_state()
    state["validation_retries"] = 0  # insight_fallback → 재시도 가치 있으므로 재시도

    update = validator_node(state)

    assert update["validation_result"]["status"] == "retry"
    assert "cautions" not in update


def test_structural_failure_early_stops_without_retry() -> None:
    # 캡이 안 걸렸어도(retries=0) 구조적 실패는 헛재시도 없이 즉시 멈춘다.
    state = _capped_fail_state()
    state["validation_retries"] = 0
    state["insight_result"] = "정상 인사이트"
    state["hypotheses"] = "정상 가설"
    state["controller_log"] = []  # no_completed_analyses (non-retryable)

    update = validator_node(state)

    assert update["validation_result"]["status"] == "pass"
    assert update["validation_result"]["failure_code"] == "no_completed_analyses"
    # 재시도를 아예 안 했으니 카운터가 올라가지 않아야 한다.
    assert "validation_retries" not in update
    codes = [c["code"] for c in update["cautions"]]
    assert "EDA_SELF_VALIDATION_FAILED" in codes


def test_missing_statistical_metadata_early_stops_without_retry() -> None:
    state = _capped_fail_state()
    state["validation_retries"] = 0
    state["insight_result"] = "정상 인사이트"
    state["hypotheses"] = "정상 가설"
    state["statistical_metadata"] = {}  # missing_statistical_metadata (non-retryable)

    update = validator_node(state)

    assert update["validation_result"]["status"] == "pass"
    assert update["validation_result"]["failure_code"] == "missing_statistical_metadata"
    assert "validation_retries" not in update


def test_llm_audit_capped_retry_surfaces_as_caution(monkeypatch) -> None:
    """#194 — LLM 감사(환각 등)가 캡 소진으로 강제통과돼도, 결정론적 실패와 동일하게
    caution이 남아야 분석에이전트가 '검증 미완료'를 알 수 있다(이전엔 여기만 조용히 사라졌음)."""
    reply = (
        '{"status":"retry","retry_target":"insight",'
        '"reason":"검증된 수치에 없는 3.575를 인용함 -> 환각 가능",'
        '"feedback":"수치를 검증된 것만 인용하라"}'
    )
    monkeypatch.setattr(validator_mod, "get_llm", lambda: _FakeLLM(reply))

    update = validator_node(_llm_audit_state())

    verdict = update["validation_result"]
    assert verdict["status"] == "pass"
    assert verdict["retryable"] is False
    assert verdict["failure_code"] == "llm_audit_unresolved"
    codes = [c["code"] for c in update["cautions"]]
    assert "EDA_SELF_VALIDATION_FAILED" in codes
    caution = next(c for c in update["cautions"] if c["code"] == "EDA_SELF_VALIDATION_FAILED")
    assert "3.575" in caution["message_ko"]


def test_llm_audit_non_capped_retry_does_not_add_caution(monkeypatch) -> None:
    """캡에 안 걸렸으면(재시도 가치 있음) 여전히 그냥 retry만 하고 caution은 안 붙는다."""
    reply = '{"status":"retry","retry_target":"insight","reason":"환각 의심","feedback":"수정하라"}'
    monkeypatch.setattr(validator_mod, "get_llm", lambda: _FakeLLM(reply))

    state = _llm_audit_state()
    state["validation_retries"] = 0

    update = validator_node(state)

    assert update["validation_result"]["status"] == "retry"
    assert "cautions" not in update


def test_analysis_facts_included_in_llm_audit_prompt(monkeypatch) -> None:
    """#194 후속 — statistical_metadata엔 없어도 분석노드 facts에 있는 수치는
    '지어낸 것'이 아니라고 판단할 근거를 validator 프롬프트가 받는지 확인한다.
    (run-019f756b 실제 사례: distribution_node가 grouped_box로 계산한 그룹별
    수치가 statistical_metadata.group_comparison엔 안 들어가 오탐이 났었음)"""
    captured: dict = {}

    class _CapturingLLM(_FakeLLM):
        def invoke(self, prompt: str):
            captured["prompt"] = prompt
            return super().invoke(prompt)

    reply = '{"status":"pass","retry_target":"none","reason":"정상","feedback":""}'
    monkeypatch.setattr(validator_mod, "get_llm", lambda: _CapturingLLM(reply))

    state = _llm_audit_state()
    state["distribution_facts"] = ["가격대별 리뷰점수 분포는 모두 중앙값 4, Q1=2, Q3=5(IQR=3)로 유사하다."]

    validator_node(state)

    assert "가격대별 리뷰점수 분포는 모두 중앙값 4" in captured["prompt"]
    assert "분석 노드별 핵심 사실" in captured["prompt"]


def test_missing_analysis_facts_renders_placeholder(monkeypatch) -> None:
    """분석노드 facts가 하나도 없는 state(구버전 호환)에서도 프롬프트 생성이 안 깨진다."""
    captured: dict = {}

    class _CapturingLLM(_FakeLLM):
        def invoke(self, prompt: str):
            captured["prompt"] = prompt
            return super().invoke(prompt)

    reply = '{"status":"pass","retry_target":"none","reason":"정상","feedback":""}'
    monkeypatch.setattr(validator_mod, "get_llm", lambda: _CapturingLLM(reply))

    validator_node(_llm_audit_state())

    assert "(없음)" in captured["prompt"]


def test_run_eda_self_check_flags_validator_failure() -> None:
    cautions = [{
        "code": "EDA_SELF_VALIDATION_FAILED",
        "source": "eda_validator",
        "message_ko": "실패 사유",
    }]

    checks = run_eda_self_check(["artifact-1"], {"columns": ["a"]}, cautions)
    validation_check = next(c for c in checks if c.name == "eda_self_validation")

    assert validation_check.passed is False
    assert validation_check.severity == "warning"
    assert validation_check.detail == "실패 사유"


def test_run_eda_self_check_passes_without_validator_failure() -> None:
    checks = run_eda_self_check(["artifact-1"], {"columns": ["a"]}, [])
    validation_check = next(c for c in checks if c.name == "eda_self_validation")

    assert validation_check.passed is True
    assert validation_check.severity == "info"


def test_run_eda_self_check_flags_retryable_validator_failure_as_error() -> None:
    cautions = [{
        "code": "EDA_SELF_VALIDATION_FAILED",
        "source": "eda_validator",
        "message_ko": "retryable failure",
        "details": {"failure_code": "insight_fallback", "retryable": True},
    }]

    checks = run_eda_self_check(["artifact-1"], {"columns": ["a"]}, cautions)
    validation_check = next(c for c in checks if c.name == "eda_self_validation")

    assert validation_check.passed is False
    assert validation_check.severity == "error"


def test_run_eda_validation_findings_separates_retryable_error_from_limitation() -> None:
    findings = run_eda_validation_findings([
        {
            "code": "EDA_SELF_VALIDATION_FAILED",
            "source": "eda_validator",
            "message_ko": "manual review only",
            "details": {"failure_code": "missing_statistical_metadata", "retryable": False},
        },
        {
            "code": "EDA_SELF_VALIDATION_FAILED",
            "source": "eda_validator",
            "message_ko": "rerun can help",
            "details": {"failure_code": "insight_fallback", "retryable": True},
        },
    ])

    assert [finding.disposition for finding in findings] == ["limitation", "error"]
    assert [finding.retryable for finding in findings] == [False, True]
