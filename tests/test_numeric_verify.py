"""numeric_verify 테스트 — 숫자 환각 차단 공용 유틸(EvidencePack 등 특정 에이전트 의존 없음).

insight에서 shared로 옮긴 순수 함수라, 여기 테스트도 EvidencePack 대신 덕타이핑한
가짜 pack(table_summary/eda/analysis 속성만)으로 독립 검증한다.
"""

from __future__ import annotations

from types import SimpleNamespace

from DATA_Analyst_Assistant_Agent.shared.numeric_verify import build_evidence_corpus, verify_texts


def _pack() -> SimpleNamespace:
    """build_evidence_corpus가 훑는 세 속성만 갖춘 최소 pack (toys 500/400·auto 300·pet 100)."""
    return SimpleNamespace(
        table_summary={
            "numeric_describe": {"total_sales": {"max": 500.0, "min": 100.0}},
            "head": [
                {"category": "toys", "total_sales": 500.0},
                {"category": "toys", "total_sales": 400.0},
                {"category": "auto", "total_sales": 300.0},
                {"category": "pet", "total_sales": 100.0},
            ],
        },
        eda={},
        analysis={},
    )


def test_verify_passes_numbers_from_evidence():
    numbers, corpus = build_evidence_corpus(_pack(), [])
    ok, missing = verify_texts(["최대 매출은 500.0입니다."], numbers, corpus)
    assert ok, missing


def test_verify_percent_and_rounding_tolerance():
    numbers = {0.4176, 0.8450704}
    ok, missing = verify_texts(["비중은 41.8%이고 비율은 0.845입니다."], numbers, "[]")
    assert ok, missing


def test_verify_rejects_fabricated_number():
    numbers, corpus = build_evidence_corpus(_pack(), [])
    ok, missing = verify_texts(["매출이 7777.7로 증가했습니다."], numbers, corpus)
    assert not ok and "7777.7" in missing


def test_verify_skips_small_ordinal_but_checks_percent():
    ok, missing = verify_texts(["상위 10개 중 3개"], set(), "[]")     # 서수 → 검증 제외
    assert ok
    ok, missing = verify_texts(["10% 증가했습니다"], set(), "[]")     # %는 주장 → 검증
    assert not ok and "10" in missing


def test_verify_compute_result_becomes_citable():
    computes = [{"ok": True, "result": {"toys": 900.0}}]
    numbers, corpus = build_evidence_corpus(_pack(), computes)
    ok, missing = verify_texts(["toys 합계는 900.0입니다."], numbers, corpus)
    assert ok, missing


def test_verify_accepts_scientific_notation_pvalue():
    # 회귀 테스트: "7.76e-315" 같은 지수 표기를 "e"에서 잘라 "7.76"과 "-315"라는
    # 가짜 숫자 두 개로 오인하던 버그. 실제 리포트 생성 실패의 원인이었다.
    numbers = {7.76189325e-315, -0.36635318280995627, 4.943713072837881e-06}
    text = "피어슨 r=-0.366, p=7.76e-315를 확인하였고 판매자 수준에서는 p=4.94e-06였다."
    ok, missing = verify_texts([text], numbers, "")
    assert ok, missing


def test_verify_rejects_fabricated_scientific_notation():
    numbers = {7.76189325e-315}
    ok, missing = verify_texts(["p=1.23e-50이었다."], numbers, "")
    assert not ok and "1.23e-50" in missing
