"""hypothesis_screening 테스트 — 순수 코드(LLM/백엔드 호출 0, 토큰 0).

검증 대상(사용자 지정 5 + 엣지):
  1) 관계/회귀에서 pearson·spearman 둘 다 |r|<0.05면 드롭
  2) 전부 드롭 후보면 최소 1개 보존(드롭 취소)
  3) 미측정(군집 skip·분류 등)는 드롭되지 않음
  4) 재실행 시 '사전신호:' 태그가 중복되지 않음
  5) 재정렬 순서: 강함 > 중간 > 미측정 > 약함
  + 트레일링([다음 분석 방향]) 보존 / silhouette 음수 처리 / 블록 없음 폴백 /
    컬럼 퍼지매칭 / 설명가능성 메타(matched_signal·score_value·drop_reason)
"""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.lib.hypothesis_screening import screen_hypotheses


def _block(n: int, obs: str, typ: str, target: str, features: str, method: str = "검증방법") -> str:
    return (
        f"[가설 {n}]\n"
        f"관찰: {obs}\n"
        f"유형: {typ}\n"
        f"H0: 귀무\n"
        f"H1: 대립\n"
        f"검증방법: {method}\n"
        f"필요변수: target=[{target}], feature=[{features}]\n"
        f"현재데이터: 검증 가능"
    )


def _corr(pear: float, spear: float, n: int = 5000) -> dict:
    return {"pearson_r": pear, "spearman_r": spear, "nonlinearity": "linear", "n": n}


def _count_tags(text: str) -> int:
    return sum(1 for ln in text.split("\n") if ln.strip().startswith("사전신호:"))


# ── 1) 명백 무상관 드롭 ────────────────────────────────────────────────────────
def test_drop_relationship_when_both_r_below_005():
    text = "\n\n".join([
        _block(1, "OBS_NULL", "관계추론", "monetary", "review_score", "스피어만 상관검정"),
        _block(2, "OBS_STRONG", "회귀", "review_score", "delivery_days", "단순선형회귀"),
    ])
    stat = {"correlation_pairs": {
        "corr_monetary_vs_review_score": _corr(-0.041, -0.038),
        "corr_review_score_vs_delivery_days": _corr(-0.45, -0.43),
    }}
    new_text, meta = screen_hypotheses(text, stat)
    null_meta = next(m for m in meta if m["orig_index"] == 0)
    assert null_meta["dropped"] is True
    assert null_meta["drop_reason"]
    assert "OBS_NULL" not in new_text        # 드롭된 가설 텍스트 제거
    assert "OBS_STRONG" in new_text          # 강한 가설은 보존


# ── 2) 최소 1개 보존 ──────────────────────────────────────────────────────────
def test_keep_at_least_one_when_all_would_drop():
    text = "\n\n".join([
        _block(1, "OBS_A", "관계추론", "monetary", "review_score", "스피어만 상관검정"),
        _block(2, "OBS_B", "관계추론", "freight", "review_score", "스피어만 상관검정"),
    ])
    stat = {"correlation_pairs": {
        "corr_monetary_vs_review_score": _corr(-0.041, -0.038),
        "corr_freight_vs_review_score": _corr(0.02, 0.03),
    }}
    new_text, meta = screen_hypotheses(text, stat)
    assert all(m["dropped"] is False for m in meta)   # 전부 드롭 취소
    assert "OBS_A" in new_text and "OBS_B" in new_text


# ── 3) 미측정는 드롭 안 함 ──────────────────────────────────────────────────
def test_unverifiable_not_dropped():
    text = "\n\n".join([
        _block(1, "OBS_CLUSTER", "군집", "segment", "monetary, frequency", "K-means silhouette"),
        _block(2, "OBS_CLASS", "분류", "high_value", "monetary", "로지스틱 회귀"),
    ])
    stat = {"correlation_pairs": {}, "clustering": {"skip": True}}
    new_text, meta = screen_hypotheses(text, stat)
    assert all(m["strength"] == "미측정" for m in meta)
    assert all(m["dropped"] is False for m in meta)
    assert "OBS_CLUSTER" in new_text and "OBS_CLASS" in new_text


# ── 4) 재실행 시 태그 중복 없음 ──────────────────────────────────────────────
def test_rerun_no_duplicate_tag():
    text = "\n\n".join([
        _block(1, "OBS_1", "회귀", "review_score", "delivery_days", "단순선형회귀"),
        _block(2, "OBS_2", "관계추론", "monetary", "review_score", "스피어만 상관검정"),
    ])
    stat = {"correlation_pairs": {
        "corr_review_score_vs_delivery_days": _corr(-0.45, -0.43),
        "corr_monetary_vs_review_score": _corr(0.25, 0.22),
    }}
    once, _ = screen_hypotheses(text, stat)
    twice, _ = screen_hypotheses(once, stat)
    assert _count_tags(once) == 2
    assert _count_tags(twice) == 2            # 재실행해도 블록당 태그 1개


# ── 5) 재정렬 순서: 강함 > 중간 > 미측정 > 약함 ────────────────────────────
def test_reorder_strong_medium_unverifiable_weak():
    text = "\n\n".join([
        _block(1, "OBS_WEAK", "관계추론", "freight", "review_score", "스피어만 상관검정"),
        _block(2, "OBS_UNVER", "군집", "segment", "monetary, frequency", "K-means silhouette"),
        _block(3, "OBS_MED", "관계추론", "price", "review_score", "스피어만 상관검정"),
        _block(4, "OBS_STRONG", "회귀", "review_score", "delivery_days", "단순선형회귀"),
    ])
    stat = {
        "correlation_pairs": {
            "corr_freight_vs_review_score": _corr(0.15, 0.13),          # 약함
            "corr_price_vs_review_score": _corr(0.30, 0.28),            # 중간
            "corr_review_score_vs_delivery_days": _corr(-0.50, -0.48),  # 강함
        },
        "clustering": {"skip": True},                                   # 미측정
    }
    new_text, _ = screen_hypotheses(text, stat)
    order = [new_text.index(tok) for tok in ("OBS_STRONG", "OBS_MED", "OBS_UNVER", "OBS_WEAK")]
    assert order == sorted(order)   # 강함 < 중간 < 미측정 < 약함 (텍스트 등장 순서)


# ── 엣지: 트레일링 보존 ───────────────────────────────────────────────────────
def test_trailing_section_preserved():
    text = (
        _block(1, "OBS_X", "회귀", "review_score", "delivery_days", "단순선형회귀")
        + "\n\n[다음 분석 방향]\n1. 교란변수 후보: category\n2. 통제 방법: 다중회귀"
    )
    stat = {"correlation_pairs": {"corr_review_score_vs_delivery_days": _corr(-0.45, -0.43)}}
    new_text, meta = screen_hypotheses(text, stat)
    assert "[다음 분석 방향]" in new_text
    assert "다중회귀" in new_text
    assert len(meta) == 1                     # 트레일링을 가설로 오파싱하지 않음


# ── 엣지: silhouette 음수는 약함(abs로 뒤집히면 안 됨) ────────────────────────
def test_negative_silhouette_is_weak():
    text = _block(1, "OBS_C", "군집", "segment", "monetary, frequency", "K-means silhouette")
    stat = {"clustering": {"skip": False, "silhouette_score": -0.6, "n_clusters": 3}}
    _, meta = screen_hypotheses(text, stat)
    assert meta[0]["strength"] == "약함"
    assert meta[0]["score_value"] == -0.6


# ── 엣지: 블록 없음 폴백 ─────────────────────────────────────────────────────
def test_no_blocks_returns_unchanged():
    text = "가설 생성 실패"
    new_text, meta = screen_hypotheses(text, {"correlation_pairs": {}})
    assert new_text == text
    assert meta == []


# ── 엣지: 컬럼 퍼지매칭(줄여 쓴 변수명) ──────────────────────────────────────
def test_column_fuzzy_match():
    # 가설은 'review'로 줄여 썼지만 상관키는 'review_score'
    text = _block(1, "OBS_F", "관계추론", "monetary", "review", "스피어만 상관검정")
    stat = {"correlation_pairs": {"corr_monetary_vs_review_score": _corr(0.35, 0.33)}}
    _, meta = screen_hypotheses(text, stat)
    assert meta[0]["matched_signal"] == "corr_monetary_vs_review_score"
    assert meta[0]["strength"] == "중간"


# ── 엣지: 설명가능성 메타 필드 존재 ──────────────────────────────────────────
def test_signal_meta_fields_present():
    text = _block(1, "OBS_M", "회귀", "review_score", "delivery_days", "단순선형회귀")
    stat = {"correlation_pairs": {"corr_review_score_vs_delivery_days": _corr(-0.45, -0.43)}}
    _, meta = screen_hypotheses(text, stat)
    m = meta[0]
    for key in ("orig_index", "type", "strength", "matched_signal", "score_value",
                "drop_candidate", "dropped", "drop_reason"):
        assert key in m
    assert m["matched_signal"] == "corr_review_score_vs_delivery_days"
    assert m["score_value"] == 0.45
