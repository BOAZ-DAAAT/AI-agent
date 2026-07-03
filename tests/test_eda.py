"""EDA 에이전트 결정론 함수 단위 테스트 (LLM 없음, 토큰 0).

EDA 코드는 mini-ReAct/LLM 부분과 순수 계산 부분이 섞여 있다. 여기서는 재현 가능한
순수 계산 부분(결측 진단·grain 판정·표본 신뢰도·계약형 cautions·가설 교정)만 검증한다.
무거운 에이전트/supervisor 임포트를 피하려고 lib 함수를 직접 임포트한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_requests import from_relationship_skill
from DATA_Analyst_Assistant_Agent.agents.eda.lib.clustering_skill import _select_k
from DATA_Analyst_Assistant_Agent.agents.eda.lib.missing import detect_missing
from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import (
    assess_sample_reliability,
    build_analysis_constraints,
    build_cautions,
    correct_hypothesis_feasibility,
    detect_data_level,
)
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner import (
    ANALYSIS_TOOLS,
    route_after_planner,
)
from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.hypothesis import _resolve_target
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.tool_runner import run_node_with_retry
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.validator import (
    _check_chart_requests,
    _deterministic_fail,
)


# ─────────────────────────────
# missing
# ─────────────────────────────
def test_detect_missing_counts_and_ratio():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    out = detect_missing(df)
    assert out["missing_count"]["a"] == 2
    assert out["missing_count"]["b"] == 0
    assert out["missing_ratio"]["a"] == 0.5


# ─────────────────────────────
# detect_data_level (grain 판정)
# ─────────────────────────────
def test_detect_data_level_aggregated_one_row_per_key():
    # category당 1행 → 집계본
    df = pd.DataFrame({"category": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    dl = detect_data_level(df, key_col="category", numeric_cols=["value"])
    assert dl["level"] == "aggregated"
    assert dl["is_aggregated"] is True
    assert dl["raw_observation_level_available"] is False


def test_detect_data_level_raw_multiple_rows_per_key():
    # category당 여러 행 → 원본
    df = pd.DataFrame({"category": ["a", "a", "b", "b", "b"], "value": [1, 2, 3, 4, 5]})
    dl = detect_data_level(df, key_col="category", numeric_cols=["value"])
    assert dl["level"] == "raw"
    assert dl["is_aggregated"] is False


# ─────────────────────────────
# assess_sample_reliability
# ─────────────────────────────
def test_assess_sample_reliability_flags_low_n_groups():
    # count_col 합이 기준 미만인 그룹을 잡는다
    df = pd.DataFrame({"category": ["a", "b"], "n_orders": [2, 500]})
    out = assess_sample_reliability(df, key_col="category", count_col="n_orders",
                                    data_level="aggregated", min_n=30)
    assert out["low_n_count"] == 1
    assert out["low_n_groups"][0]["group"] == "a"


# ─────────────────────────────
# build_cautions (계약형 구조체)
# ─────────────────────────────
def test_build_cautions_aggregated_is_structured_with_constraint_link():
    dl = {"level": "aggregated", "grain_hint": "one row per category"}
    rel = {"low_n_count": 0, "low_n_groups": [], "threshold": 30}
    cautions = build_cautions(dl, rel)
    codes = {c["code"] for c in cautions}
    assert "AGGREGATED_DATA" in codes
    agg = next(c for c in cautions if c["code"] == "AGGREGATED_DATA")
    assert agg["source"] == "rule"
    assert agg["severity"] == "high"
    assert agg["constraint_ids"] == ["C001"]   # 계약과 연결


def test_build_cautions_small_group():
    dl = {"level": "raw"}
    rel = {"low_n_count": 2, "threshold": 30,
           "low_n_groups": [{"group": "x", "n": 3}, {"group": "y", "n": 5}]}
    cautions = build_cautions(dl, rel)
    codes = {c["code"] for c in cautions}
    assert "SMALL_GROUP_SIZE" in codes
    small = next(c for c in cautions if c["code"] == "SMALL_GROUP_SIZE")
    assert small["affected_groups"][0]["group"] == "x"


# ─────────────────────────────
# build_analysis_constraints (hard 계약)
# ─────────────────────────────
def test_build_analysis_constraints_aggregated_blocks_individual_ops():
    dl = {"level": "aggregated", "grain_hint": "one row per category"}
    cons = build_analysis_constraints(dl)
    assert len(cons) == 1
    c = cons[0]
    assert c["id"] == "C001"
    assert "individual_level_regression_interpretation" in c["blocked_operations"]
    assert c["unless"] == "raw_observation_level_data_is_provided"


def test_build_analysis_constraints_raw_is_empty():
    cons = build_analysis_constraints({"level": "raw"})
    assert cons == []


# ─────────────────────────────
# correct_hypothesis_feasibility (자기교정: 불가능한 검정 강제 교정)
# ─────────────────────────────
def test_correct_hypothesis_feasibility_blocks_group_test_on_aggregated():
    # category당 1행(집계본)이면 ANOVA/그룹 간 검정은 수학적으로 불가 → 교정돼야 함
    df = pd.DataFrame({"category": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    hyp = (
        "[가설 1]\n"
        "검증방법: ANOVA\n"
        "필요변수: feature=[category]\n"
        "현재데이터: 현재 마트로 검증 가능"
    )
    new_text, corrections = correct_hypothesis_feasibility(
        df, hyp, data_level={"level": "aggregated"})
    assert corrections                      # 교정 발생
    assert "추가 필요" in new_text            # 라벨이 '추가 필요'로 교정됨


def test_correct_hypothesis_feasibility_keeps_feasible_test_on_raw():
    # 그룹당 여러 행이면 그룹 간 검정 가능 → 교정하지 않음
    df = pd.DataFrame({"category": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})
    hyp = (
        "[가설 1]\n"
        "검증방법: ANOVA\n"
        "필요변수: feature=[category]\n"
        "현재데이터: 현재 마트로 검증 가능"
    )
    new_text, corrections = correct_hypothesis_feasibility(
        df, hyp, data_level={"level": "raw"})
    assert not corrections                  # 교정 없음
    assert "추가 필요" not in new_text


# ─────────────────────────────
# planner (실행 가능한 분석 필터 = 결정론 가드레일)
# ─────────────────────────────
def _shape(n_numeric=0, n_cat=0, n_time=0, row_count=0):
    return {"n_numeric": n_numeric, "n_cat": n_cat, "n_time": n_time, "row_count": row_count}


def _feasible(shape):
    """데이터 shape로 precond를 통과하는 분석 이름 집합."""
    return {t["name"] for t in ANALYSIS_TOOLS if t["precond"](shape)}


def test_planner_precond_single_numeric_only_allows_distribution_quality():
    # 수치 1개뿐 → 분포/품질만 가능 (상관·비교·시간·군집 불가)
    feas = _feasible(_shape(n_numeric=1, row_count=5))
    assert "distribution" in feas
    assert "quality" in feas
    assert "relationship" not in feas       # 수치 2개 필요
    assert "comparison" not in feas         # 범주 필요
    assert "time" not in feas
    assert "clustering" not in feas


def test_planner_precond_relationship_needs_two_numeric():
    assert "relationship" not in _feasible(_shape(n_numeric=1, row_count=100))
    assert "relationship" in _feasible(_shape(n_numeric=2, row_count=100))


def test_planner_precond_comparison_needs_cat_and_numeric():
    assert "comparison" not in _feasible(_shape(n_numeric=1, n_cat=0))
    assert "comparison" in _feasible(_shape(n_numeric=1, n_cat=1))


def test_planner_precond_time_needs_time_column():
    assert "time" not in _feasible(_shape(n_numeric=2))
    assert "time" in _feasible(_shape(n_numeric=2, n_time=1))


def test_planner_precond_clustering_needs_enough_rows():
    assert "clustering" not in _feasible(_shape(n_numeric=2, row_count=5))    # 행 부족
    assert "clustering" in _feasible(_shape(n_numeric=2, row_count=100))


def test_planner_precond_quality_always_feasible():
    # 빈 데이터에도 품질 점검은 항상 가능
    assert "quality" in _feasible(_shape())


def test_route_after_planner_goes_to_chosen_analysis():
    assert route_after_planner({"next_analysis": "distribution"}) == "distribution"


def test_route_after_planner_done_goes_to_insight():
    assert route_after_planner({"next_analysis": "done"}) == "insight"


def test_route_after_planner_unknown_goes_to_insight():
    # 환각 방지: 유효하지 않은 선택은 insight로 (분석 노드로 안 감)
    assert route_after_planner({"next_analysis": "not_a_real_node"}) == "insight"


# ─────────────────────────────
# validator 순수 함수 (LLM 없는 결정론 체크)
# ─────────────────────────────
def _ok_state(**over):
    """검증 통과하는 기본 state (아래 테스트에서 일부만 망가뜨린다)."""
    base = {
        "insight_result": "카테고리 집중 구조가 관찰된다",
        "hypotheses": "[가설 1] ...",
        "controller_log": [{"choice": "distribution"}],
        "statistical_metadata": {"row_count": 10},
    }
    base.update(over)
    return base


def test_deterministic_fail_empty_insight_targets_insight():
    target, _ = _deterministic_fail(_ok_state(insight_result="인사이트 생성 실패"))
    assert target == "insight"


def test_deterministic_fail_no_analysis_targets_planner():
    target, _ = _deterministic_fail(_ok_state(controller_log=[]))
    assert target == "planner"


def test_deterministic_fail_empty_metadata_targets_planner():
    target, _ = _deterministic_fail(_ok_state(statistical_metadata={}))
    assert target == "planner"


def test_deterministic_fail_passes_when_all_present():
    assert _deterministic_fail(_ok_state()) is None


def test_check_chart_requests_flags_missing_intent_and_stats():
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"a": [1, 2]})))
    try:
        # intent·stats 없음, 컬럼(a)은 존재
        issues = _check_chart_requests({"chart_requests": [{"columns": {"a": {}}}]})
    finally:
        reset_context()
    assert any("intent" in i for i in issues)
    assert any("stats" in i for i in issues)


def test_check_chart_requests_flags_nonexistent_column():
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"real_col": [1, 2]})))
    try:
        state = {"chart_requests": [{"intent": "x", "columns": {"ghost_col": {}}, "stats": {"m": 1}}]}
        issues = _check_chart_requests(state)
    finally:
        reset_context()
    assert any("존재하지 않는 컬럼" in i for i in issues)


def test_check_chart_requests_clean_passes():
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"a": [1, 2]})))
    try:
        state = {"chart_requests": [{"intent": "a 분포", "columns": {"a": {}}, "stats": {"mean": 1.5}}]}
        issues = _check_chart_requests(state)
    finally:
        reset_context()
    assert issues == []


# ─────────────────────────────
# hypothesis._resolve_target (target 컬럼 우선순위, LLM 없음)
# ─────────────────────────────
def test_resolve_target_prefers_plan_metric():
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"a": [1], "b": [2]}), measure_cols=["a", "b"]))
    try:
        assert _resolve_target({"plan_metric": "b"}) == "b"
    finally:
        reset_context()


def test_resolve_target_falls_back_to_measure_cols():
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"a": [1]}), measure_cols=["a"]))
    try:
        assert _resolve_target({}) == "a"      # plan_metric 없음 → measure_cols
    finally:
        reset_context()


def test_resolve_target_ignores_column_not_in_df():
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"a": [1]}), measure_cols=[]))
    try:
        assert _resolve_target({"plan_metric": "ghost"}) == ""   # 실제 컬럼 아님 → ""
    finally:
        reset_context()


# ─────────────────────────────
# tool_runner.run_node_with_retry (재시도 로직, 가짜 콜러블로 검증)
# ─────────────────────────────
def test_run_node_with_retry_succeeds_first_try():
    result, err = run_node_with_retry(lambda: "ok", "node", max_retries=2)
    assert result == "ok"
    assert err is None


def test_run_node_with_retry_recovers_after_some_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("일시 실패")
        return "recovered"

    result, err = run_node_with_retry(flaky, "node", max_retries=3)
    assert result == "recovered"
    assert err is None


def test_run_node_with_retry_returns_fallback_when_all_fail():
    def always_fail():
        raise RuntimeError("boom")

    result, err = run_node_with_retry(always_fail, "node", fallback="FALLBACK", max_retries=1)
    assert result == "FALLBACK"
    assert err is not None and "boom" in err


# ─────────────────────────────
# clustering._select_k (실루엣 기반 k 선택)
# ─────────────────────────────
def test_select_k_small_data_returns_two():
    X = np.zeros((5, 2))                        # 10행 미만 → 2 고정
    assert _select_k(X, range(2, 6)) == 2


def test_select_k_picks_three_for_three_clusters():
    rng = np.random.default_rng(0)
    c1 = rng.normal([0, 0], 0.15, (30, 2))
    c2 = rng.normal([6, 6], 0.15, (30, 2))
    c3 = rng.normal([0, 6], 0.15, (30, 2))
    X = np.vstack([c1, c2, c3])                # 뚜렷한 3군집
    assert _select_k(X, range(2, 7)) == 3


# ─────────────────────────────
# chart_requests.from_relationship_skill (스킬 결과 → 차트 주문서 변환)
# ─────────────────────────────
def test_from_relationship_skill_emits_scatter_and_heatmap():
    result = {
        "correlation": {"stats": {"correlation_matrix": {"a": {}, "b": {}}, "strong_pairs": []}},
        "scatter_pairs": {"stats": {"a vs b": {"pearson_r": -0.7}}},
    }
    reqs = from_relationship_skill(result)
    hints = {r["hint"] for r in reqs}
    assert "heatmap" in hints                  # 상관 행렬
    assert "scatter" in hints                   # 강한 쌍 산점도
    scatter = next(r for r in reqs if r["hint"] == "scatter")
    assert "a" in scatter["columns"] and "b" in scatter["columns"]
