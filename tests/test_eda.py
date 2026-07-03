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
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight import (
    compute_categorical_distribution,
    compute_correlation_pairs,
    compute_group_comparison,
    compute_numeric_distribution,
)
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
    route_after_validator,
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


# ─────────────────────────────
# insight.compute_group_comparison (커밋4 효과크기 — 리팩터로 분리해 테스트 가능해짐)
# ─────────────────────────────
def test_compute_group_comparison_no_key_returns_empty():
    df = pd.DataFrame({"value": [1, 2, 3]})
    assert compute_group_comparison(df, None, ["value"]) == {}


def test_compute_group_comparison_strong_effect_is_large():
    # 그룹 안은 일정, 그룹끼리 크게 다름 → 그룹이 분산을 거의 다 설명 → eta 큼
    df = pd.DataFrame({"cat": ["a"] * 10 + ["b"] * 10, "value": [1.0] * 10 + [100.0] * 10})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["eta_interpretation"] == "large"
    assert out["value"]["eta_squared"] > 0.9


def test_compute_group_comparison_ss_total_zero_gives_none_eta():
    # 전부 같은 값 → SS_total==0 → 분모 0 가드가 eta를 None으로 (조용히 처리)
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "value": [5.0, 5.0, 5.0, 5.0]})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["eta_squared"] is None
    assert out["value"]["eta_interpretation"] == "undefined"


def test_compute_group_comparison_aggregated_skips_eta():
    # 그룹당 1행(집계본) → eta 무의미 → skipped_aggregated
    df = pd.DataFrame({"cat": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["eta_squared"] is None
    assert out["value"]["eta_interpretation"] == "skipped_aggregated"


def test_compute_group_comparison_spread_ratio_none_when_min_not_positive():
    # 최저 그룹 평균이 0 이하 → 배율 무의미 → None
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "value": [0.0, 0.0, 10.0, 10.0]})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["spread_ratio"] is None


# ─────────────────────────────
# insight.compute_numeric_distribution (커밋1 수치 분포)
# ─────────────────────────────
def test_compute_numeric_distribution_right_skew_suggests_log():
    df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 2, 3, 10, 50, 200]})   # 양수·우측 치우침
    out = compute_numeric_distribution(df, ["x"])
    assert out["x"]["eda_notes"]["shape"] == "right_skewed"
    assert "log_transform" in out["x"]["eda_notes"]["recommended_handling"]
    assert out["x"]["all_positive"] is True


def test_compute_numeric_distribution_has_zero_flags():
    df = pd.DataFrame({"x": [0, 1, 2, 3]})
    out = compute_numeric_distribution(df, ["x"])
    assert out["x"]["has_zero"] is True
    assert out["x"]["all_positive"] is False
    assert out["x"]["non_negative"] is True


def test_compute_numeric_distribution_empty_series():
    df = pd.DataFrame({"x": [None, None]})
    out = compute_numeric_distribution(df, ["x"])
    assert out["x"] == {"type": "numeric", "unique_count": 0}


# ─────────────────────────────
# insight.compute_categorical_distribution (커밋2 범주 분포)
# ─────────────────────────────
def test_compute_categorical_distribution_detects_id_like():
    # 전부 유니크 → id 같음 → top_values 스킵
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "n": [1, 2, 3, 4]})
    out = compute_categorical_distribution(df, ["n"])
    assert out["id"]["is_id_like"] is True
    assert "top_values" not in out["id"]


def test_compute_categorical_distribution_dominated_balance():
    df = pd.DataFrame({"cat": ["a"] * 9 + ["b"], "n": list(range(10))})
    out = compute_categorical_distribution(df, ["n"])
    assert out["cat"]["eda_notes"]["balance"] == "dominated"
    assert out["cat"]["top1_share"] > 0.5


# ─────────────────────────────
# insight.compute_correlation_pairs (커밋3 관계)
# ─────────────────────────────
def test_compute_correlation_pairs_strong_pair_gets_binned_trend():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 100)
    y = -0.8 * x + rng.normal(0, 5, 100)          # 강한 음의 상관
    df = pd.DataFrame({"x": x, "y": y})
    entry = compute_correlation_pairs(df, ["x", "y"])["corr_x_vs_y"]
    assert entry["pearson_r"] < -0.5
    assert "binned_trend" in entry                # 강한 쌍 → binned_trend 붙음


def test_compute_correlation_pairs_needs_two_numeric():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert compute_correlation_pairs(df, ["x"]) == {}


# ─────────────────────────────
# eta_interpretation 경계값 (0.06 / 0.14) 자체 검증
#   두 그룹 [-1,1] / [d-1,d+1] → eta = d^2 / (d^2 + 4) 로 경계를 정확히 겨냥
# ─────────────────────────────
def test_eta_boundary_below_006_is_small():
    # d=0.5 → eta ≈ 0.059 (< 0.06) → small
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "value": [-1.0, 1.0, -0.5, 1.5]})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["eta_interpretation"] == "small"


def test_eta_boundary_between_006_and_014_is_medium():
    # d=0.7 → eta ≈ 0.109 (0.06~0.14) → medium (small/medium 경계 뮤테이션을 잡음)
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "value": [-1.0, 1.0, -0.3, 1.7]})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["eta_interpretation"] == "medium"


def test_eta_boundary_above_014_is_large():
    # d=0.9 → eta ≈ 0.168 (> 0.14) → large (medium/large 경계를 잡음)
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "value": [-1.0, 1.0, -0.1, 1.9]})
    out = compute_group_comparison(df, "cat", ["value"])
    assert out["value"]["eta_interpretation"] == "large"


# ─────────────────────────────
# 그래프 구조 (라우팅 배선 + 컴파일)
# ─────────────────────────────
def test_route_after_validator_retry_to_valid_target():
    state = {"validation_result": {"status": "retry", "retry_target": "insight"}}
    assert route_after_validator(state) == "insight"


def test_route_after_validator_invalid_target_falls_to_chart_selector():
    # 유효하지 않은 retry_target → 되돌리지 않고 chart_selector로 (폴백)
    state = {"validation_result": {"status": "retry", "retry_target": "not_a_node"}}
    assert route_after_validator(state) == "chart_selector"


def test_route_after_validator_pass_goes_to_chart_selector():
    assert route_after_validator({"validation_result": {"status": "pass"}}) == "chart_selector"


def test_eda_graph_compiles():
    # 전체 그래프 배선이 오류 없이 컴파일되는지 (LLM 호출 없음)
    from DATA_Analyst_Assistant_Agent.agents.eda.graph import build_app
    assert build_app() is not None
