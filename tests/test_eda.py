"""EDA 에이전트 결정론 함수 단위 테스트 (LLM 없음, 토큰 0).

EDA 코드는 mini-ReAct/LLM 부분과 순수 계산 부분이 섞여 있다. 여기서는 재현 가능한
순수 계산 부분(결측 진단·grain 판정·표본 신뢰도·계약형 cautions·가설 교정)만 검증한다.
무거운 에이전트/supervisor 임포트를 피하려고 lib 함수를 직접 임포트한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_requests import (
    from_clustering_skill,
    from_comparison_skill,
    from_relationship_skill,
    from_time_skill,
)
from DATA_Analyst_Assistant_Agent.agents.eda.lib.clustering_skill import _select_k
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight import (
    classify_semantic_categorical,
    classify_semantic_numeric,
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


# ─────────────────────────────
# chart_requests 나머지 from_* (comparison/time/clustering)
# ─────────────────────────────
def test_from_comparison_skill_emits_bar_for_top_n():
    result = {"top_n_barplot": {"stats": {"review_score": {"top": []}}}}
    reqs = from_comparison_skill(result, key_col="category", measure_cols=["review_score"])
    assert any(r["hint"] == "bar" for r in reqs)


def test_from_time_skill_emits_line_for_timeseries():
    result = {"timeseries": {"stats": {"revenue": {"trend": "up"}}}}
    reqs = from_time_skill(result, time_cols=["month"])
    assert any(r["hint"] == "line" for r in reqs)


def test_from_clustering_skill_emits_heatmap():
    result = {"n_clusters": 2, "silhouette_score": 0.5,
              "cluster_centroids": {"0": {"x": 1.0}, "1": {"x": 2.0}}}
    reqs = from_clustering_skill(result)
    assert reqs and reqs[0]["hint"] == "heatmap"


def test_from_clustering_skill_skip_returns_empty():
    assert from_clustering_skill({"skip": True}) == []


# ─────────────────────────────
# LLM 노드 (가짜 LLM 주입 = FakeChatModel 패턴)
#   진짜 LLM 없이 노드의 LLM 감사 흐름을 결정론적으로 검증한다.
# ─────────────────────────────
class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """고정 응답만 돌려주는 가짜 LLM (진짜 호출 없음)."""
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        return _FakeResp(self._content)


def test_validator_node_routes_retry_when_llm_verdict_is_retry(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.validator as V
    verdict_json = '{"status": "retry", "retry_target": "insight", "reason": "근거 약함", "feedback": "보완 필요"}'
    monkeypatch.setattr(V, "get_llm", lambda: _FakeLLM(verdict_json))

    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"a": [1, 2]})))
    state = {                                   # 결정론 체크는 통과 → LLM 감사로 진입
        "insight_result": "카테고리 집중 구조가 관찰된다",
        "hypotheses": "[가설 1] ...",
        "final_summary": "",
        "controller_log": [{"choice": "distribution"}],
        "statistical_metadata": {"row_count": 2},
        "validation_retries": 0,
        "user_question": "q", "question_type": "",
        "chart_requests": [],
    }
    try:
        update = V.validator_node(state)
    finally:
        reset_context()

    assert update["validation_result"]["status"] == "retry"
    assert update["validation_result"]["retry_target"] == "insight"
    assert update["validation_retries"] == 1    # 재시도 카운트 증가


def test_planner_node_picks_feasible_analysis_from_llm(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner as P
    monkeypatch.setattr(P, "get_llm", lambda: _FakeLLM('{"next": "distribution", "reason": "분포부터"}'))
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"x": [1.0, 2.0, 3.0]}), measure_cols=["x"]))
    state = {"user_question": "q", "question_type": "", "controller_log": [], "round": 0}
    try:
        update = P.planner_node(state)
    finally:
        reset_context()
    assert update["next_analysis"] == "distribution"


def test_planner_node_rejects_infeasible_llm_choice(monkeypatch):
    # LLM이 실행 불가한 분석(수치 1개인데 relationship)을 고르면 done으로 방어
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner as P
    monkeypatch.setattr(P, "get_llm", lambda: _FakeLLM('{"next": "relationship", "reason": "관계"}'))
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"x": [1.0, 2.0, 3.0]}), measure_cols=["x"]))
    state = {"user_question": "q", "question_type": "", "controller_log": [], "round": 0}
    try:
        update = P.planner_node(state)
    finally:
        reset_context()
    assert update["next_analysis"] == "done"    # 환각/불가 선택 → 종료로 방어


def test_hypothesis_node_generates_from_llm(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.hypothesis as H
    monkeypatch.setattr(H, "get_llm", lambda: _FakeLLM("[가설 1] 배송이 리뷰에 영향\n검증방법: 상관"))
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"review": [1.0, 2.0, 3.0]}), measure_cols=["review"]))
    state = {
        "user_question": "q", "question_type": "",
        "statistical_metadata": {}, "data_level": {"level": "raw"},
        "sample_reliability": {}, "plan_metric": "review",
    }
    try:
        update = H.hypothesis_node(state)
    finally:
        reset_context()
    assert update.get("hypotheses")             # 가설 생성됨


def test_insight_node_produces_metadata_and_insight(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight as I
    monkeypatch.setattr(I, "get_llm", lambda: _FakeLLM("핵심 패턴: 카테고리 집중 구조"))
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"cat": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]}),
                           key_col="cat", measure_cols=["val"]))
    state = {"user_question": "q", "question_type": ""}
    try:
        update = I.insight_node(state)
    finally:
        reset_context()
    assert update["insight_result"]              # 인사이트 생성됨
    assert update["statistical_metadata"]["group_comparison"]  # 계산부도 채워짐


class _SeqLLM:
    """호출 순서대로 다른 응답 (같은 인스턴스 재사용 시 카운터 유지)."""
    def __init__(self, contents):
        self._contents = list(contents)
        self._i = 0

    def invoke(self, prompt):
        c = self._contents[min(self._i, len(self._contents) - 1)]
        self._i += 1
        return _FakeResp(c)


class _BoomLLM:
    def invoke(self, prompt):
        raise RuntimeError("boom")


def test_insight_node_merges_llm_inferred_caution(monkeypatch):
    # 첫 콜(_llm_infer_cautions)=caution JSON, 둘째 콜(insight)=텍스트 → source=llm_inferred 병합 확인
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight as I
    caution_json = ('[{"code": "POSSIBLE_SEASONALITY", "severity": "low", '
                    '"message_ko": "계절성 가능", "recommended_action": ["check"], '
                    '"evidence_keys": ["time_result"]}]')
    monkeypatch.setattr(I, "get_llm", lambda shared=_SeqLLM([caution_json, "핵심 패턴"]): shared)
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"cat": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]}),
                           key_col="cat", measure_cols=["val"]))
    try:
        update = I.insight_node({"user_question": "q", "question_type": ""})
    finally:
        reset_context()
    cautions = update["statistical_metadata"]["cautions"]
    assert any(c.get("source") == "llm_inferred" for c in cautions)


def test_insight_node_falls_back_when_llm_raises(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight as I
    monkeypatch.setattr(I, "get_llm", lambda: _BoomLLM())
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"cat": ["a", "a", "b"], "val": [1.0, 2.0, 3.0]}),
                           key_col="cat", measure_cols=["val"]))
    try:
        update = I.insight_node({"user_question": "q", "question_type": ""})
    finally:
        reset_context()
    assert "실패" in update["insight_result"]     # run_node_with_retry 폴백


def test_hypothesis_node_falls_back_when_llm_raises(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.hypothesis as H
    monkeypatch.setattr(H, "get_llm", lambda: _BoomLLM())
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"review": [1.0, 2.0, 3.0]}), measure_cols=["review"]))
    state = {"user_question": "q", "question_type": "",
             "statistical_metadata": {}, "data_level": {"level": "raw"},
             "sample_reliability": {}, "plan_metric": "review"}
    try:
        update = H.hypothesis_node(state)
    finally:
        reset_context()
    assert "실패" in update["hypotheses"]          # run_node_with_retry 폴백


# ─────────────────────────────
# semantic_type 분류 + handling 교정 (순수 코드, 토큰 0)
# ─────────────────────────────
def _num(col, values):
    s = pd.Series(values, dtype="float64").dropna()
    return classify_semantic_numeric(
        col, s, int(s.nunique()), bool(s.min() > 0), bool(s.min() >= 0))


def test_semantic_numeric_rating_name_and_value_agree_high():
    st, conf = _num("review_score", list(range(1, 6)) * 40)          # 정수 1~5
    assert st == "rating" and conf == "high"


def test_semantic_numeric_monetary_name_and_positive_values_high():
    st, conf = _num("order_price", [12.5, 88.0, 250.3, 9.9] * 30)     # 이름 price + 양수 실수
    assert st == "monetary" and conf == "high"


def test_semantic_numeric_count_name_and_value_agree():
    st, conf = _num("num_items", [0, 1, 2, 3, 1, 2, 0, 5] * 30)       # num_ + 음수없는 정수
    assert st == "count" and conf == "high"


def test_semantic_numeric_id_from_near_unique_integers():
    st, conf = _num("txn_id", list(range(500)))                      # 이름 _id + unique≈n
    assert st == "id" and conf == "high"


def test_semantic_numeric_name_value_disagree_is_medium():
    st, conf = _num("delivery_days", list(range(1, 41)))             # 이름 duration vs 값 count
    assert st == "duration" and conf == "medium"


def test_semantic_numeric_no_signal_is_generic_low():
    st, conf = _num("x", [-3.2, 0.5, -1.1, 4.4, -9.9])              # 이름·값 신호 없음
    assert st == "generic" and conf == "low"


def test_semantic_categorical_id_when_near_unique():
    assert classify_semantic_categorical("order_id", True)[0] == "id"


def test_semantic_categorical_category_when_repeated():
    st, conf = classify_semantic_categorical("product_category", False)
    assert st == "category" and conf == "high"


def test_handling_rating_excludes_log_and_outlier_removal():
    df = pd.DataFrame({"review_score": [1, 2, 3, 4, 5, 5, 5, 4, 1] * 40})
    h = compute_numeric_distribution(df, ["review_score"])["review_score"]["eda_notes"]["recommended_handling"]
    assert "treat_as_ordinal" in h
    assert not any("log" in x for x in h)          # 평점에 log 변환 추천 금지
    assert "avoid_naive_outlier_removal" not in h  # 1점은 이상치가 아니라 진짜 값


def test_handling_id_is_exclude_from_analysis():
    df = pd.DataFrame({"txn_id": list(range(300))})
    h = compute_numeric_distribution(df, ["txn_id"])["txn_id"]["eda_notes"]["recommended_handling"]
    assert h == ["exclude_from_analysis"]


def test_insight_node_aggregated_key_col_becomes_group_key(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight as I
    monkeypatch.setattr(I, "get_llm", lambda: _FakeLLM("[]"))       # LLM 호출 폴백
    df = pd.DataFrame({
        "product_category": [f"c{i}" for i in range(40)],           # 카테고리당 1행 = 집계본
        "avg_review_score": np.linspace(3.0, 5.0, 40),
        "total_orders": np.arange(100, 140),
    })
    reset_context()
    set_context(EdaContext(df=df, key_col="product_category",
                           measure_cols=["avg_review_score", "total_orders"]))
    try:
        update = I.insight_node({"user_question": "카테고리 비교", "question_type": ""})
    finally:
        reset_context()
    key_entry = update["statistical_metadata"]["distribution"]["product_category"]
    assert key_entry["semantic_type"] == "group_key"       # id 오판이 group_key로 교정
    assert key_entry["semantic_confidence"] == "high"
    assert key_entry["is_id_like"] is False


# ─────────────────────────────
# get_llm model_env 배선 (codegen 전용 모델 라우팅 선행) — 토큰 0
# ─────────────────────────────
def test_get_llm_caches_per_model_env(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda._runtime as R

    calls = []

    class _Sentinel:
        def __init__(self, model_env):
            self.model_env = model_env

    def fake_get_chat_model(*, temperature=0, model_env="LLM_MODEL", **kw):
        calls.append(model_env)
        return _Sentinel(model_env)

    monkeypatch.setattr(R, "get_chat_model", fake_get_chat_model)
    R._llms.clear()
    try:
        default = R.get_llm()
        same = R.get_llm()                              # 같은 model_env → 캐시 재사용
        codegen = R.get_llm("CODE_GENERATOR_MODEL")     # 다른 model_env → 별도 인스턴스
        assert default is same
        assert default.model_env == "LLM_MODEL"         # 기본값 = 하위호환
        assert codegen.model_env == "CODE_GENERATOR_MODEL"
        assert default is not codegen
        assert calls == ["LLM_MODEL", "CODE_GENERATOR_MODEL"]  # 캐시 히트는 재생성 안 함
    finally:
        R._llms.clear()
