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


def test_route_after_planner_done_with_result_goes_to_insight():
    # done + 실질 분석 결과 있음 → insight (정상 경로)
    assert route_after_planner(
        {"next_analysis": "done", "controller_log": [{"choice": "distribution"}],
         "distribution_result": "분포 요약"}) == "insight"


def test_route_after_planner_unknown_goes_to_insight():
    # 환각 방지: 유효하지 않은 선택은 insight로 (분석 노드로 안 감)
    assert route_after_planner({"next_analysis": "not_a_real_node"}) == "insight"


def test_route_after_planner_selects_codegen():
    # 질문기준: 플래너가 codegen을 직접 고르면 codegen 노드로 (존재기준 fallback 아님).
    assert route_after_planner({"next_analysis": "codegen"}) == "codegen"


def test_planner_prompt_offers_codegen_with_guardrail():
    from DATA_Analyst_Assistant_Agent.agents.eda.prompts.planner import planner_prompt
    shape = {"row_count": 100, "numeric_cols": ["a"], "cat_cols": ["b"], "time_cols": []}
    p = planner_prompt("파생 비율 질문", "", shape,
                       [{"name": "distribution", "desc": "분포"}], {},
                       round_idx=0, max_rounds=5, need_priority=False, codegen_selectable=True)
    assert "codegen" in p                       # 카드 노출
    assert "최후수단" in p and "도구 우선" in p   # over-fire 방지 울타리 문구


def test_planner_prompt_hides_codegen_when_not_selectable():
    # 기본값(codegen_selectable=False)일 땐 codegen 카드/문구가 안 나온다.
    from DATA_Analyst_Assistant_Agent.agents.eda.prompts.planner import planner_prompt
    shape = {"row_count": 100, "numeric_cols": ["a"], "cat_cols": ["b"], "time_cols": []}
    p = planner_prompt("정상 질문", "", shape,
                       [{"name": "distribution", "desc": "분포"}], {},
                       round_idx=0, max_rounds=5, need_priority=False)
    assert "codegen" not in p


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
    target, _, _ = _deterministic_fail(_ok_state(insight_result="인사이트 생성 실패"))
    assert target == "insight"


def test_deterministic_fail_no_analysis_targets_planner():
    target, _, _ = _deterministic_fail(_ok_state(controller_log=[]))
    assert target == "planner"


def test_deterministic_fail_empty_metadata_targets_planner():
    target, _, _ = _deterministic_fail(_ok_state(statistical_metadata={}))
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
    result, err = run_node_with_retry(lambda: "ok", "node")
    assert result == "ok"
    assert err is None


def test_run_node_with_retry_recovers_after_some_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("일시 실패")
        return "recovered"

    result, err = run_node_with_retry(flaky, "node")
    assert result == "recovered"
    assert err is None


def test_run_node_with_retry_returns_fallback_when_all_fail():
    def always_fail():
        raise RuntimeError("boom")

    result, err = run_node_with_retry(always_fail, "node", fallback="FALLBACK")
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


# ─────────────────────────────
# codegen AST 게이트 + denylist (안전 핵심) — 토큰 0
# ─────────────────────────────
from DATA_Analyst_Assistant_Agent.agents.eda.lib.codegen_gate import (
    CodegenRequest,
    validate_expression,
    validate_request,
)

_COLS = ["order_price", "review_score", "product_category"]


def test_gate_allows_safe_pandas_expression():
    r = validate_expression('df["order_price"].mean()', _COLS)
    assert r.ok, r.reason


def test_gate_allows_grouped_aggregation():
    r = validate_expression('df.groupby("product_category")["order_price"].median()', _COLS)
    assert r.ok, r.reason


def test_gate_rejects_multiple_statements():
    r = validate_expression('x = 1\ndf.mean()', _COLS)      # eval 모드 파싱 실패
    assert not r.ok and "single_expression" in r.reason


def test_gate_rejects_import_name():
    r = validate_expression('__import__("os").system("ls")', _COLS)
    assert not r.ok and "name_not_allowed" in r.reason


def test_gate_rejects_open_builtin():
    r = validate_expression('open("secret.txt").read()', _COLS)
    assert not r.ok and "name_not_allowed: open" in r.reason


def test_gate_rejects_dunder_escape():
    r = validate_expression('df.__class__.__mro__', _COLS)   # 샌드박스 이스케이프 시도
    assert not r.ok and "dunder" in r.reason


def test_gate_rejects_unknown_column():
    r = validate_expression('df["nope"].mean()', _COLS)
    assert not r.ok and "unknown_column: nope" in r.reason


def test_gate_rejects_merge_cross_explosion():
    r = validate_expression('df.merge(df, how="cross")', _COLS)  # N² 폭발
    assert not r.ok and "denied_method: merge" in r.reason


def test_gate_rejects_get_dummies_explosion():
    r = validate_expression('pd.get_dummies(df)', _COLS)
    assert not r.ok and "denied_method" in r.reason


def test_gate_rejects_explode():
    r = validate_expression('df["product_category"].explode()', _COLS)
    assert not r.ok and "denied_method: explode" in r.reason


def test_gate_rejects_file_io():
    r = validate_expression('df.to_csv("out.csv")', _COLS)
    assert not r.ok and "denied_method: to_csv" in r.reason


def test_gate_rejects_read_io():
    r = validate_expression('pd.read_csv("x.csv")', _COLS)
    assert not r.ok and "denied_method: read_csv" in r.reason


def test_gate_rejects_eval_exec_path():
    r = validate_expression('pd.eval("1+1")', _COLS)
    assert not r.ok and "denied_method: eval" in r.reason


def test_gate_rejects_comprehension():
    r = validate_expression('[x for x in range(10)]', _COLS)   # 반복 → 리소스/임의실행
    assert not r.ok and "comprehension" in r.reason


def test_gate_rejects_lambda():
    r = validate_expression('df["order_price"].apply(lambda v: v)', _COLS)
    assert not r.ok and "lambda" in r.reason


def test_validate_request_ok():
    req = CodegenRequest(intent="평균가", target_columns=["order_price"],
                         expression='df["order_price"].mean()', expected_shape="scalar")
    assert validate_request(req, _COLS).ok


def test_validate_request_allows_derived_target_column():
    # target_columns는 실행 안 되는 '선언'이라(오직 expression만 eval) 보안과 무관하고,
    # 파생·중간 컬럼이 섞일 수 있어(codegen의 본질) 하드 거부하지 않는다.
    req = CodegenRequest(intent="x", target_columns=["prop"],   # 원본에 없는 계산 컬럼명
                         expression='df["order_price"].mean()')
    assert validate_request(req, _COLS).ok


def test_gate_allows_derived_column_subscript():
    # gpt-5가 계산 중 만든 파생 컬럼(prop)을 첨자로 꺼내도 통과해야 한다.
    # (원본 df 직접 첨자만 실존 검증 — 중간결과 첨자는 런타임 KeyError로만 잡힌다.)
    expr = 'df.groupby("product_category").agg(prop=("review_score", "mean"))["prop"]'
    r = validate_expression(expr, _COLS)
    assert r.ok, r.reason


def test_gate_still_rejects_wrong_original_column():
    # 완화 후에도 df에 직접 붙은 오타난 원본 컬럼은 여전히 거부한다.
    r = validate_expression('df["prop"].mean()', _COLS)
    assert not r.ok and "unknown_column: prop" in r.reason


def test_gate_rejects_large_alloc():
    # 대형 배열 생성(메모리 폭발)은 거부. np.ones/zeros/arange 등.
    for expr in ('np.ones((10**9,))', 'np.zeros(10**9)', 'np.arange(10**9)',
                 'np.full(10**9, 1)', 'np.tile(df["order_price"].values, 10**6)'):
        r = validate_expression(expr, _COLS)
        assert not r.ok and "denied_method" in r.reason, expr


def test_gate_allows_df_empty_property():
    # 완화 주의: df.empty(빈 DF 체크)는 alloc denylist("empty" 제외)라 여전히 허용.
    r = validate_expression('df.empty', _COLS)
    assert r.ok, r.reason


# ─────────────────────────────
# 차트 셀렉터 (#71 B) — 가설 연계 + 선정 이유 캡션
# ─────────────────────────────
def test_selector_returns_captions_and_uses_hypotheses(monkeypatch, tmp_path):
    import os
    import DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_selector_skill as CS
    # 후보 차트 파일 3개 생성
    paths = []
    for name in ("bar_top_a.png", "dist_b.png", "violin_b.png"):
        p = tmp_path / name
        p.write_bytes(b"png")
        paths.append(str(p))

    seen_prompts = []

    class _FakeSelLLM:
        def invoke(self, prompt):
            seen_prompts.append(prompt)
            return _FakeResp(_json.dumps({
                "remove": ["violin_b.png"],
                "reason": {"violin_b.png": "dist와 중복"},
                "keep_captions": {"bar_top_a.png": "a 순위 — 가설1 근거",
                                  "dist_b.png": "b 분포",
                                  "violin_b.png": "(제거됨)"},
            }))

    monkeypatch.setattr(CS, "_load_llm", lambda: _FakeSelLLM())
    monkeypatch.setattr(CS, "_load_chart_reader_llm", lambda: _FakeSelLLM())
    selected, captions, visual_debug = CS.run_chart_selector_skill(
        chart_paths=paths, user_question="a 상위는?", analysis_results={},
        statistical_metadata={}, hypotheses="[가설 1] a는 그룹별로 다르다")
    names = [os.path.basename(p) for p in selected]
    assert "violin_b.png" not in names and "bar_top_a.png" in names
    assert captions == {"bar_top_a.png": "a 순위 — 가설1 근거", "dist_b.png": "b 분포"}  # 생존 차트만
    assert "[가설 1]" in seen_prompts[0]                 # 가설이 프롬프트에 들어감
    assert visual_debug["dropped"] == [] and visual_debug["check_failures"] == 0


def test_visual_sanity_check_drops_chart_flagged_as_broken(monkeypatch, tmp_path):
    # #166 실측: 배지가 데이터 점을 가리는 순수 렌더링 버그는 통계/이름 기반 판단으로는
    # 못 잡는다 — 최종 후보만 멀티모달로 한 번 더 훑어 걸러낸다.
    import os
    import DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_selector_skill as CS
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    p_ok = tmp_path / "interval_ok.png"
    p_bad = tmp_path / "interval_bad.png"
    p_ok.write_bytes(b"png")
    p_bad.write_bytes(b"png")

    class _FakeVisionLLM:
        def invoke(self, content):
            # HumanMessage(content=[...])의 image_url 안 base64로 어느 파일인지 구분은
            # 안 하고, 호출 순서로 구분한다(두 번째 호출을 "깨짐"으로 응답).
            calls.append(1)
            if len(calls) == 2:
                return _FakeResp('{"ok": false, "issue": "배지가 점을 가림"}')
            return _FakeResp('{"ok": true, "issue": ""}')

    calls: list = []
    monkeypatch.setattr(CS, "_load_chart_reader_llm", lambda: _FakeVisionLLM())
    kept, dropped, failures = CS._visual_sanity_check([str(p_ok), str(p_bad)])
    names = [os.path.basename(p) for p in kept]
    assert names == ["interval_ok.png"]
    assert dropped and dropped[0]["chart"] == "interval_bad.png"
    assert "배지가 점을 가림" in dropped[0]["reason"]
    assert failures == 0


def test_visual_sanity_check_passthrough_when_no_api_key(monkeypatch, tmp_path):
    import DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_selector_skill as CS
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    p = tmp_path / "interval_x.png"
    p.write_bytes(b"png")
    kept, dropped, failures = CS._visual_sanity_check([str(p)])
    assert kept == [str(p)]
    assert dropped == []
    assert failures == 0


def test_visual_sanity_check_counts_failures_separately_from_drops(monkeypatch, tmp_path):
    # Codex 리뷰 P2: 판독기가 고장나면(이미지 읽기/모델호출/파싱 실패) 조용히 통과시키는데,
    # 이게 "진짜 결함 없음"과 구분이 안 되면 검사가 꺼진 걸 아무도 못 알아챈다.
    import DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_selector_skill as CS
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    p = tmp_path / "interval_x.png"
    p.write_bytes(b"png")

    class _FakeBrokenLLM:
        def invoke(self, content):
            raise RuntimeError("모델 호출 실패")

    monkeypatch.setattr(CS, "_load_chart_reader_llm", lambda: _FakeBrokenLLM())
    kept, dropped, failures = CS._visual_sanity_check([str(p)])
    assert kept == [str(p)]   # 실패 시 보수적으로 통과
    assert dropped == []
    assert failures == 1      # 하지만 "실패했다"는 신호는 남는다


def test_backfill_after_visual_drop_fills_from_remaining_candidates(monkeypatch, tmp_path):
    # Codex 리뷰 P1: 시각점검이 드롭해도 대체 후보로 채워지지 않으면 _ensure_preferred_survives의
    # 보호가 마지막 단계에서 깨진다 — 드롭된 자리를 filtered의 다음 우선순위 후보로 채운다.
    import os
    import DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_selector_skill as CS
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    for name in ("segment_profile_a.png", "dist_b.png", "interval_c.png"):
        (tmp_path / name).write_bytes(b"png")
    filtered = [str(tmp_path / n) for n in ("segment_profile_a.png", "dist_b.png", "interval_c.png")]

    class _AlwaysOkLLM:
        def invoke(self, content):
            return _FakeResp('{"ok": true, "issue": ""}')

    monkeypatch.setattr(CS, "_load_chart_reader_llm", lambda: _AlwaysOkLLM())
    # segment_profile_a는 이미 시각점검에서 드롭된 상태를 흉내: dist_b만 살아남았고 자리 1개 빔.
    # interval_c는 filtered에는 있지만 아직 final엔 없는 미사용 후보 — 이게 채워져야 한다.
    final, dropped, failures = CS._backfill_after_visual_drop(
        final=[str(tmp_path / "dist_b.png")], filtered=filtered, target_count=2,
        exclude_names={"segment_profile_a.png"})
    names = {os.path.basename(p) for p in final}
    assert names == {"dist_b.png", "interval_c.png"}
    assert failures == 0


def test_selector_node_exposes_captions(monkeypatch, tmp_path):
    import os
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.chart_selector as N
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    chart = os.path.join(V.OUTPUT_DIR, "bar_top_x.png")
    open(chart, "wb").write(b"png")
    monkeypatch.setattr(N, "run_chart_selector_skill",
                        lambda **kw: ([chart], {"bar_top_x.png": "x 순위 근거"},
                                      {"dropped": [], "check_failures": 0}))
    out = N.chart_selector_node({"user_question": "x?", "hypotheses": "[가설 1] x"})
    assert out["key_charts"] == [chart]
    assert out["key_chart_captions"] == {"bar_top_x.png": "x 순위 근거"}
    assert "cautions" not in out   # 드롭/실패 없으면 caution도 안 붙음
    assert os.path.exists(os.path.join(V.KEY_DIR, "bar_top_x.png"))   # key/ 복사됨


def test_selector_node_adds_caution_when_visual_check_drops_chart(monkeypatch, tmp_path):
    import os
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.chart_selector as N
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    chart = os.path.join(V.OUTPUT_DIR, "bar_top_x.png")
    open(chart, "wb").write(b"png")
    monkeypatch.setattr(N, "run_chart_selector_skill",
                        lambda **kw: ([chart], {"bar_top_x.png": "x 순위 근거"},
                                      {"dropped": [{"chart": "interval_y.png", "reason": "배지가 점 가림"}],
                                       "check_failures": 0}))
    out = N.chart_selector_node({"user_question": "x?", "hypotheses": ""})
    assert out["cautions"][0]["code"] == "CHART_VISUAL_CHECK_ISSUE"
    assert out["cautions"][0]["details"]["dropped"][0]["chart"] == "interval_y.png"


def test_clear_output_dirs_removes_pngs_keeps_dirs(tmp_path):
    # #166 커밋5: outputs/all·key가 런마다 안 지워지고 무한 누적되던 문제.
    import os
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    open(os.path.join(V.OUTPUT_DIR, "old_all.png"), "wb").write(b"png")
    open(os.path.join(V.KEY_DIR, "old_key.png"), "wb").write(b"png")
    V.clear_output_dirs()
    assert os.listdir(V.OUTPUT_DIR) == []
    assert os.listdir(V.KEY_DIR) == []
    assert os.path.isdir(V.OUTPUT_DIR) and os.path.isdir(V.KEY_DIR)  # 폴더 자체는 유지


# ─────────────────────────────
# 차트 semantic 가드 (#71 A) — E2E 실측 잡차트 방지
# ─────────────────────────────
def _guard_df(n=200):
    import numpy as _np
    rng = _np.random.default_rng(0)
    return pd.DataFrame({
        "customer_uid": [f"id{i:05d}" for i in range(n)],          # 고카디널리티(ID급) 범주
        "state": rng.choice(list("ABC"), n),                        # 정상 범주
        "payment_seq": range(1, n + 1),                             # 일련번호(정수, 전부 유니크)
        "is_flag": rng.integers(0, 2, n).astype(float),             # 0/1 플래그
        "value": _np.exp(rng.normal(3, 1.5, n)),                    # 왜도 큰 양수
        "score": rng.normal(50, 10, n),                             # 평범한 수치
    })


def test_numeric_cols_exclude_id_and_flags():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    df = _guard_df()
    base = V._get_numeric_cols(df)                       # ID·일련번호는 항상 제외
    assert "payment_seq" not in base and "value" in base
    strict = V._get_numeric_cols(df, allow_flags=False)  # 분포·산점류: 플래그도 제외
    assert "is_flag" not in strict and "is_flag" in base


def test_distribution_charts_skip_flags_and_use_log(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _guard_df()
    out = V.plot_distributions(df)
    names = [__import__("os").path.basename(p) for p in out["chart_paths"]]
    assert "dist_is_flag.png" not in names               # 0/1 히스토그램 사라짐
    assert "dist_payment_seq.png" not in names           # 일련번호 분포 사라짐
    assert out["stats"]["value"]["skewness"] > 2         # 왜도 큰 컬럼은 log축으로 그려짐(스모크)


def test_ecdf_charts_generate_for_numeric_columns(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _guard_df()
    out = V.plot_ecdfs(df)
    names = [__import__("os").path.basename(p) for p in out["chart_paths"]]
    assert "ecdf_value.png" in names
    assert "ecdf_is_flag.png" not in names
    assert out["stats"]["value"]["p99"] >= out["stats"]["value"]["p90"] >= out["stats"]["value"]["p50"]


def test_distribution_skill_emits_ecdf_family():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.distribution_skill import run_distribution_skill
    df = _guard_df()
    out = run_distribution_skill(df, question_type="distribution")
    assert "ecdfs" in out
    assert out["ecdfs"]["chart_paths"]


def test_select_capped_metrics_prioritizes_and_caps():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.measure_policy import select_capped_metrics
    pool = ["a", "b", "c", "d", "e"]
    priority = [{"metric": "d", "reason": "x"}, {"metric": "z_not_in_pool", "reason": "y"}]
    assert select_capped_metrics(pool, priority, max_n=3) == ["d", "a", "b"]
    assert select_capped_metrics(pool, None, max_n=2) == ["a", "b"]


def test_pick_distribution_chart_types_caps_at_two():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.measure_policy import pick_distribution_chart_types
    skewed = pd.Series(np.exp(np.random.default_rng(0).normal(3, 1.5, 200)))
    normal = pd.Series(np.random.default_rng(0).normal(50, 10, 200))
    assert pick_distribution_chart_types(skewed) == ["dist", "ecdf"]
    assert pick_distribution_chart_types(normal) == ["dist", "box"]


def test_distribution_skill_caps_metrics_and_drops_violin_by_default():
    # #166 커밋4: 지표 전체 × 4종 무조건 생성하던 것을 priority_metrics 우선 + family 상한(6)
    # + 지표당 최대 2종으로 축소. violin은 기본 예산에서 제외된다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.distribution_skill import run_distribution_skill
    df = _guard_df()
    out = run_distribution_skill(df, question_type="distribution",
                                 priority_metrics=[{"metric": "score", "reason": "x"}])
    assert out["violins"]["chart_paths"] == []
    dist_names = {__import__("os").path.basename(p) for p in out["distributions"]["chart_paths"]}
    assert "dist_score.png" in dist_names


def test_key_charts_reject_high_cardinality_key(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _guard_df()
    # 고객ID를 key로 강제해도 저카디널리티 state 로 폴백 (해시 라벨 bar 방지)
    out = V.plot_top_n_barplot(df, key_col="customer_uid", measure_cols=["score"])
    assert out["chart_paths"], "폴백 key(state)로 차트가 나와야 함"
    assert all("customer_uid" not in p for p in out["chart_paths"])
    # 범주 컬럼이 ID뿐이면 스킵
    df_id_only = df[["customer_uid", "score"]]
    out2 = V.plot_top_n_barplot(df_id_only, key_col="customer_uid", measure_cols=["score"])
    assert out2["chart_paths"] == []


def test_scatter_skips_flags_and_tautology(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _guard_df()
    df["value_copy"] = df["value"] * 2                   # 완전상관(동어반복) 쌍
    out = V.plot_scatter_pairs(df, measure_cols=["value", "value_copy", "score", "is_flag"])
    pairs = list(out["stats"].keys())
    assert not any("is_flag" in p for p in pairs)        # 플래그 산점도 없음
    assert "value vs value_copy" not in pairs            # |r|≈1 동어반복 쌍만 정확히 스킵
    assert all(abs(v["pearson_r"]) < 0.98 for v in out["stats"].values())


def test_crosstab_rejects_id_grade_column(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _guard_df()
    out = V.plot_crosstab_heatmap(df, cat_a="customer_uid", cat_b="state")
    assert out["chart_paths"] == [] and "ID급" in out.get("skipped", "")


def test_clustering_excludes_id_and_flag_features():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.clustering_skill import run_clustering_skill
    df = _guard_df()
    out = run_clustering_skill(df, measure_cols=["payment_seq", "is_flag", "value", "score"],
                               key_col="state", question_type="comparison")
    assert not out.get("skip"), out.get("reason")
    used = set(next(iter(out["cluster_centroids"].values())).keys())
    assert used == {"value", "score"}                    # 일련번호·플래그 피처 제외됨


def test_plot_top_n_barplot_top_only_skips_bottom(tmp_path):
    # top_only=True면 하위 잉여차트를 안 그린다(codegen용).
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize
    visualize.set_output_dirs(str(tmp_path))
    d = pd.DataFrame({"cat": list("abcdef"), "value": [6.0, 5, 4, 3, 2, 1]})
    out = visualize.plot_top_n_barplot(d, measure_cols=["value"], top_only=True)
    names = [__import__("os").path.basename(p) for p in out["chart_paths"]]
    assert names and all("bar_bottom" not in n for n in names)


def test_segment_profile_chart_generates_for_flag_columns(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _guard_df()
    df["is_high_value_low_satisfaction"] = ((df["value"] > df["value"].median()) & (df["score"] < df["score"].median())).astype(int)
    out = V.plot_segment_flag_profiles(df, measure_cols=["value", "score"])
    names = [__import__("os").path.basename(p) for p in out["chart_paths"]]
    assert "segment_profile_is_high_value_low_satisfaction.png" in names
    assert out["stats"]["is_high_value_low_satisfaction"]["segment_rate"] > 0


def test_comparison_skill_emits_interval_and_segment_charts():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.comparison_skill import run_comparison_skill
    df = _guard_df()
    df["state"] = np.where(df.index % 3 == 0, "AA", np.where(df.index % 3 == 1, "BB", "CC"))
    df["is_high_value_low_satisfaction"] = ((df["value"] > df["value"].median()) & (df["score"] < df["score"].median())).astype(int)
    out = run_comparison_skill(df, key_col="state", measure_cols=["value", "score"], question_type="comparison")
    assert out["mean_ci"]["chart_paths"]
    assert out["segment_profile"]["chart_paths"]


def test_comparison_skill_caps_metrics_with_priority(tmp_path):
    # #166 커밋4: 지표 6개(cap=4 초과) 중 priority_metrics가 앞에 오고 상한을 넘지 않아야 함.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.comparison_skill import run_comparison_skill
    V.set_output_dirs(str(tmp_path))
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "state": rng.choice(list("ABC"), n),
        "m1": rng.normal(10, 2, n), "m2": rng.normal(20, 3, n), "m3": rng.normal(30, 4, n),
        "m4": rng.normal(40, 5, n), "m5": rng.normal(50, 6, n), "m6": rng.normal(60, 7, n),
    })
    out = run_comparison_skill(
        df, key_col="state", question_type="comparison",
        priority_metrics=[{"metric": "m5", "reason": "x"}, {"metric": "m6", "reason": "y"}])
    heatmap_metrics = set(out["heatmap_matrix"]["stats"].get("top3_per_metric", {}).keys())
    assert {"m5", "m6"}.issubset(heatmap_metrics)
    assert len(heatmap_metrics) <= 4


def test_pick_key_col_rejects_constant_column(tmp_path):
    # 고객 단위 집계본은 anchor_date 등 스냅샷 컬럼이 전 행 동일값(그룹 1개)인 경우가 많다.
    # 카디널리티 상한만 보면 "유효한 범주"로 오판해 그룹 비교 차트가 전부 스킵된다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    df = pd.DataFrame({
        "anchor_date": ["2018-08-29T15:00:37"] * 50,  # 상수(그룹 1개)
        "value": np.linspace(1, 50, 50),
    })
    assert V._pick_key_col(df, None) is None


def test_mean_ci_uses_flag_priority_when_no_categorical_key(tmp_path):
    # 실측 재현: RFM 집계 마트(범주 키 없음, anchor_date는 상수) + 세그먼트 플래그만 존재.
    # is_high_value_low_satisfaction/is_high_value가 이번 결과에서 전부 0(상수)이면
    # 우선순위상 다음 순번인 is_low_satisfaction으로 그룹핑해야 한다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "anchor_date": ["2018-08-29T15:00:37"] * n,          # 상수 스냅샷
        "is_high_value_low_satisfaction": [0] * n,            # 이번 결과에선 전부 0(상수)
        "is_high_value": [0] * n,                              # 이번 결과에선 전부 0(상수)
        "is_low_satisfaction": ([1] * 20) + ([0] * 80),        # 유일하게 변동 있는 플래그
        "avg_review_score": np.concatenate([
            rng.normal(2.4, 0.8, 20), rng.normal(4.6, 0.4, 80),
        ]),
    })
    out = V.plot_mean_ci_comparison(df, key_col=None, measure_cols=["avg_review_score"])
    assert out["chart_paths"], "범주 키가 없어도 세그먼트 플래그로 그룹핑해 차트가 나와야 함"
    groups = {g["is_low_satisfaction"] for g in out["stats"]["avg_review_score"]["top_groups"]}
    assert groups == {0, 1}


def test_mean_ci_badge_uses_safe_corner_not_covering_top_group_point(tmp_path, monkeypatch):
    # 실측 발견(#166): 평균 내림차순 정렬이라 최고 평균 그룹은 항상 우상단(y=0,x=최댓값)에
    # 찍히는데, 통계 배지 기본 위치도 우상단이라 그 점을 매번 가리는 렌더링 버그였다.
    # 배지가 안전한 좌상단으로 호출되는지(=우상단 데이터 포인트를 더 이상 덮지 않는지) 확인한다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    captured = {}
    original = V._add_stat_badge

    def spy(ax, lines, loc="upper right"):
        captured["loc"] = loc
        return original(ax, lines, loc=loc)

    monkeypatch.setattr(V, "_add_stat_badge", spy)
    df = pd.DataFrame({
        "flag": [0] * 80 + [1] * 20,
        "metric": np.concatenate([np.full(80, 1.0), np.full(20, 5.0)]),
    })
    V.plot_mean_ci_comparison(df, key_col="flag", measure_cols=["metric"])
    assert captured.get("loc") == "upper left"


def test_add_stat_badge_positions_cover_all_corners():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize import _BADGE_POSITIONS
    x, y, ha, va = _BADGE_POSITIONS["upper left"]
    assert x < 0.5 and y > 0.5 and ha == "left" and va == "top"
    x, y, ha, va = _BADGE_POSITIONS["upper right"]
    assert x > 0.5 and y > 0.5 and ha == "right" and va == "top"


def test_flag_group_label_translates_binary_value_to_readable_name():
    # 실측 피드백(#166): interval/multiline이 0/1 플래그를 그룹 키로 쓰면 축/범례가 그냥
    # "0","1"로만 보여 어떤 그룹 비교인지 안 읽혔다. 컬럼명 기반 라벨로 바꾼다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize import _flag_group_label
    assert _flag_group_label("is_high_value_low_satisfaction", 1) == "High value low satisfaction"
    assert _flag_group_label("is_high_value_low_satisfaction", 0) == "Rest"
    assert _flag_group_label("is_high_value", 1.0) == "High value"


def test_mean_ci_uses_readable_labels_for_flag_group(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    monkeypatch.setattr(V.plt, "close", lambda *a, **k: None)  # 렌더된 축을 닫기 전에 검사
    df = pd.DataFrame({
        "is_high_value_low_satisfaction": [0] * 80 + [1] * 20,
        "metric": np.concatenate([np.full(80, 4.3), np.full(20, 1.8)]),
    })
    V.plot_mean_ci_comparison(df, key_col=None, measure_cols=["metric"])
    ax = plt.gcf().axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "Rest" in labels
    assert "High value low satisfaction" in labels


def test_multiline_timeseries_ignores_high_cardinality_id_key(tmp_path):
    # 실측 발견(#166): key_col이 customer_unique_id처럼 ID급(고카디널리티)이면 고객마다
    # 선 하나·점 하나짜리 무의미한 차트가 나왔다. 다른 세그먼트 플래그로 폴백해야 한다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _time_gate_df()
    df["customer_unique_id"] = [f"id{i:05d}" for i in range(len(df))]  # 전부 유니크(ID급)
    df["is_high_value"] = (df["value"] > df["value"].median()).astype(int)
    out = V.plot_multiline_timeseries(df, key_col="customer_unique_id", measure_cols=["value"])
    assert out.get("key_col") != "customer_unique_id"
    assert out["chart_paths"], "ID급 키를 걸러내고 대체 키로 차트가 나와야 함"


def test_scatter_pairs_excludes_quartile_derived_columns(tmp_path):
    # 실측 피드백(#166): m_quartile/r_quartile처럼 원본에서 파생된 순서형 버킷 컬럼은
    # scatter로 그리면 몇 줄짜리 계단 모양만 나와 정보량이 낮다 — 후보에서 제외한다.
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    rng = np.random.default_rng(0)
    n = 200
    monetary = rng.exponential(200, n)
    df = pd.DataFrame({
        "monetary_value": monetary,
        "m_quartile": pd.qcut(monetary, 4, labels=False) + 1,
        "avg_review_score": rng.normal(4, 1, n),
    })
    out = V.plot_scatter_pairs(df, measure_cols=["monetary_value", "m_quartile", "avg_review_score"])
    pairs = list(out["stats"].keys())
    assert not any("m_quartile" in p for p in pairs)


# ─────────────────────────────
# 시계열 usable_time_columns 게이팅 (#166 커밋3)
# ─────────────────────────────
def _time_gate_df(n=200):
    import numpy as _np
    rng = _np.random.default_rng(0)
    base = pd.Timestamp("2018-01-01")
    day_offsets = rng.choice(_np.arange(0, 180, 6), size=n)  # 30개 날짜(~6개월)에 반복관측
    return pd.DataFrame({
        "anchor_date": [base] * n,                                    # 상수 스냅샷
        "signup_date": pd.date_range(base, periods=n, freq="D"),       # 행마다 유니크(엔티티 속성)
        "order_date": base + pd.to_timedelta(day_offsets, unit="D"),   # 날짜당 반복관측(진짜 시계열)
        "value": rng.normal(50, 10, n),
    })


def test_usable_time_columns_excludes_snapshot_and_entity_attribute_dates():
    from DATA_Analyst_Assistant_Agent.agents.eda.lib.dtype_utils import usable_time_columns
    df = _time_gate_df()
    result = usable_time_columns(df, ["anchor_date", "signup_date", "order_date"])
    assert result == ["order_date"]


def test_plot_timeseries_skips_constant_snapshot_and_uses_real_date(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _time_gate_df()
    out = V.plot_timeseries(df, measure_cols=["value"])
    assert out["chart_paths"]
    assert out["stats"]["value"]["start"] != out["stats"]["value"]["end"]  # anchor_date였다면 상수라 같았을 것


def test_plot_multiline_timeseries_no_crash_when_key_col_equals_metric(tmp_path):
    # 실측 재현: key_col이 세그먼트 플래그(is_high_value)로 잡히고 measure_cols에도 같은
    # 컬럼이 섞여 들어오면, groupby 결과에 동일 이름 컬럼을 또 넣으려다 크래시했다(#166).
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize as V
    V.set_output_dirs(str(tmp_path))
    df = _time_gate_df()
    df["is_high_value"] = (df["value"] > df["value"].median()).astype(int)
    out = V.plot_multiline_timeseries(   # 크래시하지 않아야 함(과거엔 ValueError)
        df, key_col="is_high_value", measure_cols=["value", "is_high_value"])
    names = [__import__("os").path.basename(p) for p in out["chart_paths"]]
    assert "multiline_value.png" in names
    assert "multiline_is_high_value.png" not in names  # 그룹 키 자신은 지표 후보에서 제외


def test_validate_request_rejects_bad_shape():
    req = CodegenRequest(intent="x", target_columns=["order_price"],
                         expression='df["order_price"].mean()', expected_shape="matrix")
    r = validate_request(req, _COLS)
    assert not r.ok and "invalid_expected_shape" in r.reason


def test_chart_hint_string_null_normalized_to_none():
    # LLM이 '차트 없음'을 문자열 "null"/"none"/""로 뱉어도 None으로 정규화 → 유효한 스칼라 답이
    # invalid_chart_hint로 잘못 거부되지 않아야 한다.
    for bad in ("null", "None", "NULL", "", "  none  "):
        req = CodegenRequest(intent="x", target_columns=["order_price"],
                             expression='df["order_price"].mean()', chart_hint=bad)
        assert req.chart_hint is None, bad
        assert validate_request(req, _COLS).ok


def test_codegen_generate_prompt_forbids_denied_idioms():
    # 게이트가 거부하는 관용구를 생성 프롬프트가 미리 금지하는지(신뢰성 fix, 보안 완화 아님).
    # 실제 LLM이 지키는지는 비결정적이라 테스트하지 않고, 프롬프트 문자열만 deterministic하게 검증.
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen as C
    p = C._generate_prompt("q", ["a", "b"])
    assert "query" in p          # .query() 문자열 eval 금지 명시
    assert "eval" in p
    assert "df.loc" in p         # 필터링 대안 제시


# ─────────────────────────────
# codegen 노드 + 결정론 트리거 (커밋3) — 토큰 0 (FakeLLM)
# ─────────────────────────────
import json as _json
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner import route_after_planner


def _run_codegen(monkeypatch, df, judge_json, generate_json, question="X 계산"):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen as C

    def fake_get_llm(model_env="LLM_MODEL"):
        return _FakeLLM(generate_json if model_env == "CODE_GENERATOR_MODEL" else judge_json)

    monkeypatch.setattr(C, "get_llm", fake_get_llm)
    reset_context()
    set_context(EdaContext(df=df))
    try:
        return C.codegen_node({"user_question": question})["codegen"]
    finally:
        reset_context()


def _run_codegen_seq(monkeypatch, df, judge_json, generate_jsons, question="X 계산"):
    """생성 LLM이 호출마다 generate_jsons를 순서대로 돌려주는 버전(재시도 테스트용).
    judge는 1회만 호출되고, CODE_GENERATOR_MODEL 호출마다 다음 응답을 소비한다."""
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen as C
    seq = list(generate_jsons)

    def fake_get_llm(model_env="LLM_MODEL"):
        if model_env == "CODE_GENERATOR_MODEL":
            return _FakeLLM(seq.pop(0) if seq else "{}")
        return _FakeLLM(judge_json)

    monkeypatch.setattr(C, "get_llm", fake_get_llm)
    reset_context()
    set_context(EdaContext(df=df))
    try:
        return C.codegen_node({"user_question": question})["codegen"]
    finally:
        reset_context()


def test_codegen_success_path(monkeypatch):
    df = pd.DataFrame({"order_price": [10.0, 20.0, 30.0]})
    gen = _json.dumps({"intent": "평균가", "target_columns": ["order_price"],
                       "expression": 'df["order_price"].mean()', "expected_shape": "scalar"})
    out = _run_codegen(monkeypatch, df, '{"computable": true, "reason": "가능"}', gen)
    assert out["status"] == "success"
    assert out["result"] == 20.0
    assert out["expression"] == 'df["order_price"].mean()'      # provenance
    assert "llm_generated" in out["cautions"]
    assert out["telemetry"]["output_shape"] == "scalar"
    assert out["telemetry"]["attempts"] == 1                    # 첫 시도 성공
    assert out["telemetry"]["recovered_by_retry"] is False


def test_codegen_judge_says_not_computable(monkeypatch):
    df = pd.DataFrame({"order_price": [1.0]})
    out = _run_codegen(monkeypatch, df,
                       '{"computable": false, "reason": "외부 데이터 필요"}', "{}")
    assert out["status"] == "out_of_domain" and "외부 데이터" in out["reason"]
    assert out["attempts"] == 0                                 # 생성 시도 안 함(judge에서 컷)


def test_codegen_gate_rejects_dangerous_expression(monkeypatch):
    df = pd.DataFrame({"order_price": [1.0]})
    gen = _json.dumps({"intent": "나쁨", "target_columns": [],
                       "expression": '__import__("os")', "expected_shape": "scalar"})
    out = _run_codegen(monkeypatch, df, '{"computable": true, "reason": "가능"}', gen)
    assert out["status"] == "out_of_domain" and "gate_rejected" in out["reason"]


def test_codegen_bad_generate_json_falls_back(monkeypatch):
    df = pd.DataFrame({"order_price": [1.0]})
    out = _run_codegen(monkeypatch, df, '{"computable": true, "reason": "가능"}', "not json")
    assert out["status"] == "out_of_domain" and "generate_error" in out["reason"]


def test_codegen_retry_recovers_execution_error(monkeypatch):
    # attempt1: 게이트는 통과하나 런타임에서 터지는 코드(.dt on float) → execution_error
    # attempt2: 유효한 코드 → 성공. 재시도로 회복돼야 한다.
    df = pd.DataFrame({"order_price": [10.0, 20.0, 30.0]})
    bad = _json.dumps({"intent": "x", "target_columns": ["order_price"],
                       "expression": 'df["order_price"].dt.month.mean()', "expected_shape": "scalar"})
    good = _json.dumps({"intent": "평균가", "target_columns": ["order_price"],
                        "expression": 'df["order_price"].mean()', "expected_shape": "scalar"})
    out = _run_codegen_seq(monkeypatch, df, '{"computable": true, "reason": "가능"}', [bad, good])
    assert out["status"] == "success" and out["result"] == 20.0
    assert out["telemetry"]["attempts"] == 2
    assert out["telemetry"]["recovered_by_retry"] is True


def test_codegen_retry_exhausted_reports_both_errors(monkeypatch):
    # 두 시도 모두 실패 → retry_failed + errors에 attempt1/attempt2 둘 다 기록.
    df = pd.DataFrame({"order_price": [10.0, 20.0]})
    bad1 = _json.dumps({"intent": "x", "target_columns": ["order_price"],
                        "expression": 'df["order_price"].dt.month.mean()', "expected_shape": "scalar"})
    bad2 = _json.dumps({"intent": "x", "target_columns": ["order_price"],
                        "expression": 'df.query("order_price > 0")', "expected_shape": "frame"})
    out = _run_codegen_seq(monkeypatch, df, '{"computable": true, "reason": "가능"}', [bad1, bad2])
    assert out["status"] == "out_of_domain"
    assert out["reason"].startswith("retry_failed:")
    assert out["attempts"] == 2
    assert len(out["errors"]) == 2
    assert out["errors"][0].startswith("attempt1:") and out["errors"][1].startswith("attempt2:")


def test_codegen_judge_failure_does_not_retry(monkeypatch):
    # judge computable=false는 재시도 대상이 아님(생성 자체를 시작하지 않음).
    df = pd.DataFrame({"order_price": [1.0]})
    out = _run_codegen_seq(monkeypatch, df, '{"computable": false, "reason": "외부 필요"}', ["{}", "{}"])
    assert out["status"] == "out_of_domain" and out["attempts"] == 0
    assert "errors" not in out                              # 시도 자체가 없어 errors 없음


def test_codegen_judge_error_redacts_provider_message(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen as C

    class BoomLLM:
        def invoke(self, _prompt):
            raise RuntimeError(
                "Error code: 402 - This request requires more credits, or fewer max_tokens. "
                "https://openrouter.ai/workspaces/default/keys/secret"
            )

    monkeypatch.setattr(C, "get_llm", lambda model_env="LLM_MODEL": BoomLLM())
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"order_price": [1.0]})))
    try:
        update = C.codegen_node({"user_question": "X"})
    finally:
        reset_context()

    assert update["codegen"]["reason"] == "judge_error: llm_token_budget_exceeded"
    assert "openrouter.ai" not in update["final_summary"].lower()
    assert "secret" not in update["final_summary"].lower()


def test_codegen_success_emits_chart_request(monkeypatch, tmp_path):
    # codegen 성공(차트 있음) → PNG와 별개로 chart_request 계약도 ctx에 발행된다.
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen as C
    from DATA_Analyst_Assistant_Agent.agents.eda._runtime import get_context
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import visualize
    visualize.set_output_dirs(str(tmp_path))                # 차트 파일 오염 방지
    df = pd.DataFrame({"cat": list("aabbcc"), "order_price": [1.0, 2, 3, 4, 5, 6]})
    gen = _json.dumps({"intent": "카테고리별 평균가", "target_columns": ["cat", "order_price"],
                       "expression": 'df.groupby("cat")["order_price"].mean()',
                       "expected_shape": "series", "chart_hint": "bar"})

    def fake_get_llm(model_env="LLM_MODEL"):
        return _FakeLLM(gen if model_env == "CODE_GENERATOR_MODEL"
                        else '{"computable": true, "reason": "ok"}')

    monkeypatch.setattr(C, "get_llm", fake_get_llm)
    reset_context()
    set_context(EdaContext(df=df))
    try:
        out = C.codegen_node({"user_question": "카테고리별 평균가"})["codegen"]
        reqs = list(get_context().chart_requests)
    finally:
        reset_context()
    assert out["status"] == "success"
    assert len(reqs) == 1
    assert reqs[0]["hint"] == "bar" and reqs[0]["stats"]["source"] == "codegen"
    assert reqs[0]["stats"]["expression"] == 'df.groupby("cat")["order_price"].mean()'
    # 컬럼 타입은 df에서 실제 추론(unknown 퉁치기 아님)
    assert reqs[0]["columns"]["cat"]["type"] == "categorical"
    assert reqs[0]["columns"]["order_price"]["type"] == "numeric"


def test_route_done_without_substantive_goes_codegen():
    state = {"next_analysis": "done", "controller_log": [{"choice": "quality"}],
             "quality_result": "품질 점검"}          # quality만 = 실질분석 없음 → codegen
    assert route_after_planner(state) == "codegen"


def test_planner_node_can_select_codegen(monkeypatch):
    # 질문기준 발동: 플래너가 codegen을 고르면 환각가드가 죽이지 않고 그대로 통과시킨다.
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner as P
    df = pd.DataFrame({"order_price": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    monkeypatch.setattr(P, "get_llm",
                        lambda *a, **k: _FakeLLM('{"next": "codegen", "reason": "파생 비율은 도구로 불가"}'))
    reset_context()
    set_context(EdaContext(df=df, measure_cols=["order_price"]))
    try:
        out = P.planner_node({"user_question": "X 이상 비율", "controller_log": [], "round": 0})
    finally:
        reset_context()
    assert out["next_analysis"] == "codegen"


def test_planner_node_rejects_zero_attempt_done_when_feasible(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner as P
    df = pd.DataFrame({"order_price": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    monkeypatch.setattr(P, "get_llm",
                        lambda *a, **k: _FakeLLM('{"next": "done", "reason": "enough"}'))
    reset_context()
    set_context(EdaContext(df=df, measure_cols=["order_price"]))
    try:
        out = P.planner_node({"user_question": "category average", "controller_log": [], "round": 0})
    finally:
        reset_context()
    assert out["next_analysis"] == "distribution"
    assert "planner_done_unreliable" in out["controller_log"][-1]["reason"]


def test_planner_node_rejects_hallucinated_choice(monkeypatch):
    # codegen은 허용하되, 목록 밖 엉뚱한 값은 여전히 done으로 강등한다.
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.planner as P
    df = pd.DataFrame({"order_price": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    monkeypatch.setattr(P, "get_llm",
                        lambda *a, **k: _FakeLLM('{"next": "teleport", "reason": "환각"}'))
    reset_context()
    set_context(EdaContext(df=df, measure_cols=["order_price"]))
    try:
        out = P.planner_node({"user_question": "정상 질문", "controller_log": [], "round": 0})
    finally:
        reset_context()
    assert out["next_analysis"] == "done"


def test_build_app_compiles_with_codegen():
    from DATA_Analyst_Assistant_Agent.agents.eda.graph import build_app
    assert build_app() is not None                  # codegen 노드 포함 그래프 컴파일


# ─────────────────────────────
# codegen 결과 eda_summary 편입 (커밋4) — 토큰 0
# ─────────────────────────────
def test_insight_folds_codegen_success_into_adhoc_analysis(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight as I
    monkeypatch.setattr(I, "get_llm", lambda *a, **k: _FakeLLM("[]"))
    df = pd.DataFrame({"order_price": [10.0, 20.0, 30.0]})
    reset_context()
    set_context(EdaContext(df=df, measure_cols=["order_price"]))
    codegen = {"status": "success", "intent": "평균", "expression": 'df["order_price"].mean()',
               "result": 20.0, "cautions": ["llm_generated"]}
    try:
        update = I.insight_node({"user_question": "q", "question_type": "", "codegen": codegen})
    finally:
        reset_context()
    adhoc = update["statistical_metadata"]["adhoc_analysis"]
    assert adhoc["result"] == 20.0
    assert adhoc["expression"] == 'df["order_price"].mean()'      # provenance 전달됨
    assert "llm_generated" in adhoc["cautions"]


def test_insight_does_not_fold_out_of_domain(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.insight as I
    monkeypatch.setattr(I, "get_llm", lambda *a, **k: _FakeLLM("[]"))
    df = pd.DataFrame({"order_price": [1.0, 2.0]})
    reset_context()
    set_context(EdaContext(df=df, measure_cols=["order_price"]))
    try:
        update = I.insight_node({"user_question": "q", "question_type": "",
                                 "codegen": {"status": "out_of_domain", "reason": "불가"}})
    finally:
        reset_context()
    assert "adhoc_analysis" not in update["statistical_metadata"]  # 실패는 편입 안 함(플래그로만)


# ─────────────────────────────
# _resolve_target: priority_metrics dict 정규화 (커밋5, codegen 경로가 노출한 기존 버그 회귀)
# ─────────────────────────────
def test_resolve_target_handles_priority_metrics_dicts():
    from DATA_Analyst_Assistant_Agent.agents.eda.nodes.hypothesis import _resolve_target
    df = pd.DataFrame({"review_score": [1, 2, 3], "region": ["a", "b", "c"]})
    reset_context()
    set_context(EdaContext(df=df))
    try:  # priority_metrics는 {"metric":...} dict — 컬럼명 정규화돼 매칭 (예전엔 unhashable TypeError)
        target = _resolve_target({"plan_metric": "",
                                  "analysis_plan": {"priority_metrics": [{"metric": "review_score"}]}})
    finally:
        reset_context()
    assert target == "review_score"


def test_resolve_target_empty_when_no_candidate_matches():
    from DATA_Analyst_Assistant_Agent.agents.eda.nodes.hypothesis import _resolve_target
    df = pd.DataFrame({"region": ["a", "b"]})
    reset_context()
    set_context(EdaContext(df=df))
    try:
        target = _resolve_target({"plan_metric": "ghost",
                                  "analysis_plan": {"priority_metrics": [{"metric": "also_ghost"}]}})
    finally:
        reset_context()
    assert target == ""


# ─────────────────────────────
# codegen 게이트 numpy IO 우회 차단 (보안 회귀 — 리뷰 발견 벡터)
# ─────────────────────────────
def test_gate_rejects_numpy_fromfile():
    r = validate_expression('np.fromfile("/etc/passwd")', _COLS)     # 파일 읽기
    assert not r.ok and "denied_method: fromfile" in r.reason


def test_gate_rejects_numpy_loadtxt():
    r = validate_expression('np.loadtxt("/etc/passwd")', _COLS)
    assert not r.ok and "denied_method: loadtxt" in r.reason


def test_gate_rejects_numpy_load_pickle_rce():
    r = validate_expression('np.load("x.npy", allow_pickle=True)', _COLS)  # pickle RCE 경로
    assert not r.ok and "denied_method: load" in r.reason


def test_gate_rejects_numpy_genfromtxt():
    r = validate_expression('np.genfromtxt("x.csv")', _COLS)
    assert not r.ok and "denied_method: genfromtxt" in r.reason


def test_gate_rejects_tofile_write():
    r = validate_expression('df.to_numpy().tofile("/tmp/leak.bin")', _COLS)  # 파일 쓰기
    assert not r.ok and "denied_method: tofile" in r.reason


def test_gate_rejects_numpy_save():
    r = validate_expression('np.save("out.npy", df.to_numpy())', _COLS)
    assert not r.ok and "denied_method: save" in r.reason


def test_gate_rejects_submodule_traversal_to_io():
    r = validate_expression('np.lib.npyio.zipfile_factory', _COLS)   # 서브모듈 순회 우회
    assert not r.ok and "denied_method: lib" in r.reason


def test_gate_still_allows_legit_numpy_compute():
    r = validate_expression('np.sqrt(df["order_price"]).mean()', _COLS)  # 정상 numpy 계산은 허용
    assert r.ok, r.reason


# ─────────────────────────────
# codegen 결과 차트화 (LLM chart_hint → 기존 함수 재사용) — 토큰 0
# ─────────────────────────────
def test_codegen_chart_hint_bar_renders(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize as V
    monkeypatch.setattr(V, "plot_top_n_barplot",
                        lambda *a, **k: {"chart_paths": ["/x/bar_top_success_rate.png"], "stats": {}})
    df = pd.DataFrame({"region": ["a", "a", "b", "b"], "status": ["ok", "fail", "ok", "ok"]})
    gen = _json.dumps({"intent": "지역별 성공률", "target_columns": ["region", "status"],
                       "expression": "df['status'].eq('ok').groupby(df['region']).mean()",
                       "expected_shape": "series", "chart_hint": "bar"})
    out = _run_codegen(monkeypatch, df, '{"computable": true, "reason": "ok"}', gen)
    assert out["status"] == "success"
    assert out["chart"] == "bar_top_success_rate.png"    # LLM이 bar 골라 → 기존 함수 렌더


def test_codegen_chart_hint_null_no_chart(monkeypatch):
    df = pd.DataFrame({"region": ["a", "a", "b", "b"], "status": ["ok", "fail", "ok", "ok"]})
    gen = _json.dumps({"intent": "지역별 성공률", "target_columns": ["region", "status"],
                       "expression": "df['status'].eq('ok').groupby(df['region']).mean()",
                       "expected_shape": "series", "chart_hint": None})
    out = _run_codegen(monkeypatch, df, '{"computable": true, "reason": "ok"}', gen)
    assert out["status"] == "success"
    assert out["chart"] is None                          # LLM이 차트 부적합 판단 → 차트 없음


def test_codegen_scalar_no_chart_even_with_hint(monkeypatch):
    df = pd.DataFrame({"region": ["a", "a", "b"], "status": ["ok", "fail", "ok"]})
    gen = _json.dumps({"intent": "최고 지역", "target_columns": ["region", "status"],
                       "expression": "df['status'].eq('ok').groupby(df['region']).mean().idxmax()",
                       "expected_shape": "scalar", "chart_hint": "bar"})
    out = _run_codegen(monkeypatch, df, '{"computable": true, "reason": "ok"}', gen)
    assert out["status"] == "success"
    assert out["chart"] is None                          # 스칼라 → 그릴 범주 축 없음(힌트 있어도)


def test_bar_value_col_filters_bool_flag():
    # 값 컬럼 선택이 bool 플래그(is_top)를 제외하고 실제 값(0/1 비율 포함)을 고르는가
    from DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen import _first_value_col
    df = pd.DataFrame({"region": ["a", "b"], "success_rate": [1.0, 0.0], "is_top": [True, False]})
    assert _first_value_col(df) == "success_rate"        # 0/1 비율값은 유지, bool 플래그만 제외


def test_gate_rejects_invalid_chart_hint():
    req = CodegenRequest(intent="x", target_columns=["order_price"],
                         expression='df["order_price"].mean()', chart_hint="pie")
    r = validate_request(req, _COLS)
    assert not r.ok and "invalid_chart_hint" in r.reason


# ─────────────────────────────
# out_of_domain short-circuit (도메인 밖이면 insight/hypothesis 안 돌고 정직 종료)
# ─────────────────────────────
def test_route_after_codegen_out_of_domain_ends():
    from DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen import route_after_codegen
    # 실질 결과 전혀 없음(진짜 도메인 밖) → 종료
    assert route_after_codegen({"codegen": {"status": "out_of_domain"}}) == "end"
    assert route_after_codegen({"codegen": {"status": "success"}}) == "insight"


def test_route_after_codegen_out_of_domain_with_tool_output_goes_insight():
    # 플래너가 codegen을 오버픽했지만 실패(out_of_domain) — 도구가 이미 실질 결과를 냈으면
    # 그 분석을 버리지 않고 insight로. (over-fire 피해 방지)
    from DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen import route_after_codegen
    state = {"codegen": {"status": "out_of_domain"},
             "controller_log": [{"choice": "comparison"}, {"choice": "codegen"}],
             "comparison_result": "카테고리별 평균 비교 결과"}
    assert route_after_codegen(state) == "insight"


def test_out_of_domain_is_honest_not_mixed_summary(monkeypatch):
    import DATA_Analyst_Assistant_Agent.agents.eda.nodes.codegen as C
    monkeypatch.setattr(C, "get_llm",
                        lambda model_env="LLM_MODEL": _FakeLLM('{"computable": false, "reason": "외부 데이터 필요"}'))
    reset_context()
    set_context(EdaContext(df=pd.DataFrame({"region": ["a", "b"]})))
    try:
        update = C.codegen_node({"user_question": "경쟁사 점유율 반영해서 평가해줘"})
    finally:
        reset_context()
    assert update["codegen"]["status"] == "out_of_domain"
    assert "답할 수 없습니다" in update["final_summary"]      # 정직한 요약
    assert "distribution" not in update["statistical_metadata"]  # 일반 통계 안 섞임
    assert any(c["code"] == "OUT_OF_DOMAIN" for c in update["cautions"])          # top-level
    assert any(c["code"] == "OUT_OF_DOMAIN"
               for c in update["statistical_metadata"].get("cautions", []))       # stat_metadata에도
