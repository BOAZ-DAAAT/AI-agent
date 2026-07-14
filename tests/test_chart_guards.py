"""chart_guards 테스트 — 순수 코드(LLM/파일시스템 접근 0, 토큰 0).

검증 대상(사용자 지정):
  1) 4개 가드 정탐 (catdist 단일/지배 · 수치 상수 · 순위 막대 평탄 · heatmap 전부 평탄)
  2) 경계값 바로 위/아래 (top1_share 0.95, unique_count 1↔2, group_max==group_min)
  3) 파일명 매칭 실패 시 유지 (신호 없으면 컷 안 함, 보수적)
  4) 정상 차트가 섞여 있어도 그것은 안 잘림
  5) (배선) 두 가드로 전부 걸러지면 run_chart_selector_skill 이 LLM 없이 조기 종료
"""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_guards import drop_degenerate_charts


def _kept_names(paths):
    import os
    return sorted(os.path.basename(p) for p in paths)


def _dropped_names(meta):
    return sorted(m["chart"] for m in meta)


# ── 1) 4개 가드 정탐 ──────────────────────────────────────────────────────────
def test_catdist_single_category_dropped():
    stat = {"distribution": {"order_status": {"type": "categorical", "unique_count": 1, "top1_share": 1.0}}}
    kept, dropped = drop_degenerate_charts(["catdist_order_status.png"], stat)
    assert kept == []
    assert dropped[0]["chart"] == "catdist_order_status.png" and "단일 범주" in dropped[0]["reason"]


def test_catdist_dominant_category_dropped():
    stat = {"distribution": {"flag": {"type": "categorical", "unique_count": 3, "top1_share": 0.97}}}
    kept, dropped = drop_degenerate_charts(["catdist_flag.png"], stat)
    assert kept == [] and "지배" in dropped[0]["reason"]


def test_numeric_constant_dropped():
    stat = {"distribution": {"freight": {"type": "numeric", "unique_count": 1, "std": 0.0}}}
    for name in ("dist_freight.png", "box_freight.png", "violin_freight.png"):
        kept, dropped = drop_degenerate_charts([name], stat)
        assert kept == [], f"{name} should be dropped"
        assert "상수" in dropped[0]["reason"]


def test_ranking_bar_flat_dropped():
    stat = {"group_comparison": {"score": {"n_groups": 5, "group_max": 4.1, "group_min": 4.1, "group_std": 0.0}}}
    for name in ("bar_top_score.png", "bar_bottom_score.png"):
        kept, dropped = drop_degenerate_charts([name], stat)
        assert kept == [] and "평탄" in dropped[0]["reason"]


def test_heatmap_all_metrics_flat_dropped():
    stat = {"group_comparison": {
        "a": {"n_groups": 4, "group_max": 1.0, "group_min": 1.0, "group_std": 0.0},
        "b": {"n_groups": 4, "group_max": 2.0, "group_min": 2.0, "group_std": 0.0},
    }}
    kept, dropped = drop_degenerate_charts(["heatmap_matrix.png"], stat)
    assert kept == [] and "평탄" in dropped[0]["reason"]


def test_heatmap_kept_when_any_metric_varies():
    stat = {"group_comparison": {
        "a": {"n_groups": 4, "group_max": 1.0, "group_min": 1.0, "group_std": 0.0},   # 평탄
        "b": {"n_groups": 4, "group_max": 9.0, "group_min": 2.0, "group_std": 3.0},   # 변동
    }}
    kept, dropped = drop_degenerate_charts(["heatmap_matrix.png"], stat)
    assert _kept_names(kept) == ["heatmap_matrix.png"] and dropped == []


# ── 2) 경계값 바로 위/아래 ────────────────────────────────────────────────────
def test_boundary_top1_share():
    at = {"distribution": {"c": {"type": "categorical", "unique_count": 4, "top1_share": 0.95}}}   # 딱 0.95 → 컷
    below = {"distribution": {"c": {"type": "categorical", "unique_count": 4, "top1_share": 0.94}}} # 0.94 → 유지
    assert drop_degenerate_charts(["catdist_c.png"], at)[0] == []
    assert _kept_names(drop_degenerate_charts(["catdist_c.png"], below)[0]) == ["catdist_c.png"]


def test_boundary_numeric_unique_and_std():
    drop = {"distribution": {"v": {"type": "numeric", "unique_count": 1, "std": 0.0}}}       # unique 1 → 컷
    keep = {"distribution": {"v": {"type": "numeric", "unique_count": 2, "std": 0.5}}}       # unique 2, std>0 → 유지
    assert drop_degenerate_charts(["dist_v.png"], drop)[0] == []
    assert _kept_names(drop_degenerate_charts(["dist_v.png"], keep)[0]) == ["dist_v.png"]


def test_boundary_ranking_flat_vs_varied():
    flat = {"group_comparison": {"m": {"n_groups": 3, "group_max": 5.0, "group_min": 5.0, "group_std": 0.0}}}
    varied = {"group_comparison": {"m": {"n_groups": 3, "group_max": 5.0, "group_min": 1.0, "group_std": 1.7}}}
    assert drop_degenerate_charts(["bar_top_m.png"], flat)[0] == []
    assert _kept_names(drop_degenerate_charts(["bar_top_m.png"], varied)[0]) == ["bar_top_m.png"]


def test_single_group_ranking_dropped():
    stat = {"group_comparison": {"m": {"n_groups": 1, "group_max": 5.0, "group_min": 5.0}}}
    assert drop_degenerate_charts(["bar_top_m.png"], stat)[0] == []


# ── 3) 매칭 실패 시 유지 (보수적) ────────────────────────────────────────────
def test_unmatched_filename_kept():
    # distribution 에 해당 컬럼 신호가 없으면 컷하지 않는다
    stat = {"distribution": {"other_col": {"type": "categorical", "unique_count": 1}}}
    kept, dropped = drop_degenerate_charts(["catdist_missing_col.png"], stat)
    assert _kept_names(kept) == ["catdist_missing_col.png"] and dropped == []


def test_empty_stat_keeps_all():
    paths = ["catdist_x.png", "dist_y.png", "bar_top_z.png", "heatmap_matrix.png"]
    for stat in ({}, None, {"distribution": {}, "group_comparison": {}}):
        kept, dropped = drop_degenerate_charts(list(paths), stat)
        assert _kept_names(kept) == _kept_names(paths) and dropped == []


def test_non_guard_charts_kept():
    # scatter/radar/correlation_heatmap/grouped_bar 등은 가드 대상이 아님 → 항상 유지
    paths = ["scatter_a_vs_b.png", "radar_top_categories.png", "correlation_heatmap.png",
             "grouped_bar_top_categories.png", "cluster_profile.png"]
    kept, dropped = drop_degenerate_charts(list(paths), {"distribution": {}, "group_comparison": {}})
    assert _kept_names(kept) == _kept_names(paths) and dropped == []


# ── 4) 정상 섞여 있어도 정상은 안 잘림 + 전부 퇴화면 빈 리스트 ──────────────────
def test_mixed_only_degenerate_cut():
    paths = ["catdist_status.png", "catdist_state.png", "dist_freight.png", "dist_price.png", "scatter_a_vs_b.png"]
    stat = {
        "distribution": {
            "status": {"type": "categorical", "unique_count": 1, "top1_share": 1.0},   # 컷
            "state": {"type": "categorical", "unique_count": 27, "top1_share": 0.4},    # 유지
            "freight": {"type": "numeric", "unique_count": 1, "std": 0.0},              # 컷
            "price": {"type": "numeric", "unique_count": 500, "std": 45.2},             # 유지
        },
    }
    kept, dropped = drop_degenerate_charts(paths, stat)
    assert _kept_names(kept) == ["catdist_state.png", "dist_price.png", "scatter_a_vs_b.png"]
    assert _dropped_names(dropped) == ["catdist_status.png", "dist_freight.png"]


def test_all_degenerate_returns_empty():
    paths = ["catdist_x.png", "dist_y.png"]
    stat = {"distribution": {
        "x": {"type": "categorical", "unique_count": 1},
        "y": {"type": "numeric", "unique_count": 1, "std": 0.0},
    }}
    kept, dropped = drop_degenerate_charts(paths, stat)
    assert kept == [] and _dropped_names(dropped) == ["catdist_x.png", "dist_y.png"]


def test_drop_meta_shape():
    stat = {"distribution": {"x": {"type": "categorical", "unique_count": 1}}}
    _, dropped = drop_degenerate_charts(["catdist_x.png"], stat)
    assert set(dropped[0].keys()) == {"chart", "reason"}


# ── 5) 배선: 전부 퇴화면 run_chart_selector_skill 이 LLM 없이 조기 종료 ────────
def test_selector_early_exit_skips_llm_when_all_degenerate(tmp_path):
    from DATA_Analyst_Assistant_Agent.agents.eda.lib import chart_selector_skill as css

    def _boom():
        raise AssertionError("전부 퇴화면 LLM 을 호출하면 안 된다")
    css._load_llm = _boom   # 호출되면 실패

    paths = []
    for n in ("catdist_x.png", "dist_y.png"):
        p = tmp_path / n
        p.write_bytes(b"")
        paths.append(str(p))

    stat = {"distribution": {
        "x": {"type": "categorical", "unique_count": 1},
        "y": {"type": "numeric", "unique_count": 1, "std": 0.0},
    }}
    final, captions = css.run_chart_selector_skill(paths, "질문", {}, statistical_metadata=stat)
    assert final == [] and captions == {}   # LLM 미호출 + 조기 종료
