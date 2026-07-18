"""run-019f7700 실사례: LLM이 remove와 keep_captions에 같은 파일명을 동시에 넣는
자기모순 응답을 내면, 기존 코드는 remove를 그대로 믿어 최종 key_charts가 0개가 됐다.
캡션을 쓴 파일은 remove보다 우선해 보존돼야 한다."""

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.eda.lib import chart_selector_skill as css


def _make_png(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"fake-png-bytes")
    return str(path)


def test_captioned_file_survives_even_if_llm_also_lists_it_in_remove(tmp_path, monkeypatch):
    kept_name = "groupedbox_product_avg_review_score.png"
    dropped_name = "heatmap_matrix.png"
    paths = [
        _make_png(tmp_path, kept_name),
        _make_png(tmp_path, dropped_name),
    ]

    monkeypatch.setattr(css, "_visual_sanity_check", lambda ps: (ps, [], 0))
    monkeypatch.setattr(
        css,
        "_call_llm_remove",
        lambda **kwargs: {
            # 실사례처럼 remove가 keep_captions에 있는 파일까지 통째로 포함한다.
            "remove": [kept_name, dropped_name],
            "reason": {dropped_name: "질문과 무관하다"},
            "keep_captions": {
                kept_name: "가격대별 리뷰점수 분포를 비교하는 박스플롯이다. 가설1의 근거로 골랐고, "
                           "high 그룹이 가장 높다는 걸 확인 가능하다.",
            },
        },
    )

    final, captions, _ = css.run_chart_selector_skill(
        chart_paths=paths,
        user_question="질문",
        analysis_results={},
        statistical_metadata={},
    )

    final_names = {p.split("\\")[-1].split("/")[-1] for p in final}
    assert kept_name in final_names
    assert dropped_name not in final_names
    assert kept_name in captions


def test_no_conflict_case_still_works(tmp_path, monkeypatch):
    kept_name = "dist_x.png"
    dropped_name = "dist_y.png"
    paths = [
        _make_png(tmp_path, kept_name),
        _make_png(tmp_path, dropped_name),
    ]

    monkeypatch.setattr(css, "_visual_sanity_check", lambda ps: (ps, [], 0))
    monkeypatch.setattr(
        css,
        "_call_llm_remove",
        lambda **kwargs: {
            "remove": [dropped_name],
            "reason": {dropped_name: "중복"},
            "keep_captions": {kept_name: "x 분포를 보여준다. 질문 직답 차트라 골랐다. 평균이 중앙값보다 크다는 걸 확인 가능하다."},
        },
    )

    final, captions, _ = css.run_chart_selector_skill(
        chart_paths=paths,
        user_question="질문",
        analysis_results={},
        statistical_metadata={},
    )

    final_names = {p.split("\\")[-1].split("/")[-1] for p in final}
    assert final_names == {kept_name}
    assert kept_name in captions
