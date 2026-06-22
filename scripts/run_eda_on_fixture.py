"""EDA 다변수 차트 로직을 SQL/LLM 없이 fixture로 돌려보는 테스트 하네스 (모드 A: 코드, 토큰 0).

실행:
  PYTHONUTF8=1 .venv/Scripts/python.exe scripts/run_eda_on_fixture.py [fixture_name|all]

fixture(주문단위 등) + 입력계약을 읽어 차트 함수를 직접 호출하고, 차트마다
"그림 N장 / 조건 안 맞아 스킵"을 보고하고 발행되는 chart_requests를 출력한다.
Gemini/OpenRouter 호출 없음.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from DATA_Analyst_Assistant_Agent.agents.eda.fixtures import load_fixture
from DATA_Analyst_Assistant_Agent.agents.eda.tools import chart_requests as CR
from DATA_Analyst_Assistant_Agent.agents.eda.tools import visualize
from DATA_Analyst_Assistant_Agent.agents.eda.tools.reliability import assess_sample_reliability, detect_data_level

FIXTURES = ["f1_order_level", "f2_category_level", "f3_category_month"]


def _clean_output(out_dir: str) -> None:
    """런마다 출력 폴더의 옛 PNG를 비운다(이전 런 차트가 섞이지 않게)."""
    import glob
    for sub in ("all", "key"):
        for p in glob.glob(os.path.join(out_dir, sub, "*.png")):
            try:
                os.remove(p)
            except OSError:
                pass


def run_graph(name: str) -> None:
    """모드 B — fixture를 진짜 EDA LangGraph에 통과시킨다. ⚠️ LLM(OpenRouter) 호출 발생."""
    from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
    from DATA_Analyst_Assistant_Agent.agents.eda.graph import build_app

    df, c = load_fixture(name)
    out_dir = os.path.join(_REPO_ROOT, "daaa_outputs", "fixture_preview", name + "_graph")
    visualize.set_output_dirs(out_dir)
    _clean_output(out_dir)

    target = c.target_candidates[0] if c.target_candidates else None
    reset_context()
    set_context(EdaContext(
        df=df, key_col=c.key_col, measure_cols=c.numeric, time_cols=c.datetime,
        count_col=c.count, target_col=target, question_type=c.question_type))

    print(f"\n========== [GRAPH/LLM] {name} ==========")
    app = build_app()
    try:
        result = app.invoke({
            "user_question": c.question,
            "target_table": "",
            "mart_design": {},
            "question_type": c.question_type,
            "plan_metric": target or "",
            "plan_dimension": c.key_col or "",
            "error_log": [],
        })
    finally:
        reset_context()

    print(f"completed_analyses: {sorted({e.get('choice') for e in result.get('controller_log', []) if e.get('choice')})}")
    print(f"key_charts({len(result.get('key_charts', []))}): {[os.path.basename(p) for p in result.get('key_charts', [])]}")
    print(f"chart_requests: {len(result.get('chart_requests', []))}개")
    print(f"data_level: {result.get('data_level', {}).get('level')} | cautions: {len(result.get('cautions', []))}개")
    print(f"analysis_target: {result.get('analysis_target')}")
    print(f"error_log: {result.get('error_log', [])}")
    print(f"--- final_summary ---\n{result.get('final_summary', '')[:600]}")

    summary_path = _write_run_summary(name, c, result, out_dir)
    eda_summary_path = _write_eda_summary(name, c, df, result, out_dir)
    print(f"output dir: {visualize.OUTPUT_DIR}")
    print(f"결과 요약: {summary_path}")
    print(f"analysis 전달용(eda_summary.json): {eda_summary_path}")


def _write_eda_summary(name: str, contract, df, result: dict, out_dir: str) -> str:
    """analysis가 실제로 받는 아티팩트(eda_summary.json)와 동일 구조로 떨군다.
    agents/eda/agent.py 의 payload(30~45줄)와 동일 키 — 이게 핸드오프 계약."""
    import json as _json
    import os

    payload = {
        "run_id": f"fixture-{name}",
        "source_artifacts": [name],
        "profile": {
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "quality_status": "ok",
        },
        "user_question": contract.question,
        "analysis_plan": result.get("analysis_plan", {}),
        "insight_result": result.get("insight_result", ""),
        "hypotheses": result.get("hypotheses", ""),
        "final_summary": result.get("final_summary", ""),
        "analysis_target": result.get("analysis_target", ""),
        "data_level": result.get("data_level", {}),
        "cautions": result.get("cautions", []),
        "statistical_metadata": result.get("statistical_metadata", {}),
        "key_charts": [os.path.basename(p) for p in result.get("key_charts", [])],
        "error_log": result.get("error_log", []),
    }
    path = os.path.join(out_dir, "eda_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def _write_run_summary(name: str, contract, result: dict, out_dir: str) -> str:
    """mode B 결과(글+차트목록)를 한 곳에 모아 eda_result.md 로 떨군다 (토큰 0)."""
    import glob
    import json as _json

    all_dir = os.path.join(out_dir, "all")
    charts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(all_dir, "*.png")))
    dl = result.get("data_level", {}) or {}
    analyses = sorted({e.get("choice") for e in result.get("controller_log", []) if e.get("choice")})
    caution_lines = [f"- {c}" for c in (result.get("cautions") or [])] or ["- (없음)"]
    chart_lines = [f"- all/{c}" for c in charts] or ["- (없음)"]

    lines = [
        f"# EDA 결과 — {name}", "",
        f"- 질문: {contract.question}",
        f"- grain: {contract.grain_hint}  /  data_level(자가판정): {dl.get('level')} ({dl.get('confidence')})",
        f"- 분석 수행: {', '.join(analyses)}",
        f"- analysis_target: {result.get('analysis_target')}",
        f"- error_log: {result.get('error_log', [])}", "",
        "## 해석 주의사항 (cautions)",
        *caution_lines, "",
        "## 인사이트", result.get("insight_result", "(없음)"), "",
        "## 가설", result.get("hypotheses", "(없음)"), "",
        "## 핸드오프 요약", result.get("final_summary", "(없음)"), "",
        f"## 차트 ({len(charts)}장, all/ 폴더)",
        *chart_lines, "",
        f"## chart_requests 주문서: {len(result.get('chart_requests', []))}개 (chart_requests.json 참고)",
    ]
    path = os.path.join(out_dir, "eda_result.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # 구조화 원본도 함께(디버그/하류용)
    with open(os.path.join(out_dir, "eda_result.json"), "w", encoding="utf-8") as f:
        _json.dump({k: v for k, v in result.items() if k != "statistical_metadata"},
                   f, ensure_ascii=False, indent=2, default=str)
    return path


def _report(title: str, result: dict, requests: list) -> None:
    paths = result.get("chart_paths") or ([result["chart_path"]] if result.get("chart_path") else [])
    if not paths:
        print(f"  [{title}] SKIP — {result.get('skipped', '조건 미충족/데이터 없음')}")
        return
    print(f"  [{title}] 그림 {len(paths)}장: {[os.path.basename(p) for p in paths]}")
    for r in requests:
        print(f"      주문서> [{r['hint']}] {r['intent']}")


def run_one(name: str) -> None:
    df, c = load_fixture(name)
    out_dir = os.path.join(_REPO_ROOT, "daaa_outputs", "fixture_preview", name)
    visualize.set_output_dirs(out_dir)
    _clean_output(out_dir)

    print(f"\n========== {name} ==========")
    print(f"rows={len(df)} | grain='{c.grain_hint}' level={c.level} | key={c.key_col} "
          f"numeric={c.numeric} datetime={c.datetime} target={c.target_candidates}")
    dl = detect_data_level(df, key_col=c.key_col, numeric_cols=c.numeric)
    rel = assess_sample_reliability(df, key_col=c.key_col, count_col=c.count, data_level=dl["level"])
    print(f"자가판정: level={dl['level']} ({dl['confidence']}) | {rel['note']}")

    target = c.target_candidates[0] if c.target_candidates else None
    cat2 = c.categorical[1] if len(c.categorical) > 1 else None

    # 1) 범주별 분포 (category x numeric)
    r = visualize.plot_grouped_box(df, key_col=c.key_col, measure_cols=c.numeric)
    _report("grouped_box (범주별 분포)", r, CR.from_grouped_box(r, c.key_col))

    # 2) target별 분포 (쿼리 정조준)
    r = visualize.plot_distribution_by_target(df, target_col=target, measure_cols=c.numeric)
    _report("dist_by_target (target별 분포)", r, CR.from_distribution_by_target(r))

    # 3) 시간 x 범주 (multiline)
    tcol = c.datetime[0] if c.datetime else None
    r = visualize.plot_multiline_timeseries(df, time_col=tcol, key_col=c.key_col, measure_cols=c.numeric)
    _report("multiline (시간x범주 추세)", r, CR.from_multiline(r))

    # 4) 범주 x 범주 (crosstab)
    r = visualize.plot_crosstab_heatmap(df, cat_a=c.key_col, cat_b=cat2)
    _report("crosstab (범주x범주)", r, CR.from_crosstab(r))

    print(f"output dir: {visualize.OUTPUT_DIR}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    mode = sys.argv[2] if len(sys.argv) > 2 else "code"

    if mode == "graph":  # 모드 B — 진짜 EDA 그래프(LLM)
        run_graph(arg if arg != "all" else "f1_order_level")
        print("\nDONE (graph/LLM)")
        return

    names = FIXTURES if arg == "all" else [arg]
    for n in names:
        run_one(n)
    print("\nDONE (LLM 호출 0)")


if __name__ == "__main__":
    main()
