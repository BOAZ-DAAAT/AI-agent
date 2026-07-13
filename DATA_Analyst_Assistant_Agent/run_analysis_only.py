"""분석 에이전트 단독 실행기 — 실제 파이프라인이 만든 SQL+EDA 아티팩트를 그대로 재사용.

워크플로우 (부트스트랩 1회 → 분석만 반복):
  1) (부트스트랩, 딱 1회) 실제 파이프라인을 한 번 돌린다:
       python -m DATA_Analyst_Assistant_Agent.run "<질문>"
     → .data_agent 스토어에 SQL/EDA 아티팩트(마트·프로필·차트)가 생기고,
       daaa_outputs/latest/run_summary.json 에 그 artifact_id 목록이 남는다.
  2) (반복) 분석 에이전트만 다시 돌린다:
       python -m DATA_Analyst_Assistant_Agent.run_analysis_only
     → run_summary.json 의 원래 artifact_id 를 그대로 물려 분석 에이전트만 실행한다.
       SQL·EDA·supervisor LLM 은 호출하지 않고 분석 LLM 만 쓴다. 아티팩트가 스토어에 그대로
       있으므로 EDA 프로필·차트를 실제 런과 100% 동일하게 읽는다(리매핑 불필요).

데이터 소스:
  - 기본: generated_sql 에서 마트 테이블명을 뽑아 plan.target_table 로 넣는다 → 분석이 DB 에서
    전체 마트를 조회(실제 런과 동일). DB 조회 실패 시 sql_result CSV(미리보기)로 자동 폴백.
  - --no-db: DB 조회 없이 sql_result CSV(미리보기)로만 분석(빠른 확인용).

주의:
  - .data_agent 스토어(로컬, gitignore)에 그 아티팩트가 남아 있어야 한다(부트스트랩한 그 머신).
    팀원은 각자 1) 부트스트랩을 한 번 돌리면 이후 2) 로 무한 반복할 수 있다.
  - 분석 내부 LLM(코드생성·비평·차트리더)은 그대로 호출되므로 API 크레딧이 필요하다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from DATA_Analyst_Assistant_Agent import BackendAdapter
from DATA_Analyst_Assistant_Agent.agents.analysis import AnalysisAgent
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent
_DEFAULT_RUN_SUMMARY = _REPO_ROOT / "daaa_outputs" / "latest" / "run_summary.json"
_DEFAULT_OUT = _REPO_ROOT / "daaa_outputs" / "analysis_only"

# "CREATE TABLE analytics.dm_customer_rfm_reviews AS ..." 에서 schema.table 추출.
_CREATE_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+([A-Za-z_]\w*\.[A-Za-z_]\w*)", re.IGNORECASE)


def _parse_target_table(generated_sql: str) -> str | None:
    match = _CREATE_TABLE_RE.search(generated_sql or "")
    return match.group(1) if match else None


def _load_run_summary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"run_summary.json 이 없습니다: {path}\n"
            '먼저 부트스트랩을 한 번 실행하세요: python -m DATA_Analyst_Assistant_Agent.run "<질문>"'
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _build_state(summary: dict, target_table: str | None):
    """run_summary 의 원래 artifact_id 를 그대로 물린 분석용 state 를 만든다.

    반환: (state, adapter, sql_ids, eda_ids)
    """
    artifact_ids = summary.get("artifact_ids", {}) or {}
    sql_ids = [a for a in (artifact_ids.get("sql_agent") or []) if a]
    eda_ids = [a for a in (artifact_ids.get("eda_agent") or []) if a]
    if not sql_ids and not eda_ids:
        raise SystemExit("run_summary 에 sql_agent/eda_agent 아티팩트가 없습니다.")

    user_query = str(summary.get("query") or "고정 아티팩트 기반 분석")
    route_kind = str(summary.get("route_kind") or "comprehensive")

    adapter = BackendAdapter()  # 부트스트랩 런과 같은 .data_agent 스토어를 읽는다
    run = adapter.create_run()  # 분석 산출물은 새 run 아래 등록(입력은 옛 artifact_id 그대로 읽음)
    plan = AnalysisPlan(goal=user_query, route_kind=route_kind, target_table=target_table or None)
    state = OrchestrationState(
        run_id=run.run_id,
        user_query=user_query,
        goal=user_query,
        route_kind=route_kind,
        plan=plan,
    )
    state.artifact_ids = {"sql_agent": sql_ids, "eda_agent": eda_ids}
    return state, adapter, sql_ids, eda_ids


def _save_outputs(adapter: BackendAdapter, artifact_ids: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for artifact_id in artifact_ids:
        try:
            text = adapter.read_artifact_text(artifact_id)
            filename = getattr(adapter.get_artifact(artifact_id), "filename", None) or f"{artifact_id}.json"
        except Exception:
            continue
        path = out_dir / filename
        path.write_text(text, encoding="utf-8")
        saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_summary 의 SQL+EDA 아티팩트를 재사용해 분석 에이전트만 실행"
    )
    parser.add_argument("--run-summary", default=str(_DEFAULT_RUN_SUMMARY), help="재사용할 run_summary.json 경로")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="결과 저장 디렉터리")
    parser.add_argument("--target-table", default=None, help="마트 테이블 override(기본: generated_sql 에서 자동 추출)")
    parser.add_argument("--no-db", action="store_true", help="DB 마트 조회 없이 sql_result CSV(미리보기)로만 분석")
    args = parser.parse_args()

    summary = _load_run_summary(Path(args.run_summary))
    if args.no_db:
        target_table = None
    else:
        target_table = args.target_table or _parse_target_table(summary.get("generated_sql", ""))

    state, adapter, sql_ids, eda_ids = _build_state(summary, target_table)

    source = f"DB 마트({target_table})" if target_table else "sql_result CSV(미리보기)"
    print(f"[run_analysis_only] 질문: {state.user_query[:80]}")
    print(f"[run_analysis_only] 재사용: sql {len(sql_ids)}개 · eda {len(eda_ids)}개 아티팩트")
    print(f"[run_analysis_only] 데이터: {source}")
    print("[run_analysis_only] 분석 에이전트 실행 중... (SQL/EDA 재사용, 분석 LLM만 호출)")

    # chart_artifact_loader 를 넘겨 실제 런처럼 차트를 바이트로 읽게 한다(chart_reader 는 supervisor
    # 와 동일하게 미지정 → 기본 멀티모달 리더 사용). 차트 아티팩트가 스토어에 있어야 발동한다.
    envelope = AnalysisAgent().run(
        state, AgentRuntime(adapter), chart_artifact_loader=adapter.read_artifact_bytes
    )

    print("\n=== 결과 ===")
    print(f"status : {envelope.status}")
    print(f"summary: {envelope.summary[:400]}")

    saved = _save_outputs(adapter, envelope.artifact_ids(), Path(args.out))
    print("\n=== 저장 ===")
    for path in saved:
        print(f" - {path}")


if __name__ == "__main__":
    main()
