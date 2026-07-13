"""SQL 에이전트 단독 실행기 — 질문만 주고 SQL 에이전트만 돌린다(EDA·분석 호출 안 함).

SQL 은 파이프라인 맨 앞단이라 재사용할 상류 아티팩트가 없다. 그래서 이 러너는 질문만 받아
SQL 에이전트만 실행하고, EDA/분석/리포트는 부르지 않고 멈춘다. SQL 쿼리 품질을 빠르게 반복
확인/고도화할 때 쓴다(EDA·분석을 기다릴 필요 없음).

사용:
  python -m DATA_Analyst_Assistant_Agent.run_sql_only "<질문>"
  python -m DATA_Analyst_Assistant_Agent.run_sql_only "<질문>" --route simple
  python -m DATA_Analyst_Assistant_Agent.run_sql_only "<질문>" --out <dir>

주의:
  - SQL 에이전트는 DB 에 접속해 마트를 CREATE 할 수 있다(comprehensive 경로). 실제 DB 를 건드린다.
  - SQL 내부 LLM(플래닝·쿼리생성)은 그대로 호출되므로 API 크레딧이 필요하다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from DATA_Analyst_Assistant_Agent import BackendAdapter
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan, OrchestrationState

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent
_DEFAULT_OUT = _REPO_ROOT / "daaa_outputs" / "sql_only"


def _save_outputs(adapter: BackendAdapter, artifact_ids: list[str], out_dir: Path) -> list[Path]:
    """SQL 이 등록한 아티팩트(generated_sql·sql_result·plan 등)를 out_dir 에 파일로 떨군다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for artifact_id in artifact_ids:
        try:
            record = adapter.get_artifact(artifact_id)
            filename = getattr(record, "filename", None) or f"{artifact_id}.txt"
        except Exception:
            continue
        path = out_dir / filename
        try:
            path.write_text(adapter.read_artifact_text(artifact_id), encoding="utf-8")
        except Exception:
            continue
        saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="질문만 주고 SQL 에이전트만 실행(EDA·분석 미호출)")
    parser.add_argument("query", help="분석 질문(자연어)")
    parser.add_argument("--route", default="comprehensive", choices=["comprehensive", "simple"],
                        help="경로: comprehensive(마트 생성) 또는 simple(단순 조회). 기본 comprehensive")
    parser.add_argument("--datasource", default=None, help="datasource_id (기본: default)")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="결과 저장 디렉터리")
    args = parser.parse_args()

    adapter = BackendAdapter()
    run = adapter.create_run()
    plan = AnalysisPlan(goal=args.query, route_kind=args.route)
    state = OrchestrationState(
        run_id=run.run_id,
        user_query=args.query,
        goal=args.query,
        route_kind=args.route,
        datasource_id=args.datasource,
        plan=plan,
    )

    print(f"[run_sql_only] 질문: {args.query[:80]}")
    print(f"[run_sql_only] 경로: {args.route}")
    print("[run_sql_only] SQL 에이전트 실행 중... (EDA/분석 미호출)")

    envelope = SQLAgent().run(state, AgentRuntime(adapter))

    print("\n=== 결과 ===")
    print(f"status : {envelope.status}")
    print(f"summary: {envelope.summary[:400]}")
    if state.plan is not None and state.plan.target_table:
        print(f"생성 마트: {state.plan.target_table}")

    saved = _save_outputs(adapter, envelope.artifact_ids(), Path(args.out))
    print("\n=== 저장 ===")
    for path in saved:
        print(f" - {path}")


if __name__ == "__main__":
    main()
