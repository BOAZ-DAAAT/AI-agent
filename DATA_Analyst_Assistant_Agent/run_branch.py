"""분기(재분석) 실행기 — 기존 run에서 특정 단계부터 새 지시사항을 반영해 다시 돈다.

전제: 실제 파이프라인을 먼저 한 번 돌려서 daaa_outputs/latest/run_summary.json 에
artifact_id가 남아있어야 한다.
    python -m DATA_Analyst_Assistant_Agent.run "<질문>"

사용:
    python -m DATA_Analyst_Assistant_Agent.run_branch --start-stage eda --instruction "표본이 30 미만인 판매자는 제외하고 다시 분석해줘"
    python -m DATA_Analyst_Assistant_Agent.run_branch --start-stage analysis --instruction "..."

start_stage 이전 단계는 원래 run의 artifact_id를 그대로 재사용하고(재실행 안 함),
start_stage부터 insight까지는 supervisor/branch.py::branch_from()이 직접 순서대로
실행한다(그래프 라우팅 우회 — LLM 라우팅이 단계를 건너뛰는 문제가 있어 의도적으로 안 씀).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from uuid import uuid4

from DATA_Analyst_Assistant_Agent import BackendAdapter
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.branch import BranchStage, branch_from

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent
_DEFAULT_RUN_SUMMARY = _REPO_ROOT / "daaa_outputs" / "latest" / "run_summary.json"
_DEFAULT_OUT = _REPO_ROOT / "daaa_outputs" / "branch"

_STAGE_ORDER: list[BranchStage] = ["sql", "eda", "analysis", "insight"]
_AGENT_BY_STAGE = {"sql": "sql_agent", "eda": "eda_agent", "analysis": "analysis_agent", "insight": "insight"}

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


def main() -> None:
    parser = argparse.ArgumentParser(description="기존 run에서 특정 단계부터 분기(재분석) 실행")
    parser.add_argument("--start-stage", required=True, choices=_STAGE_ORDER, help="분기를 시작할 단계")
    parser.add_argument("--instruction", required=True, help="이번 분기에서 추가로 반영할 지시사항")
    parser.add_argument("--run-summary", default=str(_DEFAULT_RUN_SUMMARY), help="재사용할 run_summary.json 경로")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="결과 저장 디렉터리")
    args = parser.parse_args()

    summary = _load_run_summary(Path(args.run_summary))
    artifact_ids: dict[str, list[str]] = summary.get("artifact_ids", {}) or {}

    start_index = _STAGE_ORDER.index(args.start_stage)
    upstream_stages = _STAGE_ORDER[:start_index]
    upstream_artifact_ids = {
        _AGENT_BY_STAGE[stage]: artifact_ids.get(_AGENT_BY_STAGE[stage], [])
        for stage in upstream_stages
        if artifact_ids.get(_AGENT_BY_STAGE[stage])
    }
    if upstream_stages and not upstream_artifact_ids:
        raise SystemExit(f"run_summary에 {args.start_stage} 이전 단계의 아티팩트가 없습니다.")

    original_question = str(summary.get("query") or "")
    target_table = _parse_target_table(summary.get("generated_sql", ""))
    run_id = str(summary.get("run_id") or f"run_{uuid4().hex}")
    thread_id = str(summary.get("thread_id") or f"thread_{uuid4().hex}")

    adapter = BackendAdapter()  # 부트스트랩 런과 같은 .data_agent 스토어를 읽는다
    runtime = AgentRuntime(adapter)

    print(f"[run_branch] 원본 질문: {original_question[:80]}")
    print(f"[run_branch] 분기 시작 단계: {args.start_stage}")
    print(f"[run_branch] 추가 지시사항: {args.instruction}")
    print(f"[run_branch] 재사용 단계: {list(upstream_artifact_ids.keys()) or '(없음, 처음부터)'}")
    print("[run_branch] 실행 중...\n")

    result = branch_from(
        args.start_stage,
        args.instruction,
        upstream_artifact_ids=upstream_artifact_ids,
        original_question=original_question,
        run_id=run_id,
        thread_id=thread_id,
        runtime=runtime,
        backend_adapter=adapter,
        target_table=target_table,
    )

    if result.failed_agent:
        print(f"[run_branch] 실패: {result.failed_agent} 단계에서 중단됨")
        print(f"[run_branch] 사유: {result.failure_reason}")
        return

    print("[run_branch] 완료. 새로 생성된 아티팩트:")
    for agent, ids in result.artifact_ids.items():
        print(f"  - {agent}: {ids}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "branch_result.json"
    out_path.write_text(
        json.dumps(
            {
                "start_stage": args.start_stage,
                "instruction": args.instruction,
                "upstream_artifact_ids": upstream_artifact_ids,
                "new_artifact_ids": result.artifact_ids,
                "last_node_id": result.last_node_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[run_branch] 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
