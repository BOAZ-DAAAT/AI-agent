"""노드 서머리 마크다운 샘플 추출기 — 포스터/데모용.

전제: 실제 파이프라인을 먼저 한 번 돌려서 daaa_outputs/latest/run_summary.json 에
4단계(sql_agent/eda_agent/analysis_agent/insight) artifact_id가 남아있어야 한다.
    python -m DATA_Analyst_Assistant_Agent.run "<질문>"

사용:
    python -m DATA_Analyst_Assistant_Agent.run_summary_sample

각 단계마다 generate_node_summary()로 노드 서머리(JSON)를 만들고, 그걸 마크다운으로
렌더링해서 daaa_outputs/summary_sample/<단계>/summary.md 로 저장한다(차트는 같은 폴더의
charts/ 밑에 PNG로 받아둬서 마크다운이 바로 렌더된다).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from DATA_Analyst_Assistant_Agent import BackendAdapter
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.summary.generator import generate_node_summary
from DATA_Analyst_Assistant_Agent.supervisor.summary.markdown import render_node_summary_markdown
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import NodeSummaryResult

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent
_DEFAULT_RUN_SUMMARY = _REPO_ROOT / "daaa_outputs" / "latest" / "run_summary.json"
_DEFAULT_OUT = _REPO_ROOT / "daaa_outputs" / "summary_sample"

_STAGE_LABELS = {
    "sql_agent": "sql",
    "eda_agent": "eda",
    "analysis_agent": "analysis",
    "insight": "insight",
}


def _load_run_summary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"run_summary.json 이 없습니다: {path}\n"
            '먼저 전체 파이프라인을 한 번 실행하세요: python -m DATA_Analyst_Assistant_Agent.run "<질문>"'
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="4단계 노드 서머리를 마크다운 샘플로 추출")
    parser.add_argument("--run-summary", default=str(_DEFAULT_RUN_SUMMARY), help="재사용할 run_summary.json 경로")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="결과 저장 디렉터리")
    args = parser.parse_args()

    summary = _load_run_summary(Path(args.run_summary))
    artifact_ids: dict[str, list[str]] = summary.get("artifact_ids", {}) or {}

    adapter = BackendAdapter()  # 파이프라인 런과 같은 .data_agent 스토어를 읽는다
    runtime = AgentRuntime(adapter)
    out_root = Path(args.out)

    print(f"[run_summary_sample] 질문: {summary.get('query', '')[:80]}")

    for agent_key, stage_name in _STAGE_LABELS.items():
        ids = [a for a in (artifact_ids.get(agent_key) or []) if a]
        if not ids:
            print(f"[run_summary_sample] {agent_key} 아티팩트 없음 — 스킵")
            continue

        print(f"[run_summary_sample] {agent_key} 서머리 생성 중...")
        ref = generate_node_summary(ids, runtime)
        payload = json.loads(adapter.read_artifact_text(ref.artifact_id))
        result = NodeSummaryResult.model_validate(payload)

        stage_dir = out_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        markdown = render_node_summary_markdown(result, runtime, stage_dir)
        (stage_dir / "summary.md").write_text(markdown, encoding="utf-8")
        print(f"[run_summary_sample]  -> {stage_dir / 'summary.md'} (fallback_used={result.fallback_used})")

    print("\n완료. daaa_outputs/summary_sample/<단계>/summary.md 를 마크다운 뷰어로 열어보세요.")


if __name__ == "__main__":
    main()
