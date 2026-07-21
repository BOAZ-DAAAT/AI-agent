"""완료된 파이프라인 노드의 원본은 유지하고 summary artifact만 다시 생성한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from backend.agent_runs.service import NodeSummaryNotFoundError, regenerate_node_summary
from data_agent_backend.config import BackendConfig
from data_agent_backend.services.factory import create_backend_services


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="완료 노드의 summary만 강제로 재생성합니다.")
    parser.add_argument("--run-id", required=True, help="대상 run ID")
    parser.add_argument("--node-id", required=True, help="agent.completed 이벤트의 node ID")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".data_agent"),
        help="Backend 데이터 디렉터리(기본값: .data_agent)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    services = create_backend_services(BackendConfig(base_data_dir=args.data_dir.resolve()))
    try:
        result = regenerate_node_summary(
            services=services,
            run_id=args.run_id,
            node_id=args.node_id,
        )
    except NodeSummaryNotFoundError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps({
        "ok": True,
        "run_id": args.run_id,
        "node_id": result.node_id,
        "agent_name": result.agent_name,
        "summary_artifact_id": result.summary_artifact_id,
        "fallback_used": result.summary.fallback_used,
        "key_finding": result.summary.key_finding,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
