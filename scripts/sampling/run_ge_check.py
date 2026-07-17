"""GE 정합성 검사를 지정한 DB(dataset_name)에 대해 돌리고, DB별로 구분된 파일명으로 저장한다.

기존 GE 로직(run_dataset_integrity_checks / merge_integrity_reports)은 그대로 재사용한다 —
수정하지 않는다. 다만 그 위의 편의함수 generate_and_write_integrity()는 결과를 항상
db_integrity_result.json(고정 경로)에 쓰기 때문에, olist_sampling에 대해 그대로 돌리면
olist(원본)의 결과를 덮어써버린다. 그래서 그 편의함수는 쓰지 않고, 저수준 함수 두 개만
가져와서 파일명을 직접 지정한다.

⚠️ .venv-ge 안에서 실행해야 한다 (GE는 메인 .venv와 의존성이 격리돼 있음, #123 참고).

사용법:
    .venv-ge/Scripts/python scripts/sampling/run_ge_check.py --dataset-name olist
    .venv-ge/Scripts/python scripts/sampling/run_ge_check.py --dataset-name olist_sampling

결과:
    olist            -> agents/sql/data/db_integrity_result.json          (기존 파일, 갱신됨)
    olist_sampling    -> agents/sql/data/db_integrity_result_sampling.json (신규, 원본 안 건드림)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner import (
    merge_integrity_reports,
    run_dataset_integrity_checks,
)
from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir


def _short_ge_root(dataset_name: str) -> Path:
    # GE는 검증 결과를 uncommitted/validations/<suite>/<run>/<ts>/<hash>.json 처럼 깊게 저장한다.
    # 저장소 경로에 한글이 섞여 있어서(대학교/대외활동/...), 기본 위치(sql_metadata_dir 밑)를
    # 쓰면 Windows MAX_PATH(260자)를 넘어 FileNotFoundError로 죽는다(#123, 실측 269자).
    # 드라이브 루트 바로 밑 짧은 경로를 쓰면 회피된다.
    # 주의: Path("C:")는 드라이브 루트가 아니라 "현재 작업 폴더 기준 C드라이브"로 해석된다
    # (Windows drive-relative path). 반드시 "C:\\"처럼 구분자를 붙여야 진짜 루트가 된다.
    drive = os.environ.get("SYSTEMDRIVE", "C:")
    return Path(drive + "\\") / "_ge_runtime" / dataset_name


def generate_and_save_as(dataset_name: str, output_filename: str) -> None:
    result = run_dataset_integrity_checks(
        dataset_name=dataset_name,
        table_scope=None,
        source_version=dataset_name,
        build_docs=False,  # 오프라인 1회 생성: 데이터독 문서는 불필요
        ge_root=_short_ge_root(dataset_name),
    )
    merged = merge_integrity_reports(result["physical_report"], result["semantic_report"])
    out_path = sql_metadata_dir() / output_filename
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장됨: {out_path}")
    print(
        f"  overall_status: {merged['summary']['overall_status']}, "
        f"테이블 수: {merged['summary']['total_tables']}"
    )


def _default_output_filename(dataset_name: str) -> str:
    if dataset_name == "olist":
        return "db_integrity_result.json"
    suffix = dataset_name.removeprefix("olist_").removeprefix("olist") or "sampling"
    return f"db_integrity_result_{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GE 정합성 검사를 돌리고 dataset별로 구분된 파일에 저장한다."
    )
    parser.add_argument("--dataset-name", required=True, help="예: olist 또는 olist_sampling")
    parser.add_argument(
        "--output-filename",
        default=None,
        help="생략하면 dataset_name으로 자동 결정 "
        "(olist -> db_integrity_result.json, olist_sampling -> db_integrity_result_sampling.json)",
    )
    args = parser.parse_args()
    output_filename = args.output_filename or _default_output_filename(args.dataset_name)
    generate_and_save_as(args.dataset_name, output_filename)


if __name__ == "__main__":
    main()
