from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv

from DATA_Analyst_Assistant_Agent import BackendAdapter, SupervisorAgent
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    OrchestrationState,
    SupervisorInterruptPayload,
    SupervisorRunResult,
    SupervisorTerminalState,
)
from DATA_Analyst_Assistant_Agent.shared.config import integrity_result_filename, sql_metadata_dir


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "daaa_outputs" / "latest"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _normalize_env_aliases() -> None:
    aliases = {
        "GOOGLE_API_KEY": "GEMINI_API_KEY",
    }
    for target, source in aliases.items():
        if not os.getenv(target) and os.getenv(source):
            os.environ[target] = os.getenv(source, "")
    os.environ.setdefault("MYSQL_PASSWORD", os.getenv("DB_PASSWORD", ""))


def _ensure_sql_agent_metadata() -> None:
    data_dir = sql_metadata_dir()
    schema_path = data_dir / "db_schema.json"
    integrity_path = data_dir / integrity_result_filename()
    if schema_path.exists() and integrity_path.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    from sqlalchemy import inspect

    from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine

    engine = get_db_engine()
    if engine is None:
        raise RuntimeError("DB engine creation failed. Check MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USERNAME, MYSQL_PASSWORD in .env.")

    inspector = inspect(engine)
    schema = {}
    for table in inspector.get_table_names():
        schema[table] = {
            "primary_key": inspector.get_pk_constraint(table).get("constrained_columns", []),
            "description": "",
            "columns": [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "description": "",
                }
                for column in inspector.get_columns(table)
            ],
            "foreign_keys": [
                {
                    "referred_table": fk.get("referred_table"),
                    "referred_columns": fk.get("referred_columns", []),
                    "constrained_columns": fk.get("constrained_columns", []),
                }
                for fk in inspector.get_foreign_keys(table)
            ],
        }

    if not schema_path.exists():
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    if not integrity_path.exists():
        integrity_path.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")


def _state_summary(state: OrchestrationState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "thread_id": state.thread_id,
        "datasource_id": state.datasource_id,
        "terminal_state": state.terminal_state.value if state.terminal_state else None,
        "route_kind": state.route_kind,
        "planner_mode": state.planner_mode,
        "current_step": state.current_step,
        "completed_agents": state.completed_agents,
        "remaining_agents": state.remaining_agents,
        "final_answer": state.final_answer,
        "generated_sql": state.generated_sql,
        "artifact_ids": state.artifact_ids,
        "approval_ids": state.approval_ids,
        "mart_id": state.mart_id,
        "error_state": state.error_state,
    }


def _new_thread_id() -> str:
    return f"thread_{uuid4().hex}"


def _shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _resume_command(
    args: argparse.Namespace,
    *,
    thread_id: str,
    resume_kind: str,
    approval_id: str | None = None,
    selection_value: str | None = None,
) -> str:
    parts = [
        sys.executable,
        "-m",
        "DATA_Analyst_Assistant_Agent.run",
        "--thread-id",
        thread_id,
    ]
    if resume_kind == "clarification":
        parts.extend(["--resume-answer", "..."])
    elif resume_kind == "approval":
        parts.append("--approve")
    elif resume_kind in {"analysis_option", "analysis_free_text"}:
        if not approval_id:
            raise ValueError("analysis review resume command에는 approval_id가 필요합니다.")
        parts.extend(["--resume-approval-id", approval_id])
        parts.extend(
            [
                "--resume-option-id" if resume_kind == "analysis_option" else "--resume-free-text",
                selection_value or "...",
            ]
        )
    else:
        raise ValueError(f"지원하지 않는 resume_kind입니다: {resume_kind}")

    if args.show_sql:
        parts.append("--show-sql")
    if args.no_open:
        parts.append("--no-open")
    if args.no_output:
        parts.append("--no-output")
    if args.json:
        parts.append("--json")
    if args.dotenv != ".env":
        parts.extend(["--dotenv", str(args.dotenv)])
    if str(args.output_dir) != str(DEFAULT_OUTPUT_DIR):
        parts.extend(["--output-dir", str(args.output_dir)])
    return _shell_command([str(part) for part in parts])


def _approval_resume_metadata(state: OrchestrationState, args: argparse.Namespace) -> dict[str, Any]:
    if state.terminal_state != SupervisorTerminalState.needs_user_approval or not state.thread_id:
        return {}
    return {
        "resume_payload": {"approved": True},
        "resume_command": _resume_command(args, thread_id=state.thread_id, resume_kind="approval"),
    }


def _should_run_interactively(args: argparse.Namespace) -> bool:
    if (
        args.json
        or args.resume_answer is not None
        or args.resume_approved
        or getattr(args, "resume_option_id", None) is not None
        or getattr(args, "resume_free_text", None) is not None
    ):
        return False
    if args.interactive is not None:
        return bool(args.interactive)
    return bool(sys.stdin.isatty())


def _interrupt_summary(result: SupervisorRunResult, args: argparse.Namespace) -> dict[str, Any]:
    if result.interrupt is None:
        return {"kind": result.kind}
    payload = result.interrupt
    summary = {
        "kind": result.kind,
        "interrupt": payload.model_dump(mode="json"),
    }
    if payload.type == "clarification":
        summary.update(
            {
                "resume_payload": {"answer": "..."},
                "resume_command": _resume_command(
                    args, thread_id=payload.thread_id, resume_kind="clarification"
                ),
            }
        )
        return summary
    review_request = payload.review_request or {}
    option_ids = [str(option.get("id") or "") for option in review_request.get("options", [])]
    option_id = str(review_request.get("recommended_option_id") or "")
    if option_id not in option_ids:
        option_id = next((item for item in option_ids if item), "...")
    summary.update(
        {
            "resume_payload": {
                "approval_id": payload.approval_id,
                "selected_option_id": option_id,
            },
            "resume_command": _resume_command(
                args,
                thread_id=payload.thread_id,
                resume_kind="analysis_option",
                approval_id=payload.approval_id,
                selection_value=option_id,
            ),
        }
    )
    if bool(review_request.get("allow_free_text")):
        summary.update(
            {
                "free_text_resume_payload": {
                    "approval_id": payload.approval_id,
                    "free_text": "...",
                },
                "free_text_resume_command": _resume_command(
                    args,
                    thread_id=payload.thread_id,
                    resume_kind="analysis_free_text",
                    approval_id=payload.approval_id,
                    selection_value="...",
                ),
            }
        )
    return summary


def _print_interrupt_summary(result: SupervisorRunResult, args: argparse.Namespace) -> None:
    if result.interrupt is None:
        print("사용자 입력 대기 상태입니다.")
        return
    payload = result.interrupt
    print("\n=== Human Input Required ===")
    print(f"type:       {payload.type}")
    print(f"status:     {payload.status}")
    print(f"run_id:     {payload.run_id}")
    print(f"thread_id:  {payload.thread_id}")
    print(f"node:       {payload.node}")
    print(f"question:   {payload.question}")
    if payload.type == "analysis_review":
        review_request = payload.review_request or {}
        print(f"approval_id: {payload.approval_id}")
        print("\noptions:")
        for option in review_request.get("options", []):
            print(f"- {option.get('id')}: {option.get('label')}")
            print(f"  method: {option.get('method')}")
            print(f"  assumptions: {option.get('assumptions', [])}")
            print(f"  advantages: {option.get('advantages', [])}")
            print(f"  limitations: {option.get('limitations', [])}")
            print(f"  impact: {option.get('impact')}")
        metadata = _interrupt_summary(result, args)
        print("\nresume payload:")
        print(json.dumps(metadata["resume_payload"], ensure_ascii=False, indent=2))
        print("\nresume command:")
        print(metadata["resume_command"])
        if "free_text_resume_payload" in metadata:
            print("\nfree text resume payload:")
            print(json.dumps(metadata["free_text_resume_payload"], ensure_ascii=False, indent=2))
            print("\nfree text resume command:")
            print(metadata["free_text_resume_command"])
        return
    print("\nresume payload:")
    print(json.dumps({"answer": "..."}, ensure_ascii=False, indent=2))
    print("\nresume command:")
    print(_resume_command(args, thread_id=payload.thread_id, resume_kind="clarification"))


def _artifact_preview(adapter: BackendAdapter, artifact_id: str) -> dict[str, Any]:
    artifact = adapter.get_artifact(artifact_id)
    return {
        "artifact_id": artifact.artifact_id,
        "type": str(artifact.type),
        "uri": artifact.uri,
        "metadata": artifact.metadata,
        "preview": artifact.preview,
        "local_path": str(artifact.local_path) if artifact.local_path else None,
    }


def _copy_artifacts(output_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> None:
    artifact_root = output_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for agent_name, items in artifacts.items():
        agent_dir = artifact_root / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            local_path = item.get("local_path")
            if not local_path:
                continue
            source = Path(local_path)
            if source.exists() and source.is_file():
                destination = agent_dir / f"{item['artifact_id']}_{source.name}"
                shutil.copy2(source, destination)


def _markdown_to_html(markdown: str, *, title: str) -> str:
    blocks: list[str] = []
    in_code = False
    code_lines: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 4)
            text = html.escape(line.lstrip("#").strip())
            blocks.append(f"<h{level}>{text}</h{level}>")
        elif line.startswith(("- ", "* ")):
            text = html.escape(line[2:].strip())
            blocks.append(f"<p class=\"bullet\">{text}</p>")
        else:
            text = html.escape(line)
            text = text.replace("**", "")
            blocks.append(f"<p>{text}</p>")

    if code_lines:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #1f2937; font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif; line-height: 1.65; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 36px 28px 64px; background: #fff; min-height: 100vh; box-shadow: 0 0 0 1px #e5e7eb; }}
    h1 {{ font-size: 28px; margin: 0 0 18px; padding-bottom: 14px; border-bottom: 2px solid #e5e7eb; }}
    h2 {{ font-size: 21px; margin-top: 28px; color: #174ea6; }}
    h3 {{ font-size: 17px; margin-top: 20px; }}
    p {{ margin: 8px 0; }}
    .bullet {{ padding-left: 16px; position: relative; }}
    .bullet::before {{ content: "-"; position: absolute; left: 0; color: #174ea6; }}
    pre {{ overflow-x: auto; background: #111827; color: #f9fafb; border-radius: 8px; padding: 16px; font-size: 13px; }}
    code {{ font-family: Consolas, "Courier New", monospace; }}
    a {{ color: #174ea6; }}
  </style>
</head>
<body>
  <main>
    {chr(10).join(blocks)}
  </main>
</body>
</html>
"""


def _write_report_html(output_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> dict[str, Path]:
    report_markdown = ""
    for item in artifacts.get("report_agent", []):
        if item.get("metadata", {}).get("kind") == "final_report" and item.get("local_path"):
            source = Path(item["local_path"])
            if source.exists():
                report_markdown = source.read_text(encoding="utf-8")
                break

    if not report_markdown:
        report_markdown = "# Data Analyst Assistant Report\n\n리포트 산출물이 생성되지 않았습니다."

    report_md_path = output_dir / "final_report.md"
    report_html_path = output_dir / "final_report.html"
    index_path = output_dir / "index.html"

    report_md_path.write_text(report_markdown, encoding="utf-8")
    report_html_path.write_text(_markdown_to_html(report_markdown, title="Final Report"), encoding="utf-8")
    index_path.write_text(_build_index_html(report_markdown, artifacts), encoding="utf-8")
    return {"final_report_md": report_md_path, "final_report_html": report_html_path, "index_html": index_path}


def _write_sql_result_csv(output_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> Path | None:
    sql_item = _first_artifact(artifacts, "sql_agent", "sql_result")
    if not sql_item or not sql_item.get("local_path"):
        return None
    source = Path(sql_item["local_path"])
    if not source.exists():
        return None
    destination = output_dir / "sql_result.csv"
    shutil.copy2(source, destination)
    return destination


def _first_artifact(artifacts: dict[str, list[dict[str, Any]]], agent_name: str, kind: str | None = None) -> dict[str, Any] | None:
    for item in artifacts.get(agent_name, []):
        if kind is None or item.get("metadata", {}).get("kind") == kind:
            return item
    return None


def _artifact_text(item: dict[str, Any] | None) -> str:
    if not item or not item.get("local_path"):
        return ""
    path = Path(item["local_path"])
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _artifact_json(item: dict[str, Any] | None) -> dict[str, Any]:
    text = _artifact_text(item)
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_csv_artifact(item: dict[str, Any] | None, *, limit: int = 50) -> tuple[list[str], list[dict[str, str]]]:
    text = _artifact_text(item)
    if not text:
        return [], []
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for idx, row in enumerate(reader):
        if idx >= limit:
            break
        rows.append({key: str(value) for key, value in row.items()})
    return list(reader.fieldnames or []), rows


def _table_html(columns: list[str], rows: list[dict[str, str]]) -> str:
    if not columns or not rows:
        return "<p>표시할 SQL 결과 행이 없습니다.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(row.get(column, ''))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _coerce_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _chart_svg(columns: list[str], rows: list[dict[str, str]], x_field: str, y_field: str) -> str:
    if not rows or x_field not in columns or y_field not in columns:
        return "<p>차트로 렌더링할 수 있는 x/y 컬럼을 찾지 못했습니다.</p>"

    points: list[tuple[str, float]] = []
    for row in rows:
        value = _coerce_float(row.get(y_field))
        if value is not None:
            points.append((str(row.get(x_field, "")), value))
    if len(points) < 2:
        return "<p>차트를 그리기 위한 숫자 데이터가 충분하지 않습니다.</p>"

    width, height = 980, 360
    pad_left, pad_right, pad_top, pad_bottom = 72, 24, 28, 62
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    y_values = [value for _, value in points]
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min, y_max = 0, y_max or 1

    coords = []
    for idx, (label, value) in enumerate(points):
        x = pad_left + (plot_w * idx / max(len(points) - 1, 1))
        y = pad_top + plot_h - ((value - y_min) / (y_max - y_min) * plot_h)
        coords.append((x, y, label, value))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5"><title>{html.escape(label)}: {value:,.2f}</title></circle>'
        for x, y, label, value in coords
    )
    tick_indexes = sorted(set([0, len(coords) // 4, len(coords) // 2, len(coords) * 3 // 4, len(coords) - 1]))
    x_ticks = "\n".join(
        f'<text x="{coords[idx][0]:.1f}" y="{height - 22}" text-anchor="middle">{html.escape(coords[idx][2])}</text>'
        for idx in tick_indexes
    )
    y_ticks = []
    for step in range(5):
        value = y_min + ((y_max - y_min) * step / 4)
        y = pad_top + plot_h - (plot_h * step / 4)
        y_ticks.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" class="grid"/>')
        y_ticks.append(f'<text x="{pad_left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:,.0f}</text>')

    return f"""
      <div class="chart-wrap">
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(y_field)} by {html.escape(x_field)}">
          <rect width="{width}" height="{height}" fill="#ffffff"/>
          {''.join(y_ticks)}
          <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis"/>
          <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis"/>
          <polyline points="{polyline}" fill="none" class="series"/>
          {circles}
          {x_ticks}
          <text x="{pad_left}" y="18" class="axis-label">{html.escape(y_field)}</text>
        </svg>
      </div>
"""


def _answer_html(answer: str) -> str:
    if not answer:
        return "<p>최종 답변이 생성되지 않았습니다.</p>"
    blocks = []
    for raw in answer.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            blocks.append(f"<p class=\"bullet\">{html.escape(line[2:].strip()).replace('**', '')}</p>")
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 4)
            blocks.append(f"<h{level}>{html.escape(line.lstrip('#').strip())}</h{level}>")
        else:
            blocks.append(f"<p>{html.escape(line).replace('**', '')}</p>")
    return "\n".join(blocks) or "<p>최종 답변이 생성되지 않았습니다.</p>"


def _list_html(items: list[Any], fallback: str) -> str:
    if not items:
        return f"<p>{html.escape(fallback)}</p>"
    return "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in items) + "</ul>"


def _artifact_index(artifacts: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[str, dict[str, Any]]]:
    """artifact_id → (agent_name, item). eda/insight의 key_charts/charts가 참조하는
    차트 아티팩트id로 실제 파일을 찾기 위한 용도(어느 agent 버킷에 있는지 몰라도 찾게)."""
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for agent_name, items in artifacts.items():
        for item in items:
            artifact_id = str(item.get("artifact_id") or "")
            if artifact_id:
                index[artifact_id] = (agent_name, item)
    return index


def _chart_gallery_html(
    chart_refs: list[Any],
    artifact_index: dict[str, tuple[str, dict[str, Any]]],
    *,
    caption_key: str,
) -> str:
    figures = []
    for ref in chart_refs or []:
        if not isinstance(ref, dict):
            continue
        artifact_id = str(ref.get("artifact_id") or "")
        agent_name, item = artifact_index.get(artifact_id, (None, None))
        if not item or not item.get("local_path"):
            continue
        image_path = f"artifacts/{agent_name}/{artifact_id}_{Path(item['local_path']).name}"
        caption = html.escape(str(ref.get(caption_key) or ""))
        figures.append(
            f'<figure><img src="{html.escape(image_path)}" alt="{caption}" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    if not figures:
        return ""
    return f'<div class="chart-gallery">{"".join(figures)}</div>'


def _build_index_html(report_markdown: str, artifacts: dict[str, list[dict[str, Any]]]) -> str:
    sql_item = _first_artifact(artifacts, "sql_agent", "sql_result")
    sql_plan = _artifact_json(_first_artifact(artifacts, "sql_agent", "sql_lang_graph_result"))
    eda = _artifact_json(_first_artifact(artifacts, "eda_agent", "eda_summary"))
    analysis = _artifact_json(_first_artifact(artifacts, "analysis_agent", "analysis_result"))
    insight = _artifact_json(_first_artifact(artifacts, "insight", "insight_payload"))
    columns, rows = _read_csv_artifact(sql_item)
    artifact_index = _artifact_index(artifacts)

    # 인사이트가 있으면(파이프라인이 정상 종료됐다면 항상 있음) 그게 진짜 최종 답변이다.
    # SQL의 final_answer는 인사이트가 없을 때(예: 실패/중단된 실행)에만 보조로 쓴다.
    final_answer = str(insight.get("answer") or "") or str(sql_plan.get("final_answer") or "")

    artifact_rows = []
    for agent_name, items in artifacts.items():
        for item in items:
            artifact_rows.append(
                "<tr>"
                f"<td>{html.escape(agent_name)}</td>"
                f"<td>{html.escape(str(item.get('artifact_id', '')))}</td>"
                f"<td>{html.escape(str(item.get('type', '')))}</td>"
                f"<td>{html.escape(str(item.get('local_path') or ''))}</td>"
                "</tr>"
            )

    eda_final_summary = str(eda.get("final_summary") or "")
    eda_hypotheses = str(eda.get("hypotheses") or "")
    eda_cautions = eda.get("cautions") or []

    # 필드명 확인: analysis_result의 실제 요약 필드는 method_summary가 아니라
    # executive_summary다(agents/analysis/agent.py::_public_result_payload).
    key_findings = analysis.get("key_findings") or []
    limitations = analysis.get("limitations") or []
    method = str(analysis.get("executive_summary") or "")

    key_insights = insight.get("key_insights") or []
    action_plan = insight.get("action_plan") or []
    insight_limitations = insight.get("limitations") or []

    # 차트는 EDA/인사이트가 실제로 등록한 이미지를 그대로 보여준다(하드코딩 placeholder 없음).
    chart_gallery = _chart_gallery_html(eda.get("key_charts") or [], artifact_index, caption_key="caption")
    chart_gallery += _chart_gallery_html(insight.get("charts") or [], artifact_index, caption_key="title")
    if not chart_gallery:
        # 등록된 차트 아티팩트가 하나도 없으면, SQL 결과 표에서라도 간단한 추세 차트를 시도한다.
        numeric_columns = [c for c in columns if any(_coerce_float(row.get(c)) is not None for row in rows)]
        if columns and numeric_columns:
            chart_gallery = _chart_svg(columns, rows, columns[0], numeric_columns[0])
        else:
            chart_gallery = "<p>표시할 차트 아티팩트가 없습니다.</p>"

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data Analysis Result</title>
  <style>
    body {{ font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif; margin: 0; color: #1f2933; line-height: 1.55; background: #f6f7f9; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 72px; background: #ffffff; min-height: 100vh; box-shadow: 0 0 0 1px #e5e7eb; }}
    section {{ margin-top: 28px; padding-top: 10px; }}
    h1 {{ margin-bottom: 10px; }}
    h2 {{ color: #174ea6; margin: 18px 0 10px; }}
    h3 {{ margin: 16px 0 8px; }}
    code, pre {{ font-family: Consolas, "Courier New", monospace; }}
    pre {{ background: #f3f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f9fafb; }}
    .answer {{ background: #f8fafc; border-left: 4px solid #174ea6; padding: 12px 16px; border-radius: 6px; }}
    .table-wrap {{ overflow-x: auto; }}
    .badge {{ display: inline-block; background: #e8f0fe; color: #174ea6; border-radius: 999px; padding: 3px 10px; font-size: 12px; }}
    .bullet {{ margin-left: 12px; }}
    .chart-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; margin-top: 14px; padding: 10px; }}
    .chart-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 14px; }}
    .chart-gallery figure {{ margin: 0; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #fff; }}
    .chart-gallery img {{ width: 100%; height: auto; border-radius: 4px; }}
    .chart-gallery figcaption {{ font-size: 12px; color: #6b7280; margin-top: 6px; }}
    svg {{ width: 100%; min-width: 680px; height: auto; }}
    .axis {{ stroke: #9ca3af; stroke-width: 1; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .series {{ stroke: #174ea6; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    circle {{ fill: #174ea6; }}
    text {{ fill: #4b5563; font-size: 12px; }}
    .axis-label {{ fill: #111827; font-weight: 700; }}
    details {{ margin-top: 24px; }}
    summary {{ cursor: pointer; font-weight: 700; color: #174ea6; }}
    a {{ color: #174ea6; }}
  </style>
</head>
<body>
  <main>
    <h1>Data Analysis Result</h1>
    <p><span class="badge">completed</span></p>

    <section>
      <h2>Answer</h2>
      <div class="answer">{_answer_html(final_answer)}</div>
    </section>

    <section>
      <h2>SQL Result Table</h2>
      <p>요청한 분석 결과는 아래 표에서 바로 확인할 수 있습니다. 원본 데이터는 <a href="sql_result.csv">sql_result.csv</a> 파일로도 저장되어 있습니다.</p>
      {_table_html(columns, rows)}
    </section>

    <section>
      <h2>EDA Findings</h2>
      <p>{html.escape(eda_final_summary or "EDA 요약이 생성되지 않았습니다(이 경로에서는 EDA가 생략됐을 수 있습니다).")}</p>
      <h3>Hypotheses</h3>
      <p>{html.escape(eda_hypotheses) if eda_hypotheses else "표시할 가설이 없습니다."}</p>
      <h3>Cautions</h3>
      {_list_html(eda_cautions, "표시할 주의 사항이 없습니다.")}
    </section>

    <section>
      <h2>Analysis Insights</h2>
      <p>{html.escape(str(method or "분석 요약이 생성되지 않았습니다."))}</p>
      <h3>Key Findings</h3>
      {_list_html(key_findings, "표시할 핵심 발견 사항이 없습니다.")}
      <h3>Limitations</h3>
      {_list_html(limitations, "표시할 제한 사항이 없습니다.")}
    </section>

    <section>
      <h2>Key Insights &amp; Action Plan</h2>
      <p>인사이트 에이전트가 SQL/EDA/분석 근거를 종합해 도출한 실행 제안입니다.</p>
      <h3>Key Insights</h3>
      {_list_html(key_insights, "표시할 핵심 인사이트가 없습니다.")}
      <h3>Action Plan</h3>
      {_list_html(action_plan, "제안된 실행 계획이 없습니다.")}
      <h3>Limitations</h3>
      {_list_html(insight_limitations, "표시할 제한 사항이 없습니다.")}
    </section>

    <section>
      <h2>Visualization</h2>
      {chart_gallery}
    </section>

    <section>
      <h2>Open Results</h2>
      <ul>
        <li><a href="final_report.html">Final report (HTML)</a></li>
        <li><a href="final_report.md">Final report (Markdown)</a></li>
        <li><a href="sql_result.csv">SQL result CSV</a></li>
        <li><a href="generated_sql.sql">Generated SQL</a></li>
        <li><a href="run_summary.json">Run summary JSON</a></li>
      </ul>
    </section>

    <details>
      <summary>Generated SQL</summary>
      <pre>{html.escape(str(sql_plan.get("sql_draft", {}).get("sql") or ""))}</pre>
    </details>

    <details>
      <summary>Technical Artifacts</summary>
      <table>
        <thead><tr><th>Agent</th><th>Artifact ID</th><th>Type</th><th>Local Path</th></tr></thead>
        <tbody>{"".join(artifact_rows)}</tbody>
      </table>
    </details>
  </main>
</body>
</html>
"""


def _write_outputs(adapter: BackendAdapter, state: OrchestrationState, query: str, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        agent_name: [_artifact_preview(adapter, artifact_id) for artifact_id in artifact_ids]
        for agent_name, artifact_ids in state.artifact_ids.items()
    }
    summary = {
        "query": query,
        **_state_summary(state),
        "artifacts": artifacts,
    }

    summary_path = output_dir / "run_summary.json"
    sql_path = output_dir / "generated_sql.sql"
    artifact_manifest_path = output_dir / "artifact_manifest.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    sql_path.write_text(state.generated_sql or "", encoding="utf-8")
    artifact_manifest_path.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _copy_artifacts(output_dir, artifacts)
    sql_result_csv = _write_sql_result_csv(output_dir, artifacts)
    report_outputs = _write_report_html(output_dir, artifacts)

    outputs = {
        "output_dir": output_dir,
        "summary": summary_path,
        "generated_sql": sql_path,
        "artifact_manifest": artifact_manifest_path,
        **report_outputs,
    }
    if sql_result_csv:
        outputs["sql_result_csv"] = sql_result_csv
    return outputs


def _print_text_summary(
    state: OrchestrationState,
    adapter: BackendAdapter,
    *,
    outputs: dict[str, Path] | None,
    show_sql: bool,
) -> None:
    print("\n=== Orchestration Result ===")
    print(f"thread_id:      {state.thread_id}")
    print(f"terminal_state: {state.terminal_state}")
    print(f"route_kind:      {state.route_kind}")
    print(f"planner_mode:    {state.planner_mode}")
    print(f"current_step:    {state.current_step}")
    print(f"completed:       {state.completed_agents}")

    if outputs:
        print(f"generated_sql:   {outputs['generated_sql']}")
        print(f"result_folder:   {outputs['output_dir']}")
        print(f"run_summary:     {outputs['summary']}")
        if "final_report_html" in outputs:
            print(f"report_html:     {outputs['final_report_html']}")
            print(f"open_this:       {outputs['index_html']}")
    else:
        print("generated_sql:   (output disabled)")

    if state.approval_ids:
        print(f"approval_ids:   {state.approval_ids}")

    if state.artifact_ids:
        print("\n=== Artifacts ===")
        for agent_name, artifact_ids in state.artifact_ids.items():
            print(f"{agent_name}: {artifact_ids}")

    if state.final_answer:
        print("\n=== Final Answer ===")
        print(state.final_answer)

    if show_sql:
        print("\n=== Generated SQL ===")
        print(state.generated_sql or "(empty)")

    if state.error_state:
        print("\n=== Error State ===")
        print(json.dumps(state.error_state, ensure_ascii=False, indent=2, default=str))

    print("\n=== Data Dir ===")
    print(Path(adapter.base_data_dir).resolve())


def _print_approval_resume_hint(state: OrchestrationState, args: argparse.Namespace) -> None:
    metadata = _approval_resume_metadata(state, args)
    if not metadata:
        return
    print("\n=== Approval Resume ===")
    print("resume payload:")
    print(json.dumps(metadata["resume_payload"], ensure_ascii=False, indent=2))
    print("\nresume command:")
    print(metadata["resume_command"])


def _handle_result(
    adapter: BackendAdapter,
    result: Any,
    args: argparse.Namespace,
    *,
    query_for_summary: str,
) -> None:
    if not isinstance(result, SupervisorRunResult):
        raise RuntimeError("Supervisor resume result is not a CLI-compatible result.")

    if result.kind == "interrupt":
        if args.json:
            print(json.dumps(_interrupt_summary(result, args), ensure_ascii=False, indent=2, default=str))
        else:
            _print_interrupt_summary(result, args)
        return

    if result.state is None:
        raise RuntimeError("Supervisor state result is missing state payload.")

    state = result.state
    summary_query = state.user_query or query_for_summary
    outputs = None
    if not args.no_output:
        outputs = _write_outputs(adapter, state, summary_query, Path(args.output_dir))

    if args.json:
        payload = _state_summary(state)
        payload.update(_approval_resume_metadata(state, args))
        if outputs:
            payload["outputs"] = {key: str(value) for key, value in outputs.items()}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        _print_text_summary(state, adapter, outputs=outputs, show_sql=args.show_sql)
        _print_approval_resume_hint(state, args)

    if outputs and not args.no_open and not args.json and hasattr(os, "startfile"):
        os.startfile(outputs["index_html"])


def _handle_interactive_result(
    adapter: BackendAdapter,
    supervisor: SupervisorAgent,
    result: SupervisorRunResult,
    args: argparse.Namespace,
    *,
    query_for_summary: str,
) -> None:
    while True:
        if result.kind == "interrupt":
            if result.interrupt is None:
                raise RuntimeError("interrupt 결과에 payload가 없습니다.")
            _print_interrupt_summary(result, args)
            if result.interrupt.type == "analysis_review":
                resume_payload = _interactive_analysis_review_payload(result.interrupt)
                result = supervisor.resume(result.interrupt.thread_id, resume_payload)
                query_for_summary = "[resume analysis review]"
                continue
            answer = input("\n답변: ").strip()
            if not answer:
                raise SystemExit("답변이 비어 있습니다.")
            result = supervisor.resume(result.interrupt.thread_id, {"answer": answer})
            query_for_summary = f"[resume] {answer}"
            continue

        if result.state is None:
            raise RuntimeError("Supervisor state result is missing state payload.")

        state = result.state
        if state.terminal_state == SupervisorTerminalState.needs_user_approval and state.thread_id:
            _handle_result(adapter, result, args, query_for_summary=query_for_summary)
            approval_answer = input("\n승인 후 계속할까요? [y/N]: ").strip().lower()
            if approval_answer not in {"y", "yes", "예", "ㅇ"}:
                return
            result = supervisor.resume(state.thread_id, {"approved": True})
            query_for_summary = "[resume approved]"
            continue

        _handle_result(adapter, result, args, query_for_summary=query_for_summary)
        return


def _interactive_analysis_review_payload(
    payload: SupervisorInterruptPayload,
) -> dict[str, str]:
    review_request = payload.review_request or {}
    option_ids = {
        str(option.get("id") or "")
        for option in review_request.get("options", [])
        if option.get("id")
    }
    allow_free_text = bool(review_request.get("allow_free_text"))
    while True:
        answer = input("\noption ID 또는 의견: ").strip()
        if not answer:
            print("입력이 비어 있습니다. 다시 입력해 주세요.")
            continue
        if answer in option_ids:
            return {
                "approval_id": str(payload.approval_id or ""),
                "selected_option_id": answer,
            }
        if allow_free_text:
            return {
                "approval_id": str(payload.approval_id or ""),
                "free_text": answer,
            }
        print("허용된 option ID를 입력해 주세요.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DATA_Analyst_Assistant_Agent through the LangGraph Supervisor.",
    )
    parser.add_argument("query", nargs="?", help="User analysis question. If omitted, stdin prompt is used.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help=(
            "Thread id for the backend run. "
            "New runs generate one automatically; resume commands must pass the interrupted thread id."
        ),
    )
    parser.add_argument("--datasource-id", default=None, help="Optional backend datasource id.")
    parser.add_argument(
        "--resume-answer",
        default=None,
        help=(
            "Resume a clarification interrupt with this answer. "
            "Use the same --thread-id from the interrupted run."
        ),
    )
    parser.add_argument(
        "--approve",
        "--resume-approved",
        dest="resume_approved",
        action="store_true",
        help="Resume a pending approval state. Use the same --thread-id from the waiting run.",
    )
    parser.add_argument(
        "--resume-option-id",
        default=None,
        help="Analysis review에서 선택할 option ID.",
    )
    parser.add_argument(
        "--resume-free-text",
        default=None,
        help="Analysis review에 제출할 자유 의견.",
    )
    parser.add_argument(
        "--resume-approval-id",
        default=None,
        help="Analysis review interrupt의 approval ID.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument("--show-sql", action="store_true", help="Print generated SQL in text output.")
    parser.add_argument("--dotenv", default=".env", help="Path to dotenv file. Defaults to .env.")
    interactive_group = parser.add_mutually_exclusive_group()
    interactive_group.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Ask for clarification and approval input in the same terminal session.",
    )
    interactive_group.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Do not prompt in the current process; print resume commands instead.",
    )
    parser.set_defaults(interactive=None)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for run_summary.json, generated_sql.sql, and copied artifacts.",
    )
    parser.add_argument("--no-output", action="store_true", help="Do not write output files.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated HTML report automatically.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    resume_answer = None
    if args.resume_answer is not None:
        resume_answer = str(args.resume_answer).strip()
        if not resume_answer:
            raise SystemExit("--resume-answer cannot be empty.")

    resume_approved = bool(args.resume_approved)
    resume_option_id = (
        str(args.resume_option_id).strip() if args.resume_option_id is not None else None
    )
    resume_free_text = (
        str(args.resume_free_text).strip() if args.resume_free_text is not None else None
    )
    resume_approval_id = (
        str(args.resume_approval_id).strip() if args.resume_approval_id is not None else None
    )
    if resume_option_id == "":
        raise SystemExit("--resume-option-id cannot be empty.")
    if resume_free_text == "":
        raise SystemExit("--resume-free-text cannot be empty.")
    resume_modes = [
        resume_answer is not None,
        resume_approved,
        resume_option_id is not None,
        resume_free_text is not None,
    ]
    is_resume_mode = any(resume_modes)
    if sum(resume_modes) > 1:
        raise SystemExit(
            "--resume-answer, --approve, --resume-option-id, --resume-free-text cannot be used together."
        )
    if resume_option_id is not None or resume_free_text is not None:
        if not resume_approval_id:
            raise SystemExit("--resume-approval-id is required for analysis review resume.")
    elif resume_approval_id is not None:
        raise SystemExit(
            "--resume-approval-id is only allowed with --resume-option-id or --resume-free-text."
        )
    if args.interactive is True and args.json:
        raise SystemExit("--interactive cannot be used with --json.")
    if is_resume_mode:
        if not args.thread_id:
            raise SystemExit(
                "--thread-id is required when using --resume-answer, --approve, "
                "--resume-option-id, or --resume-free-text."
            )
        if args.query:
            raise SystemExit("resume options cannot be used with a positional query.")
        if args.datasource_id is not None:
            raise SystemExit("resume options cannot be used with --datasource-id.")

    load_dotenv(args.dotenv)
    _normalize_env_aliases()
    _ensure_sql_agent_metadata()

    adapter = BackendAdapter()
    supervisor = SupervisorAgent(adapter)
    if resume_answer is not None:
        result = supervisor.resume(args.thread_id, {"answer": resume_answer})
        _handle_result(adapter, result, args, query_for_summary=f"[resume] {resume_answer}")
        return
    if resume_approved:
        result = supervisor.resume(args.thread_id, {"approved": True})
        _handle_result(adapter, result, args, query_for_summary="[resume approved]")
        return
    if resume_option_id is not None:
        result = supervisor.resume(
            args.thread_id,
            {"approval_id": resume_approval_id, "selected_option_id": resume_option_id},
        )
        _handle_result(adapter, result, args, query_for_summary="[resume analysis option]")
        return
    if resume_free_text is not None:
        result = supervisor.resume(
            args.thread_id,
            {"approval_id": resume_approval_id, "free_text": resume_free_text},
        )
        _handle_result(adapter, result, args, query_for_summary="[resume analysis free text]")
        return

    query = args.query or input("Query: ").strip()
    if not query:
        raise SystemExit("Query is empty.")

    thread_id = args.thread_id or _new_thread_id()
    result = supervisor.run(query, thread_id=thread_id, datasource_id=args.datasource_id)
    if _should_run_interactively(args):
        _handle_interactive_result(adapter, supervisor, result, args, query_for_summary=query)
        return
    _handle_result(adapter, result, args, query_for_summary=query)


if __name__ == "__main__":
    main()
