from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv

from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState
from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir


ROOT_DIR = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _normalize_env_aliases() -> None:
    aliases = {
        "DB_HOST": "MYSQL_HOST",
        "DB_USER": "MYSQL_USERNAME",
        "DB_NAME": "MYSQL_DATABASE",
        "DB_PORT": "MYSQL_PORT",
        "DB_PASSWORD": "MYSQL_PASSWORD",
        "GOOGLE_API_KEY": "GEMINI_API_KEY",
    }
    for target, source in aliases.items():
        if not os.getenv(target) and os.getenv(source):
            os.environ[target] = os.getenv(source, "")
    os.environ.setdefault("DB_PASSWORD", "")


def _ensure_sql_agent_metadata() -> None:
    data_dir = sql_metadata_dir()
    schema_path = data_dir / "db_schema.json"
    integrity_path = data_dir / "db_integrity_result.json"
    if schema_path.exists() and integrity_path.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    from sqlalchemy import inspect

    from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine

    engine = get_db_engine()
    if engine is None:
        raise RuntimeError("DB engine creation failed. Check MYSQL_* or DB_* values in .env.")

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
        "generated_sql": state.generated_sql,
        "artifact_ids": state.artifact_ids,
        "approval_ids": state.approval_ids,
        "mart_id": state.mart_id,
        "error_state": state.error_state,
    }


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


def _metric_cards(cards: list[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="metric-card"><div class="metric-value">{html.escape(value)}</div><div class="metric-label">{html.escape(label)}</div></div>'
        for label, value in cards
    )


def _build_index_html(report_markdown: str, artifacts: dict[str, list[dict[str, Any]]]) -> str:
    sql_item = _first_artifact(artifacts, "sql_agent", "sql_result")
    sql_plan = _artifact_json(_first_artifact(artifacts, "sql_agent", "sql_lang_graph_result"))
    analysis = _artifact_json(_first_artifact(artifacts, "analysis_agent", "analysis_result"))
    chart = _artifact_json(_first_artifact(artifacts, "visualization_agent", "vega_lite_spec"))
    columns, rows = _read_csv_artifact(sql_item)

    final_answer = str(sql_plan.get("final_answer") or sql_plan.get("preview", {}).get("final_answer") or "")
    if not final_answer:
        preview = (_first_artifact(artifacts, "sql_agent", "sql_lang_graph_result") or {}).get("preview", {})
        final_answer = str(preview.get("final_answer") or "")

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

    chart_preview = (_first_artifact(artifacts, "visualization_agent") or {}).get("preview", {})
    encoding = chart_preview.get("encoding", {}) if isinstance(chart_preview, dict) else {}
    x_field = ((encoding.get("x") or {}).get("field") if isinstance(encoding, dict) else "") or "unknown"
    y_field = ((encoding.get("y") or {}).get("field") if isinstance(encoding, dict) else "") or "unknown"
    chart_type = chart_preview.get("chart_type") or chart.get("mark", {}).get("type") or chart.get("mark") or "unknown"
    chart_title = chart_preview.get("title") or chart.get("title") or "제목 없는 차트"
    chart_body = _chart_svg(columns, rows, str(x_field), str(y_field))

    key_findings = analysis.get("key_findings") or (_first_artifact(artifacts, "analysis_agent") or {}).get("preview", {}).get("key_findings") or []
    limitations = analysis.get("limitations") or (_first_artifact(artifacts, "analysis_agent") or {}).get("preview", {}).get("limitations") or []
    method = analysis.get("method_summary") or (_first_artifact(artifacts, "analysis_agent") or {}).get("preview", {}).get("method_summary") or ""

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
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }}
    .metric-card {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
    .metric-value {{ font-size: 19px; font-weight: 700; color: #111827; }}
    .metric-label {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
    .chart-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; margin-top: 14px; padding: 10px; }}
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
      <h2>Analysis Insights</h2>
      <p>{html.escape(str(method or "분석 요약이 생성되지 않았습니다."))}</p>
      <h3>Key Findings</h3>
      {_list_html(key_findings, "표시할 핵심 발견 사항이 없습니다.")}
      <h3>Limitations</h3>
      {_list_html(limitations, "표시할 제한 사항이 없습니다.")}
    </section>

    <section>
      <h2>Visualization</h2>
      <div class="metric-grid">
        {_metric_cards([("Chart type", str(chart_type)), ("X field", str(x_field)), ("Y field", str(y_field))])}
      </div>
      <p><strong>Title:</strong> {html.escape(str(chart_title))}</p>
      {chart_body}
      <p>이번 실행에서는 Vega-Lite 차트 스펙도 artifact로 저장되었습니다.</p>
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

    if show_sql:
        print("\n=== Generated SQL ===")
        print(state.generated_sql or "(empty)")

    if state.error_state:
        print("\n=== Error State ===")
        print(json.dumps(state.error_state, ensure_ascii=False, indent=2, default=str))

    print("\n=== Data Dir ===")
    print(Path(adapter.base_data_dir).resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DATA_Analyst_Assistant_Agent directly through SQLAgentSupervisor.",
    )
    parser.add_argument("query", nargs="?", help="User analysis question. If omitted, stdin prompt is used.")
    parser.add_argument("--thread-id", default="daaa-manual-run", help="Thread id for the backend run.")
    parser.add_argument("--datasource-id", default=None, help="Optional backend datasource id.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument("--show-sql", action="store_true", help="Print generated SQL in text output.")
    parser.add_argument("--dotenv", default=".env", help="Path to dotenv file. Defaults to .env.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "daaa_outputs" / "latest"),
        help="Directory for run_summary.json, generated_sql.sql, and copied artifacts.",
    )
    parser.add_argument("--no-output", action="store_true", help="Do not write output files.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated HTML report automatically.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv(args.dotenv)
    _normalize_env_aliases()
    _ensure_sql_agent_metadata()

    query = args.query or input("Query: ").strip()
    if not query:
        raise SystemExit("Query is empty.")

    adapter = BackendAdapter()
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run(query, thread_id=args.thread_id, datasource_id=args.datasource_id)
    outputs = None
    if not args.no_output:
        outputs = _write_outputs(adapter, state, query, Path(args.output_dir))

    if args.json:
        payload = _state_summary(state)
        if outputs:
            payload["outputs"] = {key: str(value) for key, value in outputs.items()}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        _print_text_summary(state, adapter, outputs=outputs, show_sql=args.show_sql)

    if outputs and not args.no_open and not args.json and hasattr(os, "startfile"):
        os.startfile(outputs["index_html"])


if __name__ == "__main__":
    main()
