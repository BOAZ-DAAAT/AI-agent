from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANGCHAIN_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", message="Glyph .* missing from font")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor
from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT_DIR / ".env")


def _normalize_env() -> None:
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


def _artifact_preview(adapter: BackendAdapter, artifact_id: str) -> dict:
    artifact = adapter.get_artifact(artifact_id)
    return {
        "artifact_id": artifact.artifact_id,
        "type": str(artifact.type),
        "uri": artifact.uri,
        "metadata": artifact.metadata,
        "preview": artifact.preview,
        "local_path": str(artifact.local_path) if artifact.local_path else None,
    }


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
            blocks.append("")
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = html.escape(heading.group(2))
            blocks.append(f"<h{level}>{text}</h{level}>")
        elif line.startswith("- "):
            blocks.append(f"<p class=\"bullet\">{html.escape(line)}</p>")
        else:
            escaped = html.escape(line)
            escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
            blocks.append(f"<p>{escaped}</p>")

    if code_lines:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    body = "\n".join(blocks)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #1f2933;
      font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif;
      line-height: 1.65;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 36px 28px 64px;
      background: white;
      min-height: 100vh;
      box-shadow: 0 0 0 1px #e6e8ec;
    }}
    h1, h2, h3, h4 {{ line-height: 1.25; margin: 28px 0 12px; }}
    h1 {{ font-size: 28px; border-bottom: 2px solid #e5e7eb; padding-bottom: 14px; }}
    h2 {{ font-size: 21px; color: #174ea6; }}
    h3 {{ font-size: 17px; }}
    p {{ margin: 8px 0; }}
    .bullet {{ padding-left: 12px; }}
    pre {{
      overflow-x: auto;
      background: #111827;
      color: #f9fafb;
      border-radius: 8px;
      padding: 16px;
      font-size: 13px;
    }}
    code {{ font-family: Consolas, "Courier New", monospace; }}
  </style>
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>
"""


def _first_sql_result(artifacts: dict) -> dict | None:
    for item in artifacts.get("sql_agent", []):
        if item.get("type") == "sql_result":
            return item
    return None


def _sql_agent_answer(artifacts: dict) -> str:
    for item in artifacts.get("sql_agent", []):
        if item.get("metadata", {}).get("kind") == "sql_lang_graph_result":
            answer = item.get("preview", {}).get("final_answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    return ""


def _read_table_rows(csv_path: Path, *, limit: int = 50) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        return [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append({key: value for key, value in row.items()})
        return list(reader.fieldnames or []), rows


def _table_html(columns: list[str], rows: list[dict[str, str]]) -> str:
    if not columns or not rows:
        return "<p>표시할 결과 행이 없습니다.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""
  <div class="table-wrap">
    <table>
      <thead><tr>{header}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>
"""


def _answer_html(answer: str) -> str:
    if not answer:
        return "<p>자연어 요약 답변이 생성되지 않았습니다.</p>"

    def inline(text: str) -> str:
        escaped = html.escape(text)
        return re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)

    blocks = []
    bullet_items: list[str] = []

    def flush_bullets() -> None:
        if bullet_items:
            blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in bullet_items) + "</ul>")
            bullet_items.clear()

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_bullets()
            continue
        if stripped.startswith(("*", "-")):
            bullet_text = stripped.lstrip("*- ").strip()
            bullet_items.append(inline(bullet_text))
        else:
            flush_bullets()
            blocks.append(f"<p>{inline(stripped)}</p>")
    flush_bullets()
    return "\n".join(blocks)


def _load_json_artifact(item: dict | None) -> dict:
    if not item or not item.get("local_path"):
        return {}
    path = Path(item["local_path"])
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _first_artifact_by_agent(artifacts: dict, agent_name: str) -> dict | None:
    items = artifacts.get(agent_name, [])
    return items[0] if items else None


def _metric_cards_html(cards: list[tuple[str, str]]) -> str:
    return "\n".join(
        f'<div class="metric-card"><div class="metric-value">{html.escape(value)}</div><div class="metric-label">{html.escape(label)}</div></div>'
        for label, value in cards
    )


def _list_html(items: list[str]) -> str:
    if not items:
        return "<p>표시할 항목이 없습니다.</p>"
    return "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in items) + "</ul>"


def _eda_html(artifacts: dict) -> str:
    eda_item = _first_artifact_by_agent(artifacts, "eda_agent")
    eda_payload = _load_json_artifact(eda_item)
    profile = eda_payload.get("profile") or (eda_item or {}).get("preview", {})
    main_eda = eda_payload.get("main_eda_agent") or {}

    row_count = profile.get("row_count", (eda_item or {}).get("preview", {}).get("row_count", 0))
    columns = profile.get("columns", (eda_item or {}).get("preview", {}).get("columns", [])) or []
    quality_status = profile.get("quality_status", (eda_item or {}).get("preview", {}).get("quality_status", "unknown"))
    key_issues = profile.get("key_issues", (eda_item or {}).get("preview", {}).get("key_issues", [])) or []
    null_counts = profile.get("null_counts", {}) or {}
    numeric_summary = profile.get("numeric_summary", {}) or {}

    null_rows = "".join(
        f"<tr><td>{html.escape(str(column))}</td><td>{html.escape(str(count))}</td></tr>"
        for column, count in null_counts.items()
    ) or '<tr><td colspan="2">결측치 상세 정보가 없습니다.</td></tr>'

    numeric_rows = []
    for column, stats in numeric_summary.items():
        numeric_rows.append(
            "<tr>"
            f"<td>{html.escape(str(column))}</td>"
            f"<td>{html.escape(str(round(float(stats.get('mean', 0)), 4)))}</td>"
            f"<td>{html.escape(str(round(float(stats.get('min', 0)), 4)))}</td>"
            f"<td>{html.escape(str(round(float(stats.get('max', 0)), 4)))}</td>"
            "</tr>"
        )
    numeric_table = "".join(numeric_rows) or '<tr><td colspan="4">수치형 요약 정보가 없습니다.</td></tr>'

    selected = main_eda.get("selected_columns", {})
    selected_text = ", ".join(
        part for part in [
            f"key={selected.get('key_col')}" if selected.get("key_col") else "",
            f"measures={', '.join(selected.get('measure_cols', [])[:6])}" if selected.get("measure_cols") else "",
            f"time={', '.join(selected.get('time_cols', [])[:3])}" if selected.get("time_cols") else "",
        ] if part
    ) or "EDA 컬럼 선택 메타데이터가 없습니다."

    return f"""
  <section>
    <h2>EDA Profile & Data Quality</h2>
    <div class="metric-grid">
      {_metric_cards_html([
        ("Rows analyzed", f"{row_count:,}" if isinstance(row_count, int) else str(row_count)),
        ("Columns", str(len(columns))),
        ("Quality status", str(quality_status)),
        ("Key issues", str(len(key_issues))),
      ])}
    </div>
    <p><strong>Columns:</strong> {html.escape(', '.join(columns) or '없음')}</p>
    <p><strong>EDA selected fields:</strong> {html.escape(selected_text)}</p>
    <h3>Quality Issues</h3>
    {_list_html([str(issue) for issue in key_issues] or ["감지된 주요 품질 이슈가 없습니다."])}
    <h3>Null Counts</h3>
    <div class="table-wrap"><table><thead><tr><th>Column</th><th>Null count</th></tr></thead><tbody>{null_rows}</tbody></table></div>
    <h3>Numeric Summary</h3>
    <div class="table-wrap"><table><thead><tr><th>Column</th><th>Mean</th><th>Min</th><th>Max</th></tr></thead><tbody>{numeric_table}</tbody></table></div>
  </section>
"""


def _analysis_html(artifacts: dict) -> str:
    item = _first_artifact_by_agent(artifacts, "analysis_agent")
    payload = _load_json_artifact(item)
    preview = (item or {}).get("preview", {})
    findings = payload.get("key_findings") or preview.get("key_findings") or []
    limitations = payload.get("limitations") or preview.get("limitations") or []
    quality_notes = payload.get("data_quality_notes") or preview.get("data_quality_notes") or []
    method = payload.get("method_summary") or preview.get("method_summary") or ""
    return f"""
  <section>
    <h2>Analysis Insights</h2>
    <p>{html.escape(method or '분석 요약이 생성되지 않았습니다.')}</p>
    <h3>Key Findings</h3>
    {_list_html([str(item) for item in findings])}
    <h3>Data Quality Notes</h3>
    {_list_html([str(item) for item in quality_notes] or ['추가 데이터 품질 메모가 없습니다.'])}
    <h3>Limitations</h3>
    {_list_html([str(item) for item in limitations])}
  </section>
"""


def _visualization_html(artifacts: dict) -> str:
    item = _first_artifact_by_agent(artifacts, "visualization_agent")
    preview = (item or {}).get("preview", {})
    if not preview:
        return """
  <section>
    <h2>Visualization</h2>
    <p>시각화 산출물이 생성되지 않았습니다.</p>
  </section>
"""
    encoding = preview.get("encoding", {})
    x_field = (encoding.get("x") or {}).get("field", "unknown")
    y_field = (encoding.get("y") or {}).get("field", "unknown")
    return f"""
  <section>
    <h2>Visualization</h2>
    <div class="metric-grid">
      {_metric_cards_html([
        ("Chart type", str(preview.get("chart_type", "unknown"))),
        ("X field", str(x_field)),
        ("Y field", str(y_field)),
      ])}
    </div>
    <p><strong>Title:</strong> {html.escape(str(preview.get("title", "제목 없는 차트")))}</p>
    <p>이번 실행에서는 시각화 스펙 산출물이 생성되었습니다. 원본 Vega-Lite 스펙은 전체 리포트 또는 산출물 파일에서 확인할 수 있습니다.</p>
  </section>
"""


def _chart_title(filename: str) -> str:
    stem = Path(filename).stem
    labels = {
        "bar_top": "Top values",
        "bar_bottom": "Bottom values",
        "scatter": "Relationship",
        "bubble": "Volume relationship",
        "dist": "Distribution",
        "box": "Box plot",
        "violin": "Violin plot",
        "correlation_heatmap": "Correlation heatmap",
        "catdist": "Category distribution",
        "grouped_bar": "Grouped comparison",
        "heatmap": "Heatmap",
        "radar": "Radar summary",
    }
    for prefix, label in labels.items():
        if stem.startswith(prefix):
            field = stem.replace(prefix, "").strip("_").replace("_", " ")
            return f"{label}: {field}" if field else label
    return stem.replace("_", " ").title()


def _select_eda_charts(columns: list[str], *, limit: int = 8) -> list[Path]:
    # EDA 의 chart_selector 가 선별해 둔 outputs/key 를 우선 사용한다 (EDA의 실제 차트 선택).
    key_dir = ROOT_DIR / "DATA_Analyst_Assistant_Agent" / "agents" / "eda" / "outputs" / "key"
    if key_dir.exists():
        key_charts = sorted(key_dir.glob("*.png"))
        if key_charts:
            return key_charts[:limit]

    # 폴백: outputs/key 가 비어 있으면 기존 점수 기반 선택 (outputs/all)
    charts_dir = ROOT_DIR / "DATA_Analyst_Assistant_Agent" / "agents" / "eda" / "outputs" / "all"
    if not charts_dir.exists():
        return []

    column_tokens = [re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_") for column in columns]
    preferred_names = [
        "bar_top_delay_rate.png",
        "bar_top_total_orders.png",
        "bar_top_delayed_orders.png",
        "bar_top_avg_review_score.png",
        "scatter_delay_rate_vs_avg_review_score.png",
        "scatter_total_orders_vs_delay_rate.png",
        "bubble_total_orders_vs_delayed_orders.png",
        "correlation_heatmap.png",
        "dist_delay_rate.png",
        "box_delay_rate.png",
        "catdist_customer_state.png",
    ]

    candidates = {path.name: path for path in charts_dir.glob("*.png")}
    selected: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | None) -> None:
        if not path or path.name in seen or len(selected) >= limit:
            return
        selected.append(path)
        seen.add(path.name)

    for name in preferred_names:
        add(candidates.get(name))

    if len(selected) < limit:
        scored: list[tuple[int, str, Path]] = []
        for path in candidates.values():
            name = path.name.lower()
            score = sum(3 for token in column_tokens if token and token in name)
            if name.startswith("bar_top"):
                score += 6
            if name.startswith("scatter") or name.startswith("bubble"):
                score += 5
            if name == "correlation_heatmap.png":
                score += 5
            if name.startswith("dist") or name.startswith("box"):
                score += 3
            if score > 0:
                scored.append((-score, name, path))
        for _, _, path in sorted(scored):
            add(path)

    return selected


def _charts_html(columns: list[str], output_dir: Path) -> str:
    selected = _select_eda_charts(columns)
    if not selected:
        return """
  <section>
    <h2>EDA Visualizations</h2>
    <p>이번 실행에서 연결할 EDA 차트 이미지 파일을 찾지 못했습니다.</p>
  </section>
"""

    chart_cards = []
    charts_output_dir = output_dir / "charts"
    charts_output_dir.mkdir(parents=True, exist_ok=True)

    for source in selected:
        destination = charts_output_dir / source.name
        shutil.copy2(source, destination)
        encoded = base64.b64encode(destination.read_bytes()).decode("ascii")
        chart_cards.append(
            f"""
      <div class="chart-card">
        <div class="chart-name">{html.escape(source.name)}</div>
        <h3>{html.escape(_chart_title(source.name))}</h3>
        <img src="data:image/png;base64,{encoded}" alt="{html.escape(_chart_title(source.name))}">
      </div>
"""
        )

    return f"""
  <section>
    <h2>EDA Visualizations</h2>
    <p>아래 차트는 연결된 EDA agent가 생성한 시각화 결과입니다. 사용자가 최종 결과를 바로 확인할 수 있도록 리포트 안에 함께 포함했습니다.</p>
    <div class="charts-grid">
      {''.join(chart_cards)}
    </div>
  </section>
"""


def _write_demo_outputs(adapter: BackendAdapter, state, query: str) -> dict[str, Path]:
    output_dir = ROOT_DIR / "demo_outputs" / "latest"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        agent_name: [_artifact_preview(adapter, artifact_id) for artifact_id in artifact_ids]
        for agent_name, artifact_ids in state.artifact_ids.items()
    }
    summary = {
        "query": query,
        "terminal_state": str(state.terminal_state),
        "route_kind": state.route_kind,
        "current_step": state.current_step,
        "completed_agents": state.completed_agents,
        "generated_sql": state.generated_sql,
        "error_state": state.error_state,
        "artifacts": artifacts,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "generated_sql.sql").write_text(state.generated_sql or "", encoding="utf-8")
    sql_result_item = _first_sql_result(artifacts)
    sql_result_csv_path = output_dir / "sql_result.csv"
    if sql_result_item and sql_result_item.get("local_path"):
        source = Path(sql_result_item["local_path"])
        if source.exists():
            shutil.copy2(source, sql_result_csv_path)

    copied_files: list[Path] = []
    final_report_md = ""
    report_paths = [
        Path(item["local_path"])
        for item in artifacts.get("report_agent", [])
        if item.get("local_path")
    ]
    if report_paths and report_paths[0].exists():
        final_report_md = report_paths[0].read_text(encoding="utf-8")
        report_md_path = output_dir / "final_report.md"
        report_md_path.write_text(final_report_md, encoding="utf-8")
        copied_files.append(report_md_path)

    final_report_html = _markdown_to_html(final_report_md or "# Report\n\n리포트 산출물이 생성되지 않았습니다.", title="Final Report")
    report_html_path = output_dir / "final_report.html"
    report_html_path.write_text(final_report_html, encoding="utf-8")
    copied_files.append(report_html_path)

    for agent_name, items in artifacts.items():
        agent_dir = output_dir / "artifacts" / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            local_path = item.get("local_path")
            if not local_path:
                continue
            source = Path(local_path)
            if source.exists():
                destination = agent_dir / f"{item['artifact_id']}_{source.name}"
                shutil.copy2(source, destination)
                copied_files.append(destination)

    links = [
        ("Result table CSV", "sql_result.csv"),
        ("Final report (HTML)", "final_report.html"),
        ("Final report (Markdown)", "final_report.md"),
        ("Generated SQL", "generated_sql.sql"),
        ("Run summary JSON", "run_summary.json"),
    ]
    link_html = "\n".join(
        f'<li><a href="{html.escape(path)}">{html.escape(label)}</a></li>'
        for label, path in links
        if (output_dir / path).exists()
    )
    artifact_rows = []
    for agent_name, items in artifacts.items():
        for item in items:
            artifact_rows.append(
                "<tr>"
                f"<td>{html.escape(agent_name)}</td>"
                f"<td>{html.escape(item['artifact_id'])}</td>"
                f"<td>{html.escape(str(item['type']))}</td>"
                f"<td>{html.escape(str(item.get('local_path') or ''))}</td>"
                "</tr>"
            )
    result_columns, result_rows = _read_table_rows(sql_result_csv_path)
    result_table_html = _table_html(result_columns, result_rows)
    answer_html = _answer_html(_sql_agent_answer(artifacts))
    eda_html = _eda_html(artifacts)
    analysis_html = _analysis_html(artifacts)
    visualization_html = _visualization_html(artifacts)
    charts_html = _charts_html(result_columns, output_dir)
    index_html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Orchestration Demo Result</title>
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
    .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; margin-top: 14px; }}
    .chart-card {{ background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
    .chart-card h3 {{ font-size: 15px; margin: 6px 0 10px; color: #111827; }}
    .chart-card img {{ display: block; width: 100%; height: auto; border-radius: 6px; background: white; }}
    .chart-name {{ color: #6b7280; font-size: 12px; word-break: break-all; }}
    details {{ margin-top: 24px; }}
    summary {{ cursor: pointer; font-weight: 700; color: #174ea6; }}
    a {{ color: #174ea6; }}
  </style>
</head>
<body>
  <main>
  <h1>Data Analysis Result</h1>
  <p><span class="badge">{html.escape(str(state.terminal_state))}</span></p>
  <p><strong>Question:</strong> {html.escape(query)}</p>
  <p><strong>Completed workflow:</strong> {html.escape(", ".join(state.completed_agents))}</p>

  <section>
  <h2>Answer</h2>
  <div class="answer">{answer_html}</div>
  </section>

  <section>
  <h2>SQL Result Table</h2>
  <p>요청한 분석 결과는 아래 표에서 바로 확인할 수 있습니다. 원본 데이터는 <a href="sql_result.csv">sql_result.csv</a> 파일로도 저장되어 있습니다.</p>
  {result_table_html}
  </section>

  {eda_html}

  {charts_html}

  {analysis_html}

  {visualization_html}

  <section>
  <h2>Open Results</h2>
  <ul>{link_html}</ul>
  </section>

  <details>
    <summary>Generated SQL</summary>
    <pre>{html.escape(state.generated_sql or "(없음)")}</pre>
  </details>

  <details>
    <summary>Technical Artifacts</summary>
    <table>
      <thead><tr><th>Agent</th><th>Artifact ID</th><th>Type</th><th>Local Path</th></tr></thead>
      <tbody>{''.join(artifact_rows)}</tbody>
    </table>
  </details>
  </main>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return {"output_dir": output_dir, "index": index_path, "report_html": report_html_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DATA_Analyst_Assistant_Agent orchestration demo.")
    parser.add_argument("query", nargs="?", help="User question to run through the orchestration graph.")
    parser.add_argument("--thread-id", default="demo-thread", help="Optional thread id for the backend run.")
    parser.add_argument("--datasource-id", default=None, help="Optional backend datasource id.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated HTML result automatically.")
    args = parser.parse_args()

    query = args.query or input("Query: ").strip()
    if not query:
        raise SystemExit("Query is empty.")

    _load_dotenv()
    _normalize_env()
    _ensure_sql_agent_metadata()

    adapter = BackendAdapter()
    supervisor = SQLAgentSupervisor(adapter)
    state = supervisor.run(query, thread_id=args.thread_id, datasource_id=args.datasource_id)
    outputs = _write_demo_outputs(adapter, state, query)

    print("\n=== Orchestration Result ===")
    print(f"terminal_state: {state.terminal_state}")
    print(f"route_kind:      {state.route_kind}")
    print(f"current_step:    {state.current_step}")
    print(f"completed:       {state.completed_agents}")
    print(f"generated_sql:   {outputs['output_dir'] / 'generated_sql.sql'}")
    print(f"result_folder:   {outputs['output_dir']}")
    print(f"open_this:       {outputs['index']}")

    if state.error_state:
        print("\n=== Error State ===")
        print(json.dumps(state.error_state, ensure_ascii=False, indent=2, default=str))

    print("\n=== Data Dir ===")
    print(Path(adapter.base_data_dir).resolve())

    if not args.no_open and hasattr(os, "startfile"):
        os.startfile(outputs["index"])


if __name__ == "__main__":
    main()
