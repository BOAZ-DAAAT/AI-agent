"""NodeSummaryResult → 마크다운 렌더러 (포스터/데모용 샘플 추출).

UI가 아직 없어서, 노드 서머리 결과를 렌더링해서 눈으로 확인하려면 마크다운이 필요하다.
차트는 artifact_id만 갖고 있으므로, 렌더링 시점에 실제 PNG 바이트를 로컬 폴더로 받아와
상대경로로 임베드한다(마크다운 뷰어가 바로 렌더할 수 있게).

한눈에 훑어볼 수 있도록 텍스트 나열 대신 구조를 다양화한다(GitHub/VS Code 마크다운
미리보기 기준):
- 이름+짧은 설명이 반복되는 것(파생변수 등) → 표
- 품질이슈/한계/가설처럼 리스트인데 성격이 다른 것 → GFM 콜아웃(> [!WARNING] 등, 색깔 박스로
  렌더됨: WARNING=주황, TIP=초록, IMPORTANT=보라, NOTE=파랑)
- 차트+긴 서술이 함께 있는 것(통계발견/가설검정) → 차트는 항상 보이고 본문은 <details> 토글로
  접어서 클릭해야 펼쳐지게(제목만 봐도 훑을 수 있게)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import (
    AnalysisSummaryDetail,
    EDASummaryDetail,
    FindingSection,
    InsightSummaryDetail,
    NodeSummaryResult,
    SQLSummaryDetail,
)


def render_node_summary_markdown(result: NodeSummaryResult, runtime: AgentRuntime, out_dir: Path) -> str:
    """result를 마크다운 문자열로 렌더링하고, 참조된 차트는 out_dir/charts/에 PNG로 받아둔다.

    반환된 마크다운은 out_dir 기준 상대경로(charts/xxx.png)로 이미지를 참조하므로,
    .md 파일을 out_dir 안에 저장해야 이미지가 바로 보인다.
    """
    chart_map = _save_charts(result.detail, runtime, out_dir)

    lines: list[str] = [f"# {result.title}", "", f"*{result.subtitle}*", "", "## 배경", result.background, ""]
    lines += _RENDERERS[result.detail.kind](result.detail, chart_map)
    lines += ["## 결론", "", result.conclusion, ""]
    lines += [f"> **한 줄 요약**: {result.key_finding}", ""]
    if result.code_used:
        lines += ["## 실행된 코드", "", "```sql", result.code_used, "```", ""]
    if result.fallback_used:
        lines += _callout("CAUTION", "자동 서술 생성 실패", ["근거 원본으로 대체되었습니다."])
    return "\n".join(lines)


def render_node_summary_artifact_markdown(result: NodeSummaryResult) -> str:
    """실서비스에 저장할 Markdown을 만든다.

    차트 파일을 로컬로 복사하지 않고 artifact URI로 참조한다. 백엔드/프론트는 이 논리 URI를
    권한이 확인된 이미지 content URL로 해석한다.
    """
    lines: list[str] = [
        f"# {result.title}", "", f"*{result.subtitle}*", "",
        "## 분석 맥락", "", result.background, "",
    ]
    lines += _SERVICE_RENDERERS[result.detail.kind](result.detail)
    lines += ["## 단계 해석", "", result.conclusion, ""]
    if result.fallback_used:
        lines += _callout("CAUTION", "자동 서술 생성 실패", ["근거 원본으로 대체되었습니다."])
    return "\n".join(lines)


def _artifact_chart_section(section: FindingSection) -> list[str]:
    """선택 이유 → 실제 차트 → 관찰 결과 순서를 보존한다."""
    lines = [f"### {section.heading}", ""]
    if section.rationale:
        lines += [section.rationale, ""]
    for chart_id in section.chart_artifact_ids:
        lines += [f"![{section.heading}](artifact://{chart_id})", ""]
    lines += [section.body, ""]
    return lines


def _service_scope(items: list[str]) -> list[str]:
    return _callout("WARNING", "해석 범위", items)


def _service_handoff(value: str) -> list[str]:
    return ["## 다음 단계 연결", "", value, ""] if value else []


def _render_service_sql(detail: SQLSummaryDetail) -> list[str]:
    lines: list[str] = []
    if detail.design_rationale:
        lines += ["## 데이터 구성 논리", "", detail.design_rationale, ""]
    if detail.source_tables:
        tables = " · ".join(f"`{table}`" for table in detail.source_tables)
        lines += [f"사용한 원천 테이블: {tables}", ""]
    if detail.sql_snippet:
        lines += [
            "<details>", "<summary><strong>실제 실행 SQL 보기</strong></summary>", "",
            "```sql", detail.sql_snippet, "```", "", "</details>", "",
        ]
    if detail.mart_grain or detail.mart_columns or detail.mart_preview:
        lines += ["## 구성된 데이터마트", ""]
        if detail.mart_grain:
            lines += [f"분석 단위: {detail.mart_grain}", ""]
        if detail.mart_columns:
            columns = ", ".join(f"`{column}`" for column in detail.mart_columns)
            lines += [f"포함 컬럼: {columns}", ""]
        if detail.mart_preview:
            lines += [*_render_table(detail.mart_preview[:5]), ""]
    if detail.derived_columns:
        lines += ["<details>", "<summary><strong>파생 변수 정의 보기</strong></summary>", ""]
        lines += _kv_table(("컬럼", "정의"), [(item.heading, item.body) for item in detail.derived_columns])
        lines += ["", "</details>", ""]
    lines += _service_scope(detail.interpretation_scope)
    lines += _service_handoff(detail.handoff)
    return lines


def _render_service_eda(detail: EDASummaryDetail) -> list[str]:
    lines: list[str] = []
    if detail.data_profile:
        lines += ["## 데이터 조건", "", detail.data_profile, ""]
    lines += _callout("WARNING", "분석에 영향을 준 데이터 조건", detail.quality_issues)
    if detail.statistical_findings:
        lines += ["## 탐색 과정과 관찰 결과", ""]
        for section in detail.statistical_findings:
            lines += _artifact_chart_section(section)
    if detail.hypotheses:
        lines += _callout("IMPORTANT", "탐색에서 도출된 가설", detail.hypotheses)
    lines += _service_scope(detail.interpretation_scope)
    lines += _service_handoff(detail.handoff)
    return lines


def _render_service_analysis(detail: AnalysisSummaryDetail) -> list[str]:
    lines: list[str] = []
    if detail.method_decision:
        method = str(detail.method_decision.get("selected_method") or "").strip()
        rationale = str(detail.method_decision.get("rationale") or "").strip()
        lines += ["## 방법론과 선택 근거", ""]
        if method:
            lines += [f"**{method}**", ""]
        if rationale:
            lines += [rationale, ""]
    if detail.supporting_charts:
        lines += ["## 분석에 사용한 시각적 근거", ""]
        for section in detail.supporting_charts:
            lines += _artifact_chart_section(section)
    if detail.hypothesis_tests:
        lines += ["## 검정 과정과 결과", ""]
        for section in detail.hypothesis_tests:
            lines += _artifact_chart_section(section)
    if detail.evidence_tables:
        lines += ["## 실제 근거표", ""]
        for table in detail.evidence_tables:
            lines += [f"### {table.title}", "", *_render_table(table.rows[:5]), ""]
    if detail.interpretation:
        lines += ["## 결과 해석", "", detail.interpretation, ""]
    lines += _service_scope(detail.limitations)
    lines += _service_handoff(detail.handoff)
    return lines


def _render_service_insight(detail: InsightSummaryDetail) -> list[str]:
    lines: list[str] = []
    if detail.evidence_synthesis:
        lines += ["## 근거의 연결", "", detail.evidence_synthesis, ""]
    for section in detail.supporting_charts:
        lines += _artifact_chart_section(section)
    if detail.answer:
        lines += ["## 의미", "", detail.answer, ""]
    if detail.action_plan:
        lines += _callout("IMPORTANT", "활용 방향", detail.action_plan)
    lines += _service_scope(detail.limitations)
    if detail.evidence_sources:
        sources = " · ".join(f"`{item}`" for item in detail.evidence_sources)
        lines += ["## 근거 출처", "", sources, ""]
    return lines


_SERVICE_RENDERERS: dict[str, Callable[[BaseModel], list[str]]] = {
    "sql": _render_service_sql,
    "eda": _render_service_eda,
    "analysis": _render_service_analysis,
    "insight": _render_service_insight,
}


def _save_charts(detail: BaseModel, runtime: AgentRuntime, out_dir: Path) -> dict[str, str]:
    chart_ids = sorted(set(_collect_chart_ids(detail)))
    if not chart_ids:
        return {}
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_map: dict[str, str] = {}
    for chart_id in chart_ids:
        try:
            data = runtime.adapter.read_artifact_bytes(chart_id)
            record = runtime.adapter.get_artifact(chart_id)
            # ArtifactRecord엔 filename 필드가 따로 없다 — 실제 파일명은 local_path에 있다.
            filename = (os.path.basename(record.local_path) if record.local_path else "") or f"{chart_id}.png"
        except Exception:  # noqa: BLE001 — 개별 차트 로딩 실패는 그 차트만 스킵(전체를 막지 않음)
            continue
        (charts_dir / filename).write_bytes(data)
        chart_map[chart_id] = f"charts/{filename}"
    return chart_map


def _collect_chart_ids(value) -> list[str]:
    """detail 안 어디에 있든(EDA/분석/인사이트 필드명이 달라도) FindingSection의 차트 id를 모은다."""
    ids: list[str] = []
    if isinstance(value, FindingSection):
        ids.extend(value.chart_artifact_ids)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            ids.extend(_collect_chart_ids(getattr(value, name)))
    elif isinstance(value, dict):
        for v in value.values():
            ids.extend(_collect_chart_ids(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            ids.extend(_collect_chart_ids(v))
    return ids


# ─────────────────────────────
# 구조화 헬퍼 — 표 / 콜아웃 / 토글
# ─────────────────────────────
def _callout(kind: str, title: str, items: list[str]) -> list[str]:
    """GFM 콜아웃(> [!WARNING] 등) — GitHub/VS Code 미리보기에서 색깔 박스로 렌더된다.

    타입 태그 자체엔 커스텀 제목을 못 붙이는 렌더러가 많아서, 굵은 글씨 줄을 제목처럼 쓴다.
    """
    if not items:
        return []
    lines = [f"> [!{kind}]", f"> **{title}**"]
    lines += [f"> - {item}" for item in items]
    lines.append("")
    return lines


def _toggle(summary: str, body: str) -> list[str]:
    """<details> 토글 — 기본은 접혀있고 클릭해야 본문이 펼쳐진다(raw HTML, 마크다운 안에서 지원됨)."""
    return ["<details>", f"<summary>{summary}</summary>", "", body, "", "</details>", ""]


def _kv_table(headers: tuple[str, str], rows: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return []
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | --- |"]
    lines += [f"| {_format_cell(k)} | {_format_cell(v)} |" for k, v in rows]
    return lines


def _render_table(rows: list[dict]) -> list[str]:
    """dict 레코드 리스트(예: CSV 앞 5행)를 마크다운 표로 렌더링한다."""
    if not rows:
        return []
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col)) for col in columns) + " |")
    return lines


def _format_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_chart_section(section: FindingSection, chart_map: dict[str, str]) -> list[str]:
    """차트(있으면)는 항상 보이게, 서술 본문은 토글 안에 접어서 훑어보기 쉽게 만든다."""
    lines: list[str] = []
    for chart_id in section.chart_artifact_ids:
        rel_path = chart_map.get(chart_id)
        if rel_path:
            lines += [f"![{section.heading}]({rel_path})", ""]
    summary = f"{section.heading} · {section.source_label}" if section.source_label else section.heading
    lines += _toggle(summary, section.body)
    return lines


# ─────────────────────────────
# SQL
# ─────────────────────────────
def _render_sql_detail(detail: SQLSummaryDetail, chart_map: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if detail.source_tables:
        lines += ["## 사용한 원천 테이블", "", " · ".join(f"`{t}`" for t in detail.source_tables), ""]
    lines += _callout("NOTE", "정합성 확인", detail.integrity_checks)
    if detail.derived_columns:
        lines += ["## 파생 변수", ""]
        lines += _kv_table(("컬럼", "설명"), [(s.heading, s.body) for s in detail.derived_columns])
        lines += [""]
    if detail.mart_grain or detail.mart_columns:
        rows = []
        if detail.mart_grain:
            rows.append(("grain", detail.mart_grain))
        if detail.mart_columns:
            rows.append(("컬럼", ", ".join(detail.mart_columns)))
        lines += ["## 데이터마트 설계", ""]
        lines += _kv_table(("항목", "값"), rows)
        lines += [""]
    if detail.mart_preview:
        lines += ["### 최종 구성된 데이터마트는 다음과 같습니다", "", *_render_table(detail.mart_preview), ""]
    return lines


# ─────────────────────────────
# EDA
# ─────────────────────────────
def _render_eda_detail(detail: EDASummaryDetail, chart_map: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if detail.data_profile:
        lines += ["## 데이터 프로파일", "", detail.data_profile, ""]
    lines += _callout("WARNING", "품질 이슈", detail.quality_issues)
    if detail.primary_hypothesis:
        p = detail.primary_hypothesis
        lines += _callout("IMPORTANT", "1순위 가설", [
            f"target: `{p.get('target', '')}`",
            f"feature: `{p.get('feature', '')}`",
            f"method: `{p.get('method', '')}`",
        ])
    if detail.statistical_findings:
        lines += ["## 통계 발견", ""]
        for section in detail.statistical_findings:
            lines += _render_chart_section(section, chart_map)
    lines += _callout("TIP", "제안된 가설", detail.hypotheses)
    if detail.charts_generated:
        lines += ["## 생성한 차트", ""]
        for section in detail.charts_generated:
            lines += _render_chart_section(section, chart_map)
    return lines


# ─────────────────────────────
# 분석
# ─────────────────────────────
def _render_analysis_detail(detail: AnalysisSummaryDetail, chart_map: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if detail.method_decision:
        m = detail.method_decision
        lines += _callout("IMPORTANT", "방법론 선택", [
            f"선택한 방법: {m.get('selected_method', '')}",
            f"근거: {m.get('rationale', '')}",
        ])
    if detail.hypothesis_tests:
        lines += ["## 가설 검정", ""]
        for section in detail.hypothesis_tests:
            lines += _render_chart_section(section, chart_map)
    if detail.key_statistics:
        lines += ["## 핵심 통계", ""]
        for section in detail.key_statistics:
            lines += _render_chart_section(section, chart_map)
    lines += _callout("WARNING", "한계", detail.limitations)
    return lines


# ─────────────────────────────
# 인사이트
# ─────────────────────────────
def _render_insight_detail(detail: InsightSummaryDetail, chart_map: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if detail.answer:
        lines += ["## 답변", "", detail.answer, ""]
    lines += _callout("TIP", "핵심 인사이트", detail.key_insights)
    lines += _callout("IMPORTANT", "실행 제안", detail.action_plan)
    if detail.supporting_charts:
        lines += ["## 근거 차트", ""]
        for section in detail.supporting_charts:
            lines += _render_chart_section(section, chart_map)
    if detail.evidence_sources:
        lines += ["## 근거 출처", "", " · ".join(f"`{e}`" for e in detail.evidence_sources), ""]
    lines += _callout("WARNING", "한계", detail.limitations)
    return lines


_RENDERERS: dict[str, Callable[[BaseModel, dict[str, str]], list[str]]] = {
    "sql": _render_sql_detail,
    "eda": _render_eda_detail,
    "analysis": _render_analysis_detail,
    "insight": _render_insight_detail,
}
