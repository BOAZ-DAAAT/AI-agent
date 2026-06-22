from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState


def build_report(
    state: OrchestrationState,
    *,
    generated_sql: str = "",
    eda_profile: dict[str, Any] | None = None,
    analysis_result: dict[str, Any] | None = None,
    visualization_result: dict[str, Any] | None = None,
) -> str:
    evidence_lines = _evidence_lines(state)
    visual_lines = _visual_lines(state, visualization_result)
    finding_lines = _finding_lines(analysis_result)
    eda_lines = _eda_lines(eda_profile)
    return "\n".join(
        [
            f"# Data Analyst Assistant Report: {state.goal or state.user_query}",
            "",
            "## User Question",
            state.user_query,
            "",
            "## Generated SQL",
            "```sql",
            generated_sql or "이번 실행에서 SQL이 캡처되지 않았습니다.",
            "```",
            "",
            "## Summary",
            _summary_for_route(state),
            "",
            "## EDA Summary",
            *eda_lines,
            "",
            "## Key Findings",
            *finding_lines,
            "",
            "## Visuals",
            *visual_lines,
            "",
            "## Evidence",
            *evidence_lines,
            "",
            "## Limitations",
            *_limitation_lines(analysis_result),
            "",
            "## Next Actions",
            "- 상세 근거가 필요하면 상위 산출물과 SQL 결과 CSV를 확인하세요.",
            "- 검증 단계에서 재시도 가능한 품질 이슈가 보고되면 SQL 또는 라우팅 조건을 조정하세요.",
        ]
    )


def _evidence_lines(state: OrchestrationState) -> list[str]:
    lines: list[str] = []
    for agent_name, artifact_ids in sorted(state.artifact_ids.items()):
        if agent_name == "report_agent":
            continue
        for artifact_id in artifact_ids:
            lines.append(f"- {agent_name}: {artifact_id}")
    return lines or ["- 등록된 근거 산출물이 없습니다."]


def _summary_for_route(state: OrchestrationState) -> str:
    route_kind = state.route_kind
    if route_kind == "eda":
        return "등록된 SQL 결과 산출물을 바탕으로 데이터 프로파일과 품질 요약을 생성했습니다."
    if route_kind == "trend":
        return "등록된 산출물을 바탕으로 추세 중심의 기술 분석과 차트 구성을 생성했습니다."
    if route_kind == "comprehensive":
        return "EDA, 기술 분석, 시각화 산출물을 하나의 근거 기반 리포트로 통합했습니다."
    if route_kind == "mart":
        return "재사용 가능한 데이터마트 생성 의도를 식별했으며, 영구 저장은 승인 단계가 필요합니다."
    return "등록된 백엔드 산출물을 사용해 SQL 기반 응답을 완료했습니다."


def _eda_lines(profile: dict[str, Any] | None) -> list[str]:
    if not profile:
        return ["- 이 라우트에서는 EDA 프로파일 산출물이 생성되지 않았습니다."]
    return [
        f"- 분석 행 수: {profile.get('row_count', 0)}",
        f"- 컬럼: {', '.join(profile.get('columns', []) or []) or '없음'}",
        f"- 품질 상태: {profile.get('quality_status', 'unknown')}",
        f"- 주요 이슈: {'; '.join(profile.get('key_issues', []) or []) or '없음'}",
    ]


def _finding_lines(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return ["- 이 라우트에서는 분석 산출물이 요청되지 않았습니다."]
    findings = result.get("key_findings", []) or []
    return [f"- {finding}" for finding in findings] or ["- 생성된 분석 인사이트가 없습니다."]


def _visual_lines(state: OrchestrationState, visualization: dict[str, Any] | None) -> list[str]:
    refs = state.artifact_ids.get("visualization_agent", [])
    if not refs:
        return ["- 시각화 산출물이 생성되지 않았습니다."]
    chart_type = (visualization or {}).get("chart_type", "unknown")
    lines = [f"- 시각화 산출물: {artifact_id}" for artifact_id in refs]
    lines.append(f"- 차트 유형: {chart_type}")
    return lines


def _limitation_lines(result: dict[str, Any] | None) -> list[str]:
    limitations = (result or {}).get("limitations", []) or [
        "이번 분석은 현재 실행에서 사용 가능한 SQL 결과 산출물에 한정됩니다.",
        "분석 결과는 기술적 해석이며 인과관계로 해석하면 안 됩니다.",
    ]
    return [f"- {limitation}" for limitation in limitations]
