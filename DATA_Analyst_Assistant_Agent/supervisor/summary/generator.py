"""노드 서머리 생성기 — API 레이어가 나중에 호출할 유일한 진입점: generate_node_summary.

설계 원칙(계획 문서 참고):
- API/비동기 실행을 전혀 가정하지 않는 순수 함수. artifact_ids 주면 요약 아티팩트 하나를
  만들어(또는 캐시에서 찾아) ArtifactRef를 리턴한다.
- 숫자 검증은 shared/numeric_verify.py(공용 유틸)를 쓴다 — insight 코드를 직접 가져다 쓰지
  않는다. 검증 실패 시 1회만 재시도, 그래도 실패하면 LLM 없이 근거 그대로 채우는 템플릿
  폴백(절대 숫자를 지어내지 않는다).
- 결과는 "단계형 상세 패널"(title/subtitle → background → checked_items → findings(소제목+
  본문+차트 반복) → conclusion) — 서비스 리포트 톤으로 풍부하게 쓴다.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.shared.numeric_verify import collect_numbers, verify_texts
from DATA_Analyst_Assistant_Agent.supervisor.summary.evidence import ChartRef, NodeEvidence, read_node_evidence
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import FindingSection, NodeSummaryResult

_TOOL_NAME = "supervisor.summary.generator"
_SUMMARY_VERSION = 4                                   # 프롬프트/스키마 바뀌면 올려서 옛 캐시 무효화
                                                        # (v3: findings 섹션 구조 + 서비스 톤 전면 개편)
                                                        # (v4: sql_plan 증거 추출 버그 수정 — 실제 SQL이
                                                        # sql_draft.sql 안에 중첩돼있던 걸 못 읽고 있었음)
_MAX_ATTEMPTS = 2                                      # 최초 1회 + 재시도 1회 (insight 8라운드 루프 아님)

_KIND_LABELS = {
    "sql_result": "SQL 조회",
    "sql_plan": "SQL 조회",
    "eda_summary": "EDA 검증",
    "analysis_result": "분석",
    "insight_payload": "인사이트",
}

# 폴백에서 raw dict key를 그대로 노출하지 않기 위한 사람이 읽는 라벨(Codex 리뷰 반영).
_FACT_LABELS = {
    "final_summary": "핵심 요약",
    "hypotheses": "가설",
    "cautions": "주의할 점",
    "data_level": "데이터 수준 정보",
    "statistical_metadata": "통계 지표",
    "generated_sql": "실행된 SQL",
    "target_table": "대상 테이블",
    "columns": "컬럼 구성",
    "row_count": "행 수",
    "preview": "데이터 미리보기",
    "title": "제목",
    "executive_summary": "핵심 요약",
    "key_findings": "주요 발견",
    "limitations": "한계",
    "method_notes": "분석 방법",
    "answer": "핵심 답변",
    "key_insights": "핵심 인사이트",
    "action_plan": "실행 제안",
}


def generate_node_summary(artifact_ids: list[str], runtime: AgentRuntime) -> ArtifactRef:
    if not artifact_ids:
        raise ValueError("generate_node_summary는 artifact_ids가 최소 1개 필요합니다.")

    cached = _find_cached(artifact_ids, runtime)
    if cached is not None:
        return cached

    evidence = read_node_evidence(artifact_ids, runtime)
    result = _generate_with_llm(evidence) if evidence.facts else None
    if result is None:
        result = _fallback_result(evidence)

    return _register(artifact_ids, result, runtime)


def _find_cached(artifact_ids: list[str], runtime: AgentRuntime) -> ArtifactRef | None:
    """이미 만들어둔 같은 버전 요약이 있으면 그걸 그대로 쓴다(재생성 없음, LLM 호출 0)."""
    anchor = runtime.adapter.get_artifact(artifact_ids[0])
    wanted = set(artifact_ids)
    for record in runtime.adapter.list_artifacts(run_id=anchor.run_id, artifact_type=ArtifactType.file):
        meta = record.metadata
        if meta.get("kind") != "node_summary":
            continue
        if set(meta.get("source_artifact_ids") or []) != wanted:
            continue
        if meta.get("summary_version") != _SUMMARY_VERSION:
            continue
        return record.ref()
    return None


def _generate_with_llm(evidence: NodeEvidence) -> NodeSummaryResult | None:
    llm = get_chat_model(model=os.getenv("SUMMARY_MODEL") or None, model_env="LLM_MODEL")
    numbers = collect_numbers(evidence.facts)
    corpus = json.dumps(evidence.facts, ensure_ascii=False, default=str)
    known_chart_ids = {c.artifact_id for c in evidence.charts}

    feedback = ""
    for _ in range(_MAX_ATTEMPTS):
        try:
            raw = llm.invoke(_build_prompt(evidence, feedback)).content
        except Exception:  # noqa: BLE001 — LLM 호출 실패는 폴백으로 처리
            return None
        parsed = _parse_llm_json(raw)
        if parsed is None:
            feedback = '[형식 오류] JSON 하나만 출력하라. 지정된 필드(title/subtitle/background/checked_items/findings/conclusion/key_finding)를 모두 채워라.'
            continue

        result = _to_result(parsed, evidence, known_chart_ids)
        if result is None:
            feedback = "[누락] title/subtitle/background/conclusion/key_finding은 비울 수 없고 findings는 최소 1개 이상이어야 한다."
            continue

        verify_targets = [result.background, result.conclusion, result.key_finding] + [f.body for f in result.findings]
        ok, missing = verify_texts(verify_targets, numbers, corpus)
        if not ok:
            feedback = f"[검증 실패] 다음 숫자가 근거에 없다: {missing} — 근거에 있는 숫자만 써라. 새 숫자가 필요하면 쓰지 말고 서술만 하라."
            continue

        return result
    return None


def _to_result(parsed: dict[str, Any], evidence: NodeEvidence, known_chart_ids: set[str]) -> NodeSummaryResult | None:
    title = str(parsed.get("title") or "").strip()
    subtitle = str(parsed.get("subtitle") or "").strip()
    background = str(parsed.get("background") or "").strip()
    conclusion = str(parsed.get("conclusion") or "").strip()
    key_finding = str(parsed.get("key_finding") or "").strip()
    checked_items = [str(x).strip() for x in (parsed.get("checked_items") or []) if str(x).strip()][:6]

    raw_findings = parsed.get("findings")
    if not isinstance(raw_findings, list) or not raw_findings:
        return None
    findings: list[FindingSection] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading") or "").strip()
        body = str(item.get("body") or "").strip()
        if not heading or not body:
            continue
        # LLM이 존재하지 않는 차트 id를 지어낼 수 있으니 근거에 실제로 있는 것만 통과시킨다.
        chart_ids = [str(c) for c in (item.get("chart_artifact_ids") or []) if str(c) in known_chart_ids]
        findings.append(FindingSection(
            heading=heading,
            body=body,
            source_label=(str(item.get("source_label")).strip() or None) if item.get("source_label") else None,
            chart_artifact_ids=chart_ids,
        ))

    if not title or not subtitle or not background or not conclusion or not key_finding or not findings:
        return None

    return NodeSummaryResult(
        title=title,
        subtitle=subtitle,
        background=background,
        checked_items=checked_items,
        code_used=evidence.code_used,
        findings=findings,
        conclusion=conclusion,
        key_finding=key_finding,
        source_kind=evidence.source_kind,
        fallback_used=False,
    )


def _fallback_result(evidence: NodeEvidence) -> NodeSummaryResult:
    """LLM 실패/검증 실패 시 — 숫자를 지어내지 않고 근거 그대로 최소 필드를 보장한다.

    raw dict key를 그대로 노출하지 않고 사람이 읽는 라벨로 번역한다(Codex 리뷰 반영).
    """
    label = _KIND_LABELS.get(evidence.source_kind, evidence.source_kind or "알 수 없는 단계")

    findings: list[FindingSection] = []
    checked_items: list[str] = []
    for key, value in evidence.facts.items():
        if not value:
            continue
        section_label = _FACT_LABELS.get(key, key)
        checked_items.append(section_label)
        findings.append(FindingSection(
            heading=section_label,
            body=_format_fact_value(value),
            source_label=section_label,
            chart_artifact_ids=[],
        ))
    if evidence.charts:
        findings.append(FindingSection(
            heading="관련 차트",
            body="자동 요약 생성에 실패해 관련 차트만 안내합니다.",
            source_label="차트",
            chart_artifact_ids=[c.artifact_id for c in evidence.charts],
        ))
    if not findings:
        findings.append(FindingSection(heading=f"{label} 결과", body="근거 데이터가 비어 있습니다.", chart_artifact_ids=[]))

    return NodeSummaryResult(
        title=f"{label} 결과",
        subtitle="자동 요약 생성에 실패해 근거 데이터를 그대로 안내합니다.",
        background=f"이 단계는 {label} 아티팩트입니다. 자동 서술 생성에 실패해 근거 원본을 정리해 보여드립니다.",
        checked_items=checked_items[:6],
        code_used=evidence.code_used,
        findings=findings,
        conclusion="자동 서술 생성에 실패했습니다. 위 항목의 원본 데이터를 참고하세요.",
        key_finding=f"{label} 단계 완료 — 상세는 근거 데이터를 참고하세요.",
        source_kind=evidence.source_kind,
        fallback_used=True,
    )


def _format_fact_value(value: Any) -> str:
    if isinstance(value, str):
        return value or "(내용 없음)"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) if value else "(없음)"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str) if value else "(없음)"
    return str(value)


def _register(artifact_ids: list[str], result: NodeSummaryResult, runtime: AgentRuntime) -> ArtifactRef:
    anchor = runtime.adapter.get_artifact(artifact_ids[0])
    return runtime.adapter.register_artifact(
        anchor.run_id,
        ArtifactType.file,
        content_text=json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        filename="node_summary.json",
        created_by_tool=_TOOL_NAME,
        parent_ids=artifact_ids,
        metadata={
            "kind": "node_summary",
            "source_artifact_ids": sorted(artifact_ids),
            "summary_version": _SUMMARY_VERSION,
        },
        preview={
            "title": result.title,
            "subtitle": result.subtitle,
            "key_finding": result.key_finding,
            "source_kind": result.source_kind,
            "fallback_used": result.fallback_used,
        },
    )


def _build_prompt(evidence: NodeEvidence, feedback: str) -> str:
    facts_text = json.dumps(evidence.facts, ensure_ascii=False, default=str)[:3000]
    charts_text = json.dumps(
        [{"artifact_id": c.artifact_id, "caption": c.caption} for c in evidence.charts],
        ensure_ascii=False,
    )
    feedback_section = f"\n[이전 시도 피드백]\n{feedback}\n" if feedback else ""
    return f"""너는 데이터 분석 파이프라인의 한 단계를 설명하는 전문 리포트 작성자다. 아래 근거만
보고 이 단계가 무엇을 했는지, 실제 서비스에 들어갈 수준으로 풍부하고 전문적으로 서술하라.
새로운 계산·추측을 하지 말고, 근거에 없는 숫자를 만들지 마라.

[근거 종류] {evidence.source_kind}
[근거 내용] {facts_text}
[사용 가능한 차트] {charts_text}
{feedback_section}
[말투] "~를 진행하여 ~를 확인하였습니다. 결론적으로 ~합니다" 같은 정중하고 전문적인
보고서체를 써라. 캐주얼한 구어체는 금지.

[구조 — 반드시 이 필드로 JSON 출력]
- title: 이 단계를 나타내는 구체적인 제목(짧게)
- subtitle: 제목 아래 붙는 한 줄 태그라인
- background: 왜 이 단계가 필요했는지(목적/배경)만. 최대 2문단, 각 문단 2~4문장.
  실제 관찰 내용·수치는 여기 넣지 말고 findings로 내려보내라(background가 길어지면
  findings보다 무거워져서 안 된다).
- checked_items: 이 단계에서 실제로 확인한 항목. 4~6개만(너무 길면 findings와 역할이 겹친다).
- findings: 발견한 내용을 다루는 섹션 리스트(핵심 파트, 최소 1개 이상). 각 섹션:
  {{"heading":"소제목", "body":"본문(1문단 이상, 근거의 구체적 수치를 최대한 인용)",
    "source_label":"분포 차트/요약 테이블/이상치 후보 같은 짧은 카테고리 태그(선택, 없으면 생략)",
    "chart_artifact_ids":["위 [사용 가능한 차트]에 있는 artifact_id만, 없으면 빈 리스트"]}}
  위 차트 목록의 캡션을 보고 관련 있는 섹션에 반드시 연결하라(차트를 findings 밖에 방치하지
  마라). 차트 없는 통찰도 섹션으로 만들 수 있다.
- conclusion: 이 단계에서 얻은 결론. 1문단.
- key_finding: 위 전체를 압축한 한 문장(트리 노드에 짧게 표시될 것이므로 간결하게)

근거에 등장한 숫자만 인용하라. [사용 가능한 차트] 목록에 없는 artifact_id는 만들지 마라.

JSON만 출력하라."""


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            parsed = json.loads(text[start:end])
            break
        except json.JSONDecodeError:
            continue
    else:
        return None
    return parsed if isinstance(parsed, dict) else None
