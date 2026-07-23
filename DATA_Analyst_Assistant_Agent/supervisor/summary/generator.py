"""노드 서머리 생성기 — API 레이어가 나중에 호출할 유일한 진입점: generate_node_summary.

설계 원칙(계획 문서 참고):
- API/비동기 실행을 전혀 가정하지 않는 순수 함수. artifact_ids 주면 요약 아티팩트 하나를
  만들어(또는 캐시에서 찾아) ArtifactRef를 리턴한다.
- 숫자 검증(shared/numeric_verify.py)은 쓰지 않는다 — report/generator.py와 같은 이유로
  팀 판단에 따라 의도적으로 뺐다(2026-07-23). 검증 실패로 원본 덤프 폴백에 떨어져 요약
  UX가 깨지는 걸 우선 피한다. JSON 파싱 실패나 필수 필드 누락처럼 구조 자체가 무너진
  경우에만 재시도 후 폴백으로 간다.
- 그마저 실패해도(2026-07-23) 곧장 정적 템플릿("자동 생성에 실패했습니다")으로 가지
  않는다. shared/plain_narrative.py로 형식 요구를 낮춘 텍스트 생성을 한 번 더 시도해
  실제 근거를 읽고 쓴 문장을 확보한다(_plain_narrative_result). 이것도 LLM 호출 자체가
  죽어야만(API 전체 장애) 최종 정적 템플릿(_fallback_result)으로 간다.
- 결과는 노드 종류(kind)별로 다른 의미적 구조를 쓴다(schemas.py의 discriminated union) —
  SQL은 정합성/파생변수/마트설계, EDA는 프로파일링/통계발견/차트생성, 분석은 방법론선택/
  가설검정, 인사이트는 근거종합/직답. 프롬프트·파서·폴백 전부 kind별로 분기한다.
- primary_hypothesis/method_decision 같이 상류에서 이미 구조화된 값은 LLM이 새로 쓰지
  않고 근거에서 그대로 발췌한다(code_used와 같은 철학 — 지어내지 않는다).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.shared.plain_narrative import generate_plain_narrative
from DATA_Analyst_Assistant_Agent.supervisor.summary.evidence import NodeEvidence, read_node_evidence
from DATA_Analyst_Assistant_Agent.supervisor.summary.markdown import render_node_summary_artifact_markdown
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import (
    AnalysisItem,
    AnalysisSummaryDetail,
    EDASummaryDetail,
    EvidenceTable,
    FindingSection,
    InsightSummaryDetail,
    NodeSummaryResult,
    SQLSummaryDetail,
)

_TOOL_NAME = "supervisor.summary.generator"
_SUMMARY_VERSION = 21                                  # v21: 정적 템플릿 전에 간이 텍스트 생성 폴백 추가
                                                        # (v20: 숫자 검증 폴백 제거(과학적 표기 오탐으로 폴백되던 문제))
                                                        # (v19: Insight AS-IS/TO-BE/ACTION 구조 추가)
                                                        # (v13: SQL 핵심 어구 강조 문법 반영)
                                                        # (v12: SQL 화면 구조/10행 데이터마트 미리보기 반영)
                                                        # (v11: SQL 요약 톤/중복 SQL 노출 정리)
                                                        # (v10: 차트 근거 숫자 검증 + 강제 재생성 지원)
                                                        # (v7: read_node_evidence가 artifact_ids[0]만 보던 버그 수정)
                                                        # (v6: chart_artifact_ids가 숫자검증에 잘못 포함되던 버그 수정)
                                                        # (v5: 노드 종류별 의미적 구조(discriminated union)로 전면 개편)
                                                        # (v4: sql_plan 증거 추출 버그 수정)
_MAX_ATTEMPTS = 2                                      # 최초 1회 + 재시도 1회 (insight 8라운드 루프 아님)

_KIND_LABELS = {
    "sql_result": "SQL 조회",
    "sql_plan": "SQL 조회",
    "eda_summary": "EDA 검증",
    "analysis_result": "분석",
    "insight_payload": "인사이트",
}

# source_kind(아티팩트 종류) → detail_kind(스키마 종류). sql_plan/sql_result 둘 다 "SQL 단계"다.
_DETAIL_KIND = {
    "sql_plan": "sql",
    "sql_result": "sql",
    "eda_summary": "eda",
    "analysis_result": "analysis",
    "insight_payload": "insight",
}

# 폴백에서 raw dict key를 그대로 노출하지 않기 위한 사람이 읽는 라벨(Codex 리뷰 반영).
_FACT_LABELS = {
    "final_summary": "핵심 요약",
    "hypotheses": "가설",
    "primary_hypothesis": "1순위 가설",
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
    "method_decision": "방법론 선택 근거",
    "hypothesis_tests": "가설 검정",
    "answer": "핵심 답변",
    "key_insights": "핵심 인사이트",
    "action_plan": "실행 제안",
    "evidence_labels": "근거 출처",
}


def generate_node_summary(
    artifact_ids: list[str],
    runtime: AgentRuntime,
    *,
    branch_instruction: str | None = None,
    force_regenerate: bool = False,
) -> ArtifactRef:
    """branch_instruction: 분기(재분석) 시 이 단계에 추가로 반영된 지시사항.

    원본 파이프라인 실행에서는 항상 None이라 기존 동작에 영향이 없다. 분기 실행에서
    값이 있으면, 이 단계가 그 지시사항을 어떻게 반영했는지 결과 서술에 명시적으로
    드러나도록 프롬프트에 덧붙인다(_generate_with_llm 참고).
    """
    if not artifact_ids:
        raise ValueError("generate_node_summary는 artifact_ids가 최소 1개 필요합니다.")

    if not force_regenerate:
        cached = _find_cached(artifact_ids, runtime)
        if cached is not None:
            return cached

    evidence = read_node_evidence(artifact_ids, runtime)
    if branch_instruction and branch_instruction.strip():
        evidence.facts["branch_instruction"] = branch_instruction.strip()
    result = _generate_with_llm(evidence) if evidence.facts else None
    if result is None:
        result = _plain_narrative_result(evidence)
    if result is None:
        result = _fallback_result(evidence)

    return _register(artifact_ids, result, runtime)


def _find_cached(artifact_ids: list[str], runtime: AgentRuntime) -> ArtifactRef | None:
    """이미 만들어둔 같은 버전 요약이 있으면 그걸 그대로 쓴다(재생성 없음, LLM 호출 0).

    list_artifacts는 created_at 오름차순(오래된 것부터)이라, 매칭되는 게 여러 개면
    가장 나중 것(=최신)을 쓴다 — 재생성으로 더 나은 결과가 생겼는데도 옛날 폴백
    아티팩트를 계속 돌려주는 걸 방지한다.
    """
    anchor = runtime.adapter.get_artifact(artifact_ids[0])
    wanted = set(artifact_ids)
    latest: ArtifactRef | None = None
    for record in runtime.adapter.list_artifacts(run_id=anchor.run_id, artifact_type=ArtifactType.file):
        meta = record.metadata
        if meta.get("kind") != "node_summary":
            continue
        if set(meta.get("source_artifact_ids") or []) != wanted:
            continue
        if meta.get("summary_version") != _SUMMARY_VERSION:
            continue
        latest = record.ref()
    return latest


def _generate_with_llm(evidence: NodeEvidence) -> NodeSummaryResult | None:
    detail_kind = _DETAIL_KIND.get(evidence.source_kind)
    if detail_kind is None:
        return None

    llm = get_chat_model(model=os.getenv("SUMMARY_MODEL") or None, model_env="LLM_MODEL")
    known_chart_ids = {c.artifact_id for c in evidence.charts}
    build_prompt = _PROMPT_BUILDERS[detail_kind]
    parse_result = _RESULT_PARSERS[detail_kind]
    branch_instruction = str(evidence.facts.get("branch_instruction") or "").strip()

    feedback = ""
    for _ in range(_MAX_ATTEMPTS):
        try:
            prompt = build_prompt(evidence, feedback)
            if branch_instruction:
                prompt += (
                    "\n\n[분기 지시사항] 이 결과는 원본 분석에서 분기하여 다음 추가 지시사항을 "
                    f"반영해 재실행한 것이다: \"{branch_instruction}\"\n"
                    "이 지시사항을 어떻게 반영했는지 background 또는 conclusion에서 명시적으로 서술하라.\n"
                    "key_finding(전체를 압축한 한 문장)에도 이 지시사항이 결과에 영향을 줬는지 "
                    "짧게 드러내라 — 예: '~에도 관계는 유지되었다', '~로 제한하니 검정이 불가했다', "
                    "'~기준으로도 동일한 경향이 확인되었다'처럼, 지시사항 반영 여부와 결과가 "
                    "달라졌는지/유지됐는지를 한 문장 안에 압축하라."
                )
            raw = llm.invoke(prompt).content
        except Exception:  # noqa: BLE001 — LLM 호출 실패는 폴백으로 처리
            return None
        parsed = _parse_llm_json(raw)
        if parsed is None:
            feedback = '[형식 오류] JSON 하나만 출력하라. 지정된 필드를 모두 채워라.'
            continue

        result = parse_result(parsed, evidence, known_chart_ids)
        if result is None:
            feedback = "[누락] title/subtitle/background/conclusion/key_finding은 비울 수 없고, 핵심 리스트는 최소 1개 이상이어야 한다."
            continue

        return result
    return None


# ─────────────────────────────
# 공통 파싱 헬퍼
# ─────────────────────────────
def _extract_base(parsed: dict[str, Any]) -> dict[str, str] | None:
    title = str(parsed.get("title") or "").strip()
    subtitle = str(parsed.get("subtitle") or "").strip()
    background = str(parsed.get("background") or "").strip()
    conclusion = str(parsed.get("conclusion") or "").strip()
    key_finding = str(parsed.get("key_finding") or "").strip()
    if not title or not subtitle or not background or not conclusion or not key_finding:
        return None
    return {
        "title": title, "subtitle": subtitle, "background": background,
        "conclusion": conclusion, "key_finding": key_finding,
    }


def _extract_sections(raw: Any, known_chart_ids: set[str]) -> list[FindingSection]:
    if not isinstance(raw, list):
        return []
    out: list[FindingSection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading") or "").strip()
        body = str(item.get("body") or "").strip()
        if not heading or not body:
            continue
        chart_ids = [str(c) for c in (item.get("chart_artifact_ids") or []) if str(c) in known_chart_ids]
        out.append(FindingSection(
            heading=heading,
            rationale=str(item.get("rationale") or "").strip(),
            body=body,
            source_label=(str(item.get("source_label")).strip() or None) if item.get("source_label") else None,
            chart_artifact_ids=chart_ids,
        ))
    return out


def _extract_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _ensure_one_bold_phrase(text: str) -> str:
    """UI 강조가 완전히 빠진 경우 핵심 목적 어구 하나만 보정한다."""
    if not text or "**" in text:
        return text
    patterns = [
        r"([^.,\n]{2,30}?)(을|를) 위해",
        r"([^.,\n]{2,30}?)(이|가) 가능",
        r"([^.,\n]{2,30}?)(을|를) 가능",
        r"([^.,\n]{2,30}?)(으로|로) 구성",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        phrase = match.group(1).strip()
        if phrase:
            return text[:match.start(1)] + f"**{phrase}**" + text[match.end(1):]
    first_sentence = re.split(r"[.!?。]\s*|\n", text, maxsplit=1)[0]
    phrase = first_sentence[: min(len(first_sentence), 18)].strip()
    return text.replace(phrase, f"**{phrase}**", 1) if phrase else text


def _ensure_eda_finding_emphasis(text: str) -> str:
    """EDA 발견 본문에는 핵심 수치나 관계 신호가 한 번은 눈에 들어오게 한다."""
    if not text or "**" in text:
        return text
    patterns = [
        r"피어슨 상관 -?\d+(?:\.\d+)?, 스피어만 상관 -?\d+(?:\.\d+)?",
        r"\d+(?:\.\d+)?일에서 \d+(?:\.\d+)?일",
        r"\d{1,3}(?:,\d{3})*개 그룹",
        r"음의 관계",
        r"월별 편차",
        r"배송 속도 개선",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return text[:match.start()] + f"**{match.group(0)}**" + text[match.end():]
    return _ensure_one_bold_phrase(text)


def _ensure_analysis_emphasis(text: str) -> str:
    """분석 결과에는 핵심 검정 수치나 판정 경계를 한 번은 강조한다."""
    if not text or "**" in text:
        return text
    patterns = [
        r"p-value[= ]-?\d+(?:\.\d+)?(?:e[+-]?\d+)?",
        r"피어슨 상관 -?\d+(?:\.\d+)?",
        r"스피어만 상관 -?\d+(?:\.\d+)?",
        r"통계적으로 유의",
        r"통계적으로 확정하기 어렵습니다",
        r"음의 관계",
        r"유의미한 차이",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return text[:match.start()] + f"**{match.group(0)}**" + text[match.end():]
    return _ensure_one_bold_phrase(text)


def _quoted_korean_sentence(label: str, value: Any) -> str:
    text = str(value).strip().rstrip(".")
    return f"{label}은 '{text}'입니다." if text else ""


# ─────────────────────────────
# SQL — 정합성 확인 → 파생변수 생성 → 마트 설계
# ─────────────────────────────
def _build_sql_prompt(evidence: NodeEvidence, feedback: str) -> str:
    facts_text, _, feedback_section = _prompt_common_parts(evidence, feedback)
    return f"""
너는 SQL/마트 설계 결과를 설명하는 데이터 분석가다. 작업 순서를 일기처럼 나열하지 말고,
데이터 조건이 왜 특정 grain·조인·집계 선택으로 이어졌는지와 그 결과 어떤 분석이 가능해졌는지
짧은 보고서 문장으로 서술하라. 새로운 계산·추측이나 근거에 없는 이름·숫자는 만들지 마라.

[근거 종류] {evidence.source_kind}
[근거 내용] {facts_text}
{feedback_section}
[문체] "에이전트가 ~했습니다"를 반복하지 마라. 데이터와 분석 논리를 주어로 삼고, 앞 문장의
판단이 다음 문장의 방법 선택 이유가 되게 작성하라. 모든 문장은 보고서 톤의 존댓말
"~습니다/~됩니다/~하였습니다"로 끝내라. "~한다/~했다/~이다" 같은 평서체는 쓰지 마라.
강조가 필요한 핵심 판단, 분석 목적, 산출물 의미에만 Markdown bold 문법 **...**을 사용하라.
문장 전체나 모든 문장에 bold를 넣지 말고, 한 문단에 짧은 어구 1개만 강조하라.

이 단계는 SQL 에이전트의 결과다 — 고유 역할은 (1) 데이터 정합성·결측 확인, (2) 파생 컬럼
생성, (3) 최종 데이터마트 설계다. 아래 필드로 이 세 가지를 명확히 구분해서 채워라(근거에
정보가 부족한 항목은 빈 리스트로 남겨도 된다 — 억지로 만들지 마라).

[구조 — 반드시 이 필드로 JSON 출력]
- title: 이 마트를 나타내는 구체적인 제목(짧게)
- subtitle: 제목 아래 붙는 한 줄 태그라인
- background: 사용자 질문이 요구한 분석 판단과 이 마트가 필요한 이유를 함께 설명한다. 1문단, 최대 2문장.
- design_rationale: 데이터 특성 때문에 이 grain·조인·집계 방식을 선택한 이유. 1문단, 최대 3문장. 핵심 선택 이유 1개를 **...**로 강조한다.
- source_tables: 이 SQL이 사용한 원천 테이블명 리스트(근거에 등장한 것만)
- integrity_checks: 결과 해석에 영향을 주는 핵심 정합성 처리만 최대 2개
- derived_columns: 파생/계산된 컬럼 각각을 설명하는 섹션 리스트. 각 섹션:
  {{"heading":"컬럼명", "body":"정의·계산식·의도. 최대 2문장."}}
- mart_grain: 최종 마트의 행 단위(grain)를 한 문장으로
- mart_columns: 최종 마트에 포함된 컬럼명 리스트
- conclusion: 어떤 최종 데이터마트가 어떤 후속 분석을 가능하게 했는지 구체적으로 쓴다. 1문단, 최대 2문장. 핵심 산출물 의미 1개를 **...**로 강조한다.
- interpretation_scope: 이 마트를 해석할 때 주의할 범위. 최대 2개.
- handoff: 빈 문자열로 둔다.
- key_finding: 위 전체를 압축한 한 문장

근거에 등장한 이름·숫자만 인용하라.

JSON만 출력하라."""


def _to_sql_result(parsed: dict[str, Any], evidence: NodeEvidence, known_chart_ids: set[str]) -> NodeSummaryResult | None:
    base = _extract_base(parsed)
    if base is None:
        return None
    source_tables = _extract_str_list(parsed.get("source_tables"))
    integrity_checks = _extract_str_list(parsed.get("integrity_checks"))
    derived_columns = _extract_sections(parsed.get("derived_columns"), known_chart_ids)
    mart_columns = _extract_str_list(parsed.get("mart_columns"))
    if not (source_tables or integrity_checks or derived_columns or mart_columns):
        return None
    design_rationale = _ensure_one_bold_phrase(str(parsed.get("design_rationale") or "").strip())
    conclusion = _ensure_one_bold_phrase(base["conclusion"])
    detail = SQLSummaryDetail(
        design_rationale=design_rationale,
        source_tables=source_tables,
        integrity_checks=integrity_checks[:3],
        derived_columns=derived_columns,
        mart_grain=str(parsed.get("mart_grain") or "").strip(),
        mart_columns=mart_columns,
        mart_preview=list(evidence.facts.get("preview") or []),   # 근거 그대로(LLM이 안 씀)
        sql_snippet="",
        interpretation_scope=_extract_str_list(parsed.get("interpretation_scope"))[:2],
        handoff="",
    )
    return NodeSummaryResult(
        title=base["title"], subtitle=base["subtitle"], background=base["background"],
        code_used=evidence.code_used, detail=detail,
        conclusion=conclusion, key_finding=base["key_finding"],
        source_kind=evidence.source_kind, fallback_used=False,
    )


# ─────────────────────────────
# EDA — 컬럼 프로파일링 → 통계 탐색 → 차트 생성 → 가설 형성
# ─────────────────────────────
def _build_eda_prompt(evidence: NodeEvidence, feedback: str) -> str:
    facts_text, charts_text, feedback_section = _prompt_common_parts(evidence, feedback)
    return f"""
너는 EDA 결과를 설명하는 데이터 분석가다. 수행 항목을 나열하지 말고, 데이터 조건이 왜 특정
탐색 방법과 차트 선택으로 이어졌는지, 그 근거에서 무엇이 관찰되어 어떤 가설이 생겼는지를
연결해서 서술하라. 새로운 계산·추측이나 근거에 없는 숫자는 만들지 마라.

[근거 종류] {evidence.source_kind}
[근거 내용] {facts_text}
[사용 가능한 차트] {charts_text}
{feedback_section}
[문체] "EDA 에이전트가 ~했습니다"를 반복하지 마라. 데이터와 분석 내용을 주어로 삼아
선택 이유 → 실제 근거 → 관찰 결과가 자연스럽게 이어지게 작성하라. 모든 문장은 보고서 톤의
존댓말 "~습니다/~됩니다/~하였습니다"로 끝내라. "~한다/~했다/~이다" 같은 평서체는 쓰지 마라.
강조가 필요한 핵심 판단, 데이터 조건, 발견 의미에만 Markdown bold 문법 **...**을 사용하라.
문장 전체나 모든 문장에 bold를 넣지 말고, 한 문단에 짧은 어구 1개만 강조하라.

이 단계는 EDA 에이전트의 결과다 — 고유 역할은 (1) 컬럼 프로파일링·품질 확인, (2) 통계 탐색,
(3) 근거가 되는 차트 생성, (4) 가설 형성이다.

[구조 — 반드시 이 필드로 JSON 출력]
- title: 이 단계를 나타내는 구체적인 제목(짧게)
- subtitle: 제목 아래 붙는 한 줄 태그라인
- background: 왜 이 단계가 필요했는지(목적/배경)만. 1문단, 최대 2문장.
- data_profile: 컬럼 타입·카디널리티·결측 등 데이터 구조 요약. 1문단, 최대 3문장. 한눈에
  들어오도록 핵심 데이터 조건 1개를 **...**로 강조한다.
- quality_issues: 발견된 품질 이슈(문장 리스트)
- statistical_findings: 분포/상관/그룹비교 발견을 다루는 섹션 리스트(핵심 파트, 최소 1개
  이상). 각 섹션: {{"heading":"소제목", "rationale":"이 방법/차트가 필요한 이유",
  "body":"차트나 통계에서 관찰된 내용과 의미(구체 수치 인용)",
  "source_label":"분포 차트 등(선택)", "chart_artifact_ids":["[사용 가능한 차트]에 있는
  artifact_id 1개만"]}}. 차트 목록의 캡션을 보고 가장 대표적인 차트 1개만 연결하라.
  차트가 직접 뒷받침하지 않는 발견은 chart_artifact_ids를 빈 리스트로 둔다.
  body에는 핵심 수치나 관계 신호 1개를 **...**로 강조한다.
- hypotheses: 제안된 가설을 한 문장씩 요약한 리스트(근거의 가설 텍스트 기반). 각 문장은
  무엇을 검증할지와 왜 그 가설이 필요한지가 함께 드러나야 한다.
- interpretation_scope: 이 탐색 결과를 해석할 때 지켜야 할 범위. 최대 2개.
- handoff: 빈 문자열로 둔다.
- conclusion: 이 단계에서 얻은 결론. 1문단, 최대 2문장. 다음 분석에서 바로 검증해야 할
  구체적 방향 1개를 **...**로 강조한다.
- key_finding: 위 전체를 압축한 한 문장

근거에 등장한 숫자만 인용하라. [사용 가능한 차트] 목록에 없는 artifact_id는 만들지 마라.

JSON만 출력하라."""


def _to_eda_result(parsed: dict[str, Any], evidence: NodeEvidence, known_chart_ids: set[str]) -> NodeSummaryResult | None:
    base = _extract_base(parsed)
    if base is None:
        return None
    statistical_findings = _extract_sections(parsed.get("statistical_findings"), known_chart_ids)
    if not statistical_findings:
        return None
    statistical_findings = [
        section.model_copy(update={
            "body": _ensure_eda_finding_emphasis(section.body),
            "chart_artifact_ids": section.chart_artifact_ids[:1],
        })
        for section in statistical_findings
    ]
    data_profile = _ensure_one_bold_phrase(str(parsed.get("data_profile") or "").strip())
    conclusion = _ensure_one_bold_phrase(base["conclusion"])
    detail = EDASummaryDetail(
        data_profile=data_profile,
        quality_issues=_extract_str_list(parsed.get("quality_issues")),
        statistical_findings=statistical_findings,
        hypotheses=_extract_str_list(parsed.get("hypotheses")),
        primary_hypothesis=evidence.facts.get("primary_hypothesis") or {},   # 근거 그대로(LLM이 안 씀)
        interpretation_scope=_extract_str_list(parsed.get("interpretation_scope"))[:2],
        handoff="",
    )
    return NodeSummaryResult(
        title=base["title"], subtitle=base["subtitle"], background=_ensure_one_bold_phrase(base["background"]),
        code_used=evidence.code_used, detail=detail,
        conclusion=conclusion, key_finding=base["key_finding"],
        source_kind=evidence.source_kind, fallback_used=False,
    )


# ─────────────────────────────
# 분석 — 방법론 선택 → 가설 검정
# ─────────────────────────────
def _build_analysis_prompt(evidence: NodeEvidence, feedback: str) -> str:
    facts_text, _, feedback_section = _prompt_common_parts(evidence, feedback)
    return f"""
너는 통계 분석 결과를 설명하는 데이터 분석가다. 검정 목록을 일기처럼 나열하지 말고, EDA에서
생긴 질문과 데이터 조건이 왜 이 방법론 선택으로 이어졌는지, 여러 결과를 함께 읽으면 무엇을
의미하는지 설명하라. 새로운 계산·추측이나 근거에 없는 숫자는 만들지 마라.

[근거 종류] {evidence.source_kind}
[근거 내용] {facts_text}
{feedback_section}
[문체] "분석 에이전트가 ~했습니다"를 반복하지 마라. 방법론의 성질과 데이터 조건을 주어로
삼고, 방법 선택 → 검정 근거 → 결과 해석이 이어지게 작성하라. 모든 문장은 보고서 톤의
존댓말 "~습니다/~됩니다/~하였습니다"로 끝내라. "~한다/~했다/~이다" 같은 평서체는 쓰지 마라.
강조가 필요한 핵심 검정 수치, 판정, 해석 경계에만 Markdown bold 문법 **...**을 사용하라.
문장 전체나 모든 문장에 bold를 넣지 말고, 한 문단에 짧은 어구 1개만 강조하라.

이 단계는 분석 에이전트의 결과다 — 고유 역할은 (1) 통계 방법론 선택, (2) 가설 검정이다.
방법론 선택 근거(method_decision)는 이미 구조화된 근거로 별도 제공되니 새로 쓰지 마라 —
너는 검정 결과 서술에 집중하라.

[구조 — 반드시 이 필드로 JSON 출력]
- title: 이 분석을 나타내는 구체적인 제목(짧게)
- subtitle: 제목 아래 붙는 한 줄 태그라인
- background: 왜 이 분석이 필요했는지(목적/배경)만. 1문단, 최대 2문장.
- hypothesis_tests: 가설 검정 각각을 설명하는 섹션 리스트(핵심 파트, 최소 1개 이상). 각
  검정의 decision을 그대로 보존하라. decision이 inconclusive이면 관찰된 방향과 통계적
  확정을 구분하고, conclusion/key_finding에도 "경향은 관찰됐지만 통계적으로 확정하기
  어렵다"는 경계를 반드시 유지하라.
  섹션: {{"heading":"가설 요약", "rationale":"이 검정이 필요한 이유",
  "body":"H0/H1/판정(지지됨·지지안됨)과 근거(p-value 등)"}}
- key_statistics: 핵심 수치 근거를 다루는 섹션 리스트(선택, 없으면 빈 리스트)
- interpretation: 여러 검정과 실제 근거표를 함께 읽었을 때의 분석적 의미. 1문단, 최대 3문장.
- limitations: 이 분석의 한계(문장 리스트)
- handoff: Insight 단계에서 채택할 결론과 반드시 유지할 해석 경계를 한 문장으로.
- conclusion: 이 단계에서 얻은 결론. 1문단, 최대 2문장. 핵심 판정 또는 해석 경계 1개를
  **...**로 강조한다.
- key_finding: 위 전체를 압축한 한 문장

근거에 등장한 숫자만 인용하라.

JSON만 출력하라."""


def _ground_analysis_test_sections(
    generated: list[FindingSection], raw_tests: Any
) -> list[FindingSection]:
    """가설 판정과 수치는 LLM 표현 대신 Analysis 원본 계약에서 결정론적으로 옮긴다."""
    if not isinstance(raw_tests, list):
        return generated
    grounded: list[FindingSection] = []
    for index, raw in enumerate(raw_tests):
        if not isinstance(raw, dict):
            continue
        generated_section = generated[index] if index < len(generated) else None
        heading = str(
            raw.get("hypothesis")
            or raw.get("alternative_hypothesis")
            or (generated_section.heading if generated_section else "")
            or f"가설 검정 {index + 1}"
        ).strip()
        parts: list[str] = []
        if raw.get("null_hypothesis"):
            parts.append(_quoted_korean_sentence("귀무가설", raw["null_hypothesis"]))
        if raw.get("alternative_hypothesis"):
            parts.append(_quoted_korean_sentence("대립가설", raw["alternative_hypothesis"]))
        if raw.get("decision"):
            parts.append(_quoted_korean_sentence("원본 판정", raw["decision"]))
        evidence_parts: list[str] = []
        for field_name, label in (
            ("test_name", "검정"),
            ("statistic", "통계량"),
            ("p_value", "p-value"),
            ("effect_size", "효과크기"),
            ("n", "표본수"),
        ):
            value = raw.get(field_name)
            if value is not None and value != "":
                evidence_parts.append(f"{label}={value}")
        if evidence_parts:
            parts.append("근거는 " + ", ".join(evidence_parts) + "입니다.")
        caveats = raw.get("caveats") or []
        if isinstance(caveats, str):
            caveats = [caveats]
        if caveats:
            caveat_text = " ".join(str(item) for item in caveats if item).strip()
            if caveat_text:
                parts.append(_quoted_korean_sentence("주의사항", caveat_text))
        body = _ensure_analysis_emphasis(" ".join(parts) or (generated_section.body if generated_section else ""))
        grounded.append(FindingSection(
            heading=heading,
            rationale=generated_section.rationale if generated_section else "",
            body=body,
            source_label=generated_section.source_label if generated_section else None,
            chart_artifact_ids=generated_section.chart_artifact_ids if generated_section else [],
        ))
    return grounded or generated


def _analysis_decision_boundaries(raw_tests: Any) -> list[str]:
    if not isinstance(raw_tests, list):
        return []
    boundaries: list[str] = []
    for raw in raw_tests:
        if not isinstance(raw, dict) or str(raw.get("decision") or "").lower() != "inconclusive":
            continue
        hypothesis = str(raw.get("hypothesis") or raw.get("alternative_hypothesis") or "해당 가설")
        p_value = raw.get("p_value")
        suffix = f" (p-value={p_value})" if p_value is not None else ""
        boundaries.append(
            f"가설 '{hypothesis}'은 관찰된 방향성이 있더라도 원본 판정이 inconclusive이므로 "
            f"통계적으로 확정하기 어렵습니다{suffix}."
        )
    return boundaries


def _analysis_key_numbers(raw: dict[str, Any]) -> list[str]:
    numbers: list[str] = []
    for field_name, label in (
        ("statistic", "statistic"),
        ("p_value", "p-value"),
        ("effect_size", "effect size"),
        ("n", "n"),
    ):
        value = raw.get(field_name)
        if value is not None and value != "":
            numbers.append(f"{label}={value}")
    return numbers


def _analysis_caution(raw: dict[str, Any]) -> str:
    caveats = raw.get("caveats") or []
    if isinstance(caveats, str):
        caveats = [caveats]
    return " ".join(str(item).strip() for item in caveats if str(item).strip())


def _analysis_result_sentence(raw: dict[str, Any]) -> str:
    decision = str(raw.get("decision") or "").strip()
    if not decision:
        return ""
    test_name = str(raw.get("test_name") or "").strip()
    if test_name:
        return f"{test_name} 결과의 원본 판정은 {decision}입니다."
    return f"원본 판정은 {decision}입니다."


def _analysis_item_title(raw: dict[str, Any], generated: FindingSection | None, index: int) -> str:
    return str(
        raw.get("hypothesis")
        or raw.get("alternative_hypothesis")
        or (generated.heading if generated else "")
        or f"본 분석 {index + 1}"
    ).strip()


def _analysis_item_purpose(raw: dict[str, Any], generated: FindingSection | None, index: int) -> str:
    if generated and generated.rationale.strip():
        return generated.rationale.strip()

    title = _analysis_item_title(raw, generated, index)
    method = str(raw.get("test_name") or "").strip()
    if method and title:
        return f"이 항목은 {title} 항목의 판단 근거를 {method} 방법으로 확인했습니다."
    if title:
        return f"이 항목은 {title} 항목의 판단 근거를 확인하기 위해 분석했습니다."
    if method:
        return f"{method}을 사용해 핵심 관계와 판정 근거를 확인했습니다."
    return "이 분석 항목의 방향성과 판단 근거를 확인했습니다."


def _analysis_items_from_tests(
    *,
    raw_tests: Any,
    generated_sections: list[FindingSection],
) -> list[AnalysisItem]:
    items: list[AnalysisItem] = []
    if isinstance(raw_tests, list):
        for index, raw in enumerate(raw_tests):
            if not isinstance(raw, dict):
                continue
            generated = generated_sections[index] if index < len(generated_sections) else None
            item = AnalysisItem(
                title=_analysis_item_title(raw, generated, index),
                method=str(raw.get("test_name") or "").strip(),
                purpose=_analysis_item_purpose(raw, generated, index),
                result=_analysis_result_sentence(raw),
                key_numbers=_analysis_key_numbers(raw),
                interpretation="",
                caution=_analysis_caution(raw),
                decision=str(raw.get("decision") or "").strip(),
            )
            items.append(item)

    if items:
        return items

    return [
        AnalysisItem(
            title=section.heading,
            purpose=section.rationale,
            result=section.body,
            interpretation=section.body,
        )
        for section in generated_sections
    ]


def _ensure_insight_sentence(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    value = (
        value
        .replace("보여준다.", "보여줍니다.")
        .replace("나타난다.", "나타납니다.")
        .replace("보인다.", "보입니다.")
        .replace("시사한다.", "시사합니다.")
        .replace("지지한다.", "지지합니다.")
        .replace("맞다.", "맞습니다.")
        .replace("했다.", "했습니다.")
        .replace("한다.", "합니다.")
        .replace("이다.", "입니다.")
    )
    if re.search(r"(습니다|됩니다|하였습니다|입니다|였습니다)\.$", value):
        return value
    if value.endswith("됨"):
        return f"{value[:-1]}됩니다."
    if re.search(r"[.!?]$", value):
        return value
    return f"{value}입니다."


def _build_insight_to_be(parsed: dict[str, Any]) -> str:
    to_be = str(parsed.get("to_be") or "").strip()
    if len([part for part in re.split(r"(?<=[.!?。])\s+", to_be) if part.strip()]) >= 3:
        return to_be

    action_plan = _extract_str_list(parsed.get("action_plan"))
    seeds = [to_be]
    if action_plan:
        seeds.append(action_plan[0])
    seeds.extend([
        "우선 점검 대상과 추적 지표를 분리해 운영 변화가 실제로 유지되는지 확인해야 합니다.",
        "결과 해석은 상관 수준의 근거로 제한하고, 후속 검증에서 같은 방향이 유지되는지 확인해야 합니다.",
    ])

    sentences: list[str] = []
    for seed in seeds:
        text = _ensure_insight_sentence(seed.strip())
        if not text:
            continue
        if text not in sentences:
            sentences.append(text)
        if len(sentences) >= 3:
            break
    return " ".join(sentences)


def _emphasize_sections(sections: list[FindingSection]) -> list[FindingSection]:
    return [
        section.model_copy(update={"body": _ensure_analysis_emphasis(section.body)})
        for section in sections
    ]


def _to_analysis_result(parsed: dict[str, Any], evidence: NodeEvidence, known_chart_ids: set[str]) -> NodeSummaryResult | None:
    base = _extract_base(parsed)
    if base is None:
        return None
    hypothesis_tests = _extract_sections(parsed.get("hypothesis_tests"), known_chart_ids)
    if not hypothesis_tests:
        return None
    raw_tests = evidence.facts.get("hypothesis_tests") or []
    hypothesis_tests = _ground_analysis_test_sections(hypothesis_tests, raw_tests)
    decision_boundaries = _analysis_decision_boundaries(raw_tests)
    method_decision = evidence.facts.get("method_decision") or _infer_analysis_method_decision(evidence.facts)
    interpretation_parts = [
        _ensure_analysis_emphasis(str(parsed.get("interpretation") or "").strip()),
        *(_ensure_analysis_emphasis(boundary) for boundary in decision_boundaries),
    ]
    detail = AnalysisSummaryDetail(
        method_decision=method_decision,
        analysis_items=_analysis_items_from_tests(
            raw_tests=raw_tests,
            generated_sections=hypothesis_tests,
        ),
        hypothesis_tests=hypothesis_tests,
        key_statistics=_emphasize_sections(_extract_sections(parsed.get("key_statistics"), known_chart_ids)),
        evidence_tables=[
            EvidenceTable.model_validate(table)
            for table in (evidence.facts.get("evidence_tables") or [])
        ],
        supporting_charts=_charts_to_sections(evidence),
        interpretation="\n\n".join(part for part in interpretation_parts if part),
        limitations=[*_extract_str_list(parsed.get("limitations")), *decision_boundaries],
        handoff=str(parsed.get("handoff") or "").strip(),
    )
    conclusion = _ensure_analysis_emphasis(base["conclusion"])
    key_finding = base["key_finding"]
    if decision_boundaries:
        conclusion = "\n\n".join([conclusion, *(_ensure_analysis_emphasis(boundary) for boundary in decision_boundaries)])
        key_finding = f"{key_finding} 다만 일부 추세는 통계적으로 확정되지 않았습니다."
    return NodeSummaryResult(
        title=base["title"], subtitle=base["subtitle"], background=_ensure_analysis_emphasis(base["background"]),
        code_used=evidence.code_used, detail=detail,
        conclusion=conclusion, key_finding=key_finding,
        source_kind=evidence.source_kind, fallback_used=False,
    )


def _infer_analysis_method_decision(facts: dict[str, Any]) -> dict[str, Any]:
    tests = facts.get("hypothesis_tests") or []
    selected_method = ""
    if isinstance(tests, list):
        for test in tests:
            if not isinstance(test, dict):
                continue
            selected_method = str(test.get("test_name") or "").strip()
            if selected_method:
                break

    rationale_candidates: list[str] = []
    summary = str(facts.get("executive_summary") or "").strip()
    if summary:
        rationale_candidates.append(summary)
    for note in facts.get("method_notes") or []:
        text = str(note or "").strip()
        if text:
            rationale_candidates.append(text)
    if not selected_method and not rationale_candidates:
        return {}
    return {
        "selected_method": selected_method or "unspecified_analysis_method",
        "rationale": rationale_candidates[0] if rationale_candidates else "상류 분석 아티팩트에서 방법 선택 근거를 복원했습니다.",
    }


# ─────────────────────────────
# 인사이트 — 근거 종합 → 직답
# ─────────────────────────────
def _build_insight_prompt(evidence: NodeEvidence, feedback: str) -> str:
    facts_text, charts_text, feedback_section = _prompt_common_parts(evidence, feedback)
    return f"""
너는 상류 분석 근거를 사용자 의사결정으로 압축하는 데이터 분석가다. SQL/EDA/분석 내용을
다시 길게 요약하지 말고, 사용자 질문에 대한 최종 답과 현재 상태(AS-IS), 지향 방향(TO-BE),
실행 제안(ACTION)을 짧고 구체적으로 정리하라. 새로운 계산·추측이나 근거 없는 숫자는 만들지 마라.
화면에는 AS-IS/TO-BE/ACTION이 같은 위계로 표시된다. answer와 as_is를 같은 말로 반복하지 말고,
as_is는 현재 상태, to_be는 관리 방향, action_plan은 그 방향으로 가기 위한 행동만 담아라.
TO-BE는 단순히 "관리해야 한다"로 끝내지 말고, AS-IS에서 확인된 문제를 어떤 운영 상태로 바꾸어야
하는지 3~4문장으로 구체화하라.

[근거 종류] {evidence.source_kind}
[근거 내용] {facts_text}
[사용 가능한 차트] {charts_text}
{feedback_section}
[문체] "인사이트 에이전트가 ~했습니다"를 반복하지 마라. 모든 문장은 보고서 톤의 존댓말
"~습니다/~됩니다/~하였습니다"로 끝내라. "~한다/~했다/~이다" 같은 평서체는 쓰지 마라.
인사이트는 최종 압축 단계이므로 같은 근거를 evidence_synthesis, answer, conclusion에서 반복하지 마라.

이 단계는 인사이트 에이전트의 결과다 — 고유 역할은 상류 근거(SQL/EDA/분석)를 종합해 사용자
질문에 직접 답하는 것이다. 근거 출처(evidence_sources)는 이미 구조화된 근거로 별도 제공되니
새로 쓰지 마라.

[구조 — 반드시 이 필드로 JSON 출력]
- title: 이 답변을 나타내는 구체적인 제목(짧게)
- subtitle: 제목 아래 붙는 한 줄 태그라인. 반드시 "~습니다/~됩니다" 존댓말 문장으로 쓴다.
- background: 사용자가 무엇을 물었는지(질문 배경)만. 1문단, 최대 2문장.
- evidence_synthesis: 내부 근거 요약이다. 결론을 지지하는 핵심 근거만 1문장으로 압축한다. 숫자는 가장 중요한 1~3개만 쓴다.
- answer: 사용자 질문에 대한 직접적인 답변이다. 최대 1문장. 화면의 AS-IS와 같은 문장을 반복하지 마라.
- as_is: AS-IS. "분석으로 확인한 현재 상태"만 1~2문장으로 쓴다. 핵심 수치가 있으면 여기만 넣는다.
- to_be: TO-BE. "이 결과가 가리키는 지향 방향"을 3~4문장으로 쓴다. 반드시 채워라.
  첫 문장은 지향 상태, 둘째 문장은 우선 관리 대상, 셋째 문장은 모니터링/검증 방향을 설명한다.
  근거에 없는 목표 숫자는 만들지 마라.
- key_insights: 핵심 통찰 리스트. 최대 2개.
- action_plan: ACTION. TO-BE를 달성하기 위한 실행 제안 리스트. 최대 3개. 각 항목은 구체적 행동 동사로 시작한다.
- limitations: 주의사항 리스트. 최대 2개.
- conclusion: answer와 반복하지 말고 최종 활용 경계만 1문장으로 쓴다.
- key_finding: 위 전체를 압축한 한 문장

근거에 등장한 숫자만 인용하라.

JSON만 출력하라."""


def _to_insight_result(parsed: dict[str, Any], evidence: NodeEvidence, known_chart_ids: set[str]) -> NodeSummaryResult | None:
    base = _extract_base(parsed)
    if base is None:
        return None
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        return None
    detail = InsightSummaryDetail(
        evidence_synthesis=str(parsed.get("evidence_synthesis") or "").strip(),
        answer=answer,
        as_is=str(parsed.get("as_is") or parsed.get("evidence_synthesis") or "").strip(),
        to_be=_build_insight_to_be(parsed),
        key_insights=_extract_str_list(parsed.get("key_insights")),
        action_plan=_extract_str_list(parsed.get("action_plan")),
        evidence_sources=[str(x) for x in (evidence.facts.get("evidence_labels") or [])],   # 근거 그대로
        limitations=_extract_str_list(parsed.get("limitations")),
        supporting_charts=_charts_to_sections(evidence),   # 근거 그대로(LLM이 안 씀)
    )
    return NodeSummaryResult(
        title=base["title"], subtitle=base["subtitle"], background=base["background"],
        code_used=evidence.code_used, detail=detail,
        conclusion=base["conclusion"], key_finding=base["key_finding"],
        source_kind=evidence.source_kind, fallback_used=False,
    )


_PROMPT_BUILDERS: dict[str, Callable[[NodeEvidence, str], str]] = {
    "sql": _build_sql_prompt,
    "eda": _build_eda_prompt,
    "analysis": _build_analysis_prompt,
    "insight": _build_insight_prompt,
}
_RESULT_PARSERS: dict[str, Callable[[dict[str, Any], NodeEvidence, set[str]], NodeSummaryResult | None]] = {
    "sql": _to_sql_result,
    "eda": _to_eda_result,
    "analysis": _to_analysis_result,
    "insight": _to_insight_result,
}


def _prompt_common_parts(evidence: NodeEvidence, feedback: str) -> tuple[str, str, str]:
    facts_text = json.dumps(evidence.facts, ensure_ascii=False, default=str)[:3000]
    charts_text = json.dumps(
        [{"artifact_id": c.artifact_id, "caption": c.caption} for c in evidence.charts],
        ensure_ascii=False,
    )
    feedback_section = f"\n[이전 시도 피드백]\n{feedback}\n" if feedback else ""
    return facts_text, charts_text, feedback_section


# ─────────────────────────────
# 2차 폴백 — 구조화 JSON이 실패했을 때, 형식 요구를 낮춰 LLM에 한 번 더 맡긴다(2026-07-23).
# 서사(제목/배경/결론)와 각 fact 섹션 본문을 전부 같은 호출·같은 목소리로 받아, "사람이 쓴
# 서사 + 기계가 덤프한 raw 값"처럼 톤이 갈리지 않게 한다. 라벨 하나를 LLM이 빼먹으면 그
# 섹션만 원본 값으로 메운다(전체를 정적 템플릿으로 버리지 않는다).
# ─────────────────────────────
def _plain_narrative_result(evidence: NodeEvidence) -> NodeSummaryResult | None:
    detail_kind = _DETAIL_KIND.get(evidence.source_kind)
    if detail_kind is None:
        return None
    label = _KIND_LABELS.get(evidence.source_kind, evidence.source_kind or "알 수 없는 단계")

    fact_items = [(k, v) for k, v in evidence.facts.items() if v]
    line_specs: list[tuple[str, str]] = [
        ("제목", "8자 내외 소제목"),
        ("한줄요약", "핵심을 압축한 1문장"),
        ("배경", "왜 이 단계가 필요했는지, 1~2문장"),
    ]
    fact_labels = [_FACT_LABELS.get(k, k) for k, _ in fact_items]
    line_specs += [(lbl, "이 항목 원본 값을 보고 자연스러운 문장 1~2개, 숫자는 있는 그대로 인용")
                   for lbl in fact_labels]
    if evidence.charts:
        line_specs.append(("관련 차트", "이 차트들이 보여주는 내용을 문장으로"))
    line_specs.append(("결론", "1~2문장"))

    llm = get_chat_model(model=os.getenv("SUMMARY_MODEL") or None, model_env="LLM_MODEL")
    facts_text, charts_text, _ = _prompt_common_parts(evidence, "")
    parsed = generate_plain_narrative(
        llm, label=label, line_specs=line_specs, evidence_text=facts_text, charts_text=charts_text)
    if parsed is None:
        return None

    catch_all = parsed.get("__all__", "")
    sections: list[FindingSection] = []
    for (key, value), lbl in zip(fact_items, fact_labels):
        body = parsed.get(lbl) or _format_fact_value(value)
        sections.append(FindingSection(heading=lbl, body=body, source_label=lbl, chart_artifact_ids=[]))
    if evidence.charts:
        chart_body = parsed.get("관련 차트") or "다음 차트가 이 단계의 근거로 함께 제공됩니다."
        sections.append(FindingSection(heading="관련 차트", body=chart_body, source_label="차트",
                                       chart_artifact_ids=[c.artifact_id for c in evidence.charts]))

    if detail_kind == "sql":
        detail: Any = SQLSummaryDetail(derived_columns=sections,
                                        mart_preview=list(evidence.facts.get("preview") or []),
                                        sql_snippet="")
    elif detail_kind == "eda":
        detail = EDASummaryDetail(statistical_findings=sections,
                                   primary_hypothesis=evidence.facts.get("primary_hypothesis") or {})
    elif detail_kind == "analysis":
        detail = AnalysisSummaryDetail(hypothesis_tests=sections,
                                        method_decision=evidence.facts.get("method_decision") or {},
                                        limitations=[str(x) for x in (evidence.facts.get("limitations") or [])])
    else:
        answer = parsed.get("배경") or catch_all
        detail = InsightSummaryDetail(
            answer=answer, as_is=answer,
            key_insights=[str(x) for x in (evidence.facts.get("key_insights") or [])],
            action_plan=[str(x) for x in (evidence.facts.get("action_plan") or [])],
            evidence_sources=[str(x) for x in (evidence.facts.get("evidence_labels") or [])],
            limitations=[str(x) for x in (evidence.facts.get("limitations") or [])])

    background = parsed.get("배경") or catch_all or f"이 단계는 {label} 아티팩트입니다."
    return NodeSummaryResult(
        title=parsed.get("제목") or f"{label} 결과",
        subtitle=parsed.get("한줄요약") or background[:60],
        background=background,
        code_used=evidence.code_used,
        detail=detail,
        conclusion=parsed.get("결론") or catch_all or background,
        key_finding=parsed.get("한줄요약") or background[:80],
        source_kind=evidence.source_kind,
        fallback_used=True,
    )


# ─────────────────────────────
# 폴백 — LLM 실패/검증 실패 시 숫자를 지어내지 않고 근거 그대로 최소 필드를 보장한다.
# ─────────────────────────────
def _fallback_result(evidence: NodeEvidence) -> NodeSummaryResult:
    label = _KIND_LABELS.get(evidence.source_kind, evidence.source_kind or "알 수 없는 단계")
    detail_kind = _DETAIL_KIND.get(evidence.source_kind)
    generic_sections = _facts_to_sections(evidence)

    if detail_kind == "sql":
        detail: Any = SQLSummaryDetail(
            derived_columns=generic_sections,
            mart_preview=list(evidence.facts.get("preview") or []),
            sql_snippet="",
        )
    elif detail_kind == "eda":
        detail = EDASummaryDetail(
            statistical_findings=generic_sections or [FindingSection(
                heading=f"{label} 결과", body="근거 데이터가 비어 있습니다.", chart_artifact_ids=[])],
            primary_hypothesis=evidence.facts.get("primary_hypothesis") or {},
        )
    elif detail_kind == "analysis":
        detail = AnalysisSummaryDetail(
            hypothesis_tests=generic_sections or [FindingSection(
                heading=f"{label} 결과", body="근거 데이터가 비어 있습니다.", chart_artifact_ids=[])],
            method_decision=evidence.facts.get("method_decision") or {},
            limitations=[str(x) for x in (evidence.facts.get("limitations") or [])],
        )
    elif detail_kind == "insight":
        answer = str(evidence.facts.get("answer") or "자동 요약 생성에 실패했습니다.")
        detail = InsightSummaryDetail(
            answer=answer,
            as_is=answer,
            to_be="근거가 보강된 뒤 실행 방향을 판단해야 합니다.",
            key_insights=[str(x) for x in (evidence.facts.get("key_insights") or [])],
            action_plan=[str(x) for x in (evidence.facts.get("action_plan") or [])],
            evidence_sources=[str(x) for x in (evidence.facts.get("evidence_labels") or [])],
            limitations=[str(x) for x in (evidence.facts.get("limitations") or [])],
        )
    else:
        # 알 수 없는 근거 종류 — insight 모양의 최소 폴백으로 채운다(아무것도 안 만드는 것보단 낫다).
        detail = InsightSummaryDetail(answer="근거 종류를 인식할 수 없어 자동 요약을 생성하지 못했습니다.")

    return NodeSummaryResult(
        title=f"{label} 결과",
        subtitle="자동 요약 생성에 실패해 근거 데이터를 그대로 안내합니다.",
        background=f"이 단계는 {label} 아티팩트입니다. 자동 서술 생성에 실패해 근거 원본을 정리해 보여드립니다.",
        code_used=evidence.code_used,
        detail=detail,
        conclusion="자동 서술 생성에 실패했습니다. 위 항목의 원본 데이터를 참고하세요.",
        key_finding=f"{label} 단계 완료 — 상세는 근거 데이터를 참고하세요.",
        source_kind=evidence.source_kind,
        fallback_used=True,
    )


def _charts_to_sections(evidence: NodeEvidence) -> list[FindingSection]:
    """근거의 차트를 그대로 섹션으로 변환한다(LLM이 안 씀 — caption은 이미 상류에서 확정됨)."""
    return [
        FindingSection(heading=c.caption or c.artifact_id, body=c.caption or "", chart_artifact_ids=[c.artifact_id])
        for c in evidence.charts
    ]


def _facts_to_sections(evidence: NodeEvidence) -> list[FindingSection]:
    sections: list[FindingSection] = []
    for key, value in evidence.facts.items():
        if not value:
            continue
        section_label = _FACT_LABELS.get(key, key)
        sections.append(FindingSection(
            heading=section_label,
            body=_format_fact_value(value),
            source_label=section_label,
            chart_artifact_ids=[],
        ))
    if evidence.charts:
        sections.append(FindingSection(
            heading="관련 차트",
            body="자동 요약 생성에 실패해 관련 차트만 안내합니다.",
            source_label="차트",
            chart_artifact_ids=[c.artifact_id for c in evidence.charts],
        ))
    return sections


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
    markdown_artifact_id: str | None = None
    try:
        markdown_ref = runtime.adapter.register_artifact(
            anchor.run_id,
            ArtifactType.report,
            content_text=render_node_summary_artifact_markdown(result),
            filename="node_summary.md",
            created_by_tool=_TOOL_NAME,
            parent_ids=artifact_ids,
            metadata={
                "kind": "node_summary_markdown",
                "source_artifact_ids": sorted(artifact_ids),
                "summary_version": _SUMMARY_VERSION,
            },
            preview={
                "title": result.title,
                "source_kind": result.source_kind,
                "format": "markdown",
            },
        )
        markdown_artifact_id = markdown_ref.artifact_id
    except Exception:  # noqa: BLE001 - Markdown 실패가 노드 완료와 JSON 요약 저장을 막지 않는다.
        markdown_artifact_id = None

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
            "markdown_artifact_id": markdown_artifact_id,
            "fallback_used": result.fallback_used,
        },
        preview={
            "title": result.title,
            "subtitle": result.subtitle,
            "key_finding": result.key_finding,
            "source_kind": result.source_kind,
            "fallback_used": result.fallback_used,
            "markdown_artifact_id": markdown_artifact_id,
        },
    )


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
