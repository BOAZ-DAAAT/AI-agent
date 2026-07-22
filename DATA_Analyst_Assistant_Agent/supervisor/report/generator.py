"""리포트 생성기 — API/버튼 레이어가 나중에 호출할 유일한 진입점: generate_report.

summary/generator.py와 같은 설계 원칙을 따른다:
- API/비동기 실행을 전혀 가정하지 않는 순수 함수. artifact_ids(경로 전체) 주면 리포트
  아티팩트 하나를 만들어(또는 캐시에서 찾아) ArtifactRef를 리턴한다.
- 숫자 검증은 shared/numeric_verify.py(공용 유틸)를 쓴다. 검증 실패 시 1회만 재시도,
  그래도 실패하면 LLM 없이 근거 그대로 채우는 템플릿 폴백(절대 숫자를 지어내지 않는다).

summary와 다른 점은 "노드 하나"가 아니라 "경로 전체(존재하는 단계만)"를 한 번의 LLM
호출로 하나의 흐름 있는 글로 종합한다는 것 — 단계별 나열("SQL 단계에서는~")은 명시적으로
금지하고, 실제로 나열식으로 나오면 구조 검증 실패로 보고 재시도 피드백을 준다.
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
from DATA_Analyst_Assistant_Agent.supervisor.report.evidence import PathEvidence, read_path_evidence
from DATA_Analyst_Assistant_Agent.supervisor.report.schemas import ReportResult, FindingSection

_TOOL_NAME = "supervisor.report.generator"
_REPORT_KIND = "report"
_REPORT_VERSION = 4                                    # 프롬프트/스키마 바뀌면 올려서 옛 캐시 무효화
_DEFAULT_REPORT_MAX_TOKENS = 12288
_MAX_ATTEMPTS = 2                                      # 최초 1회 + 재시도 1회

_STAGE_LABELS = {"sql": "SQL 조회", "eda": "EDA 검증", "analysis": "분석", "insight": "인사이트"}

# 나열식 감지용 — "SQL 단계에서는", "1단계에서는" 같은 문단 시작을 잡는다(대소문자 무시).
_LISTING_PREFIXES = (
    "sql 단계", "eda 단계", "분석 단계", "인사이트 단계",
    "1단계", "2단계", "3단계", "4단계",
)


def generate_report(artifact_ids: list[str], runtime: AgentRuntime) -> ArtifactRef:
    if not artifact_ids:
        raise ValueError("generate_report는 artifact_ids가 최소 1개 필요합니다.")

    cached = _find_cached(artifact_ids, runtime)
    if cached is not None:
        return cached

    evidence = read_path_evidence(artifact_ids, runtime)
    result = _generate_with_llm(evidence) if evidence.stages else None
    if result is None:
        result = _fallback_result(evidence)

    return _register(artifact_ids, result, runtime)


def _find_cached(artifact_ids: list[str], runtime: AgentRuntime) -> ArtifactRef | None:
    """이미 만들어둔 같은 버전 리포트가 있으면 그걸 그대로 쓴다(재생성 없음, LLM 호출 0)."""
    anchor = runtime.adapter.get_artifact(artifact_ids[0])
    wanted = set(artifact_ids)
    for record in runtime.adapter.list_artifacts(run_id=anchor.run_id, artifact_type=ArtifactType.file):
        meta = record.metadata
        if meta.get("kind") != _REPORT_KIND:
            continue
        if set(meta.get("source_artifact_ids") or []) != wanted:
            continue
        if meta.get("report_version") != _REPORT_VERSION:
            continue
        return record.ref()
    return None


def _generate_with_llm(evidence: PathEvidence) -> ReportResult | None:
    llm = get_chat_model(
        model=os.getenv("REPORT_MODEL") or None,
        model_env="LLM_MODEL",
        max_tokens=int(os.getenv("REPORT_MAX_TOKENS", str(_DEFAULT_REPORT_MAX_TOKENS))),
    )
    facts_by_stage = {stage.stage: stage.facts for stage in evidence.stages}
    numbers = collect_numbers(facts_by_stage)
    corpus = json.dumps(facts_by_stage, ensure_ascii=False, default=str)
    known_chart_ids = {c.artifact_id for c in evidence.charts}

    feedback = ""
    for _ in range(_MAX_ATTEMPTS):
        try:
            raw = llm.invoke(_build_prompt(evidence, feedback)).content
        except Exception:  # noqa: BLE001 — LLM 호출 실패는 폴백으로 처리
            return None
        parsed = _parse_llm_json(raw)
        if parsed is None:
            feedback = (
                "[형식 오류] JSON 하나만 출력하라. 지정된 필드(title/executive_summary/"
                "background_and_question/methodology_narrative/key_findings/limitations/"
                "conclusion_and_recommendations/key_finding)를 모두 채워라."
            )
            continue

        result = _to_result(parsed, evidence, known_chart_ids)
        if result is None:
            feedback = "[누락] 필수 필드가 비어있거나 key_findings가 최소 1개 이상이어야 한다."
            continue

        narrative_texts = [
            result.executive_summary,
            result.background_and_question,
            result.methodology_narrative,
            result.conclusion_and_recommendations,
            *(f.body for f in result.key_findings),
        ]
        if _looks_like_listing(narrative_texts):
            feedback = (
                '[구조 실패] "SQL 단계에서는~", "EDA 단계에서는~" 같이 순서대로 나열하는 '
                "문장이 감지됐다. 이렇게 쓰지 말고, 단계 간 인과관계를 따라 하나의 이야기로 다시 써라."
            )
            continue

        ok, missing = verify_texts(narrative_texts, numbers, corpus)
        if not ok:
            feedback = f"[검증 실패] 다음 숫자가 근거에 없다: {missing} — 근거에 있는 숫자만 써라. 새 숫자가 필요하면 쓰지 말고 서술만 하라."
            continue

        return result
    return None


def _looks_like_listing(texts: list[str]) -> bool:
    hits = 0
    for text in texts:
        normalized = (text or "").strip().lower()
        if normalized.startswith(_LISTING_PREFIXES):
            hits += 1
    return hits >= 2


def _to_result(parsed: dict[str, Any], evidence: PathEvidence, known_chart_ids: set[str]) -> ReportResult | None:
    title = str(parsed.get("title") or "").strip()
    executive_summary = str(parsed.get("executive_summary") or "").strip()
    background_and_question = str(parsed.get("background_and_question") or "").strip()
    methodology_narrative = str(parsed.get("methodology_narrative") or "").strip()
    conclusion = str(parsed.get("conclusion_and_recommendations") or "").strip()
    key_finding = str(parsed.get("key_finding") or "").strip()
    limitations = [str(x).strip() for x in (parsed.get("limitations") or []) if str(x).strip()]

    raw_findings = parsed.get("key_findings")
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

    if not title or not executive_summary or not background_and_question or not methodology_narrative or not conclusion or not key_finding or not findings:
        return None

    code_used = next((s.code_used for s in evidence.stages if s.code_used), "")

    return ReportResult(
        title=title,
        executive_summary=executive_summary,
        background_and_question=background_and_question,
        methodology_narrative=methodology_narrative,
        code_used=code_used,
        key_findings=findings,
        limitations=limitations,
        conclusion_and_recommendations=conclusion,
        key_finding=key_finding,
        included_stages=evidence.present_stage_names,
        fallback_used=False,
    )


def _fallback_result(evidence: PathEvidence) -> ReportResult:
    """LLM 실패/검증 실패 시 — 숫자를 지어내지 않고 근거 그대로 단계별로 나열한다.

    (여기서는 나열식이 되는 걸 감수한다 — 폴백은 애초에 "정제된 글쓰기"가 목적이 아니라
    "실패해도 근거를 안전하게 보여주기"가 목적이라 fallback_used=True로 구분해서 표시한다.)
    """
    findings: list[FindingSection] = []
    for stage in evidence.stages:
        for key, value in stage.facts.items():
            if not value:
                continue
            findings.append(FindingSection(
                heading=f"{stage.label} — {key}",
                body=_format_fact_value(value),
                source_label=stage.label,
                chart_artifact_ids=[],
            ))
        if stage.charts:
            findings.append(FindingSection(
                heading=f"{stage.label} 관련 차트",
                body="자동 리포트 생성에 실패해 관련 차트만 안내합니다.",
                source_label=stage.label,
                chart_artifact_ids=[c.artifact_id for c in stage.charts],
            ))
    if not findings:
        findings.append(FindingSection(heading="근거 없음", body="경로에 유효한 근거 데이터가 없습니다.", chart_artifact_ids=[]))

    stage_labels = [s.label for s in evidence.stages] or ["알 수 없음"]
    code_used = next((s.code_used for s in evidence.stages if s.code_used), "")
    journey = " → ".join(stage_labels)

    return ReportResult(
        title="분석 여정 리포트",
        executive_summary="자동 리포트 생성에 실패해 근거 데이터를 단계별로 그대로 안내합니다.",
        background_and_question=f"이 리포트는 {journey} 경로의 근거를 담고 있습니다.",
        methodology_narrative="자동 서술 생성에 실패해 이 섹션은 생략합니다.",
        code_used=code_used,
        key_findings=findings,
        limitations=["자동 생성에 실패해 근거 원본을 그대로 정리했습니다."],
        conclusion_and_recommendations="자동 결론 생성에 실패했습니다. 위 항목의 원본 데이터를 참고하세요.",
        key_finding=f"{journey} 경로 완료 — 상세는 근거 데이터를 참고하세요.",
        included_stages=evidence.present_stage_names,
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


def _register(artifact_ids: list[str], result: ReportResult, runtime: AgentRuntime) -> ArtifactRef:
    anchor = runtime.adapter.get_artifact(artifact_ids[0])
    return runtime.adapter.register_artifact(
        anchor.run_id,
        ArtifactType.file,
        content_text=json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        filename="report.json",
        created_by_tool=_TOOL_NAME,
        parent_ids=artifact_ids,
        metadata={
            "kind": _REPORT_KIND,
            "source_artifact_ids": sorted(artifact_ids),
            "report_version": _REPORT_VERSION,
        },
        preview={
            "title": result.title,
            "executive_summary": result.executive_summary,
            "key_finding": result.key_finding,
            "included_stages": result.included_stages,
            "fallback_used": result.fallback_used,
        },
    )


def _build_prompt(evidence: PathEvidence, feedback: str) -> str:
    stage_blocks = []
    for stage in evidence.stages:
        facts_text = json.dumps(stage.facts, ensure_ascii=False, default=str)[:2500]
        stage_blocks.append(f"[{_STAGE_LABELS.get(stage.stage, stage.stage)} 근거]\n{facts_text}")
    evidence_text = "\n\n".join(stage_blocks)

    charts_text = json.dumps(
        [{"artifact_id": c.artifact_id, "caption": c.caption, "stage": c.stage} for c in evidence.charts],
        ensure_ascii=False,
    )
    question_line = f"[사용자 원 질문] {evidence.user_question}\n" if evidence.user_question else ""
    feedback_section = f"\n[이전 시도 피드백]\n{feedback}\n" if feedback else ""

    return f"""너는 데이터 분석 프로젝트 전체를 정리해 제출하는 전문 보고서 작성자다. 아래는 한 분석
여정의 전체 근거(SQL→EDA→분석→인사이트 순으로 실제 일어난 순서, 없는 단계는 생략됨)다.
근거만 보고, 실제 제출하는 수준으로 정제되고 전문적인 보고서를 써라. 새로운 계산·추측을
하지 말고, 근거에 없는 숫자나 없는 단계를 지어내지 마라(없는 단계는 그냥 언급하지 않으면 된다).

{question_line}[중요] 이 근거들을 "1단계는 이랬고, 2단계는 이랬고..." 식으로 순서대로
나열하지 마라. 그렇게 쓰면 보고서가 아니라 회의록이 된다. 대신 아래 인과선을 따라
하나의 이야기로 엮어라:
- SQL에서 이 지표/테이블을 왜 선택했는지 → (있다면) EDA가 그 데이터의 신뢰도를 어떻게
  검증했는지, 그 결과가 이후 분석 방향에 어떤 영향을 줬는지
- 분석에서 어떤 방법을 왜 선택했고(method_decision이 있다면 그 rationale) 무엇을 확인했는지
  (evidence의 각 항목: method/statistics/finding/caveats), 그 방법 선택이 앞선 근거(SQL
  설계나 EDA 발견) 때문이라면 그 인과를 밝혀라 — 분석 근거의 evidence/method_decision은
  analysis_kind와 무관하게 항상 있으니 절대 건너뛰지 말고 반드시 이 흐름 안에 녹여라.
  **분석 방법이 실제로 무엇이었는지(상관검정/회귀/세그멘테이션/코호트/이상탐지 등)에 맞는
  말로 서술하라 — 근거에 없는 개념(예: 가설검정이 아닌데 "귀무가설")을 갖다 붙이지 마라.**
  분석 근거에 hypotheses(귀무/대립가설)가 실제로 있을 때만 그 표현을 써라.
- (있다면) 최종 인사이트/결론이 앞의 어떤 구체적 발견에서 도출됐는지 반드시 되짚어 연결하라
  (예: "앞서 확인된 X 이상치를 제외하고 재검토한 결과 Y로 나타났다" 같은 식). 인사이트 근거에
  answer가 있다면 executive_summary는 그 answer와 결이 어긋나지 않게 써라

[정직성 — 반드시 지켜라] 분석 evidence의 finding이나 hypotheses의 decision이 확정적이지
않다면(예: "inconclusive"/판정불가, 또는 caveats에 불확실성이 명시됨) 확정된 경향이나
결론처럼 서술하지 마라. "감소했다/확인되었다"처럼 단정하지 말고 "경향은 보이나 통계적으로
판정되지 않았다" 식으로 그 불확실성을 그대로 남겨라. 확정된 것과 불확정인 것을 절대 같은
확신도로 쓰지 마라.

[금지] 각 문단을 "SQL 단계에서는~", "EDA 단계에서는~", "분석 단계에서는~" 같은 말로
시작하지 마라. 이런 문장이 나오면 나열식으로 간주해 실패로 처리한다.

{evidence_text}

[사용 가능한 차트] {charts_text}
{feedback_section}
[말투] "~를 진행하여 ~를 확인하였습니다. 결론적으로 ~합니다" 같은 정중하고 전문적인
보고서체를 써라. 캐주얼한 구어체는 금지.

[구조 — 반드시 이 필드로 JSON 출력]
- title: 이 분석 여정 전체를 나타내는 구체적인 제목
- executive_summary: 결론부터 먼저 요약(인사이트 근거의 answer가 있다면 그 답변과 일관되게).
  3~5문장.
- background_and_question: 사용자가 뭘 궁금해했는지, 왜 이 분석 여정이 필요했는지. 1~2문단.
- methodology_narrative: SQL/EDA/분석을 하나로 묶어 "이렇게 데이터를 준비하고, 검증하고,
  어떤 방법으로 무엇을 확인했는지"의 흐름으로 서술(있는 것만 자연스럽게). 분석 evidence가
  있으면 어떤 방법(method)을 왜 선택했고 무엇을 확인했는지(finding)를 반드시 포함하라 —
  hypotheses(귀무/대립가설)가 있으면 그 형태로, 없으면 evidence의 method/finding을 그
  분석 방법에 맞는 말로 서술하라(위 [정직성] 규칙을 지키면서). 최대 3문단.
- key_findings: 발견을 다루는 섹션 리스트(핵심 파트, 최소 1개 이상). 각 섹션:
  {{"heading":"소제목", "body":"본문(근거의 구체적 수치를 최대한 인용, 이전 발견과의
  연결이 자연스러우면 그 연결도 문장에 녹여라)",
    "source_label":"짧은 카테고리 태그(선택, 없으면 생략)",
    "chart_artifact_ids":["위 [사용 가능한 차트]에 있는 artifact_id만, 없으면 빈 리스트"]}}
  위 차트 목록의 캡션을 보고 관련 있는 섹션에 반드시 연결하라.
- limitations: 이 분석 여정의 한계(문자열 리스트)
- conclusion_and_recommendations: 최종 결론과 실행 제안(있다면 인사이트의 action_plan을
  자연스럽게 녹여서). 1~2문단.
- key_finding: 위 전체를 압축한 한 문장(리포트가 접혀있을 때 보일 미리보기용)

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
