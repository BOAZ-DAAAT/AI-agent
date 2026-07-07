"""InsightAgent — evidence-to-answer 최종 해석자 (구 report agent 대체 예정).

상류 아티팩트(읽기 전용) → 증거팩 → bounded ReAct 루프 → 직답 문구 + 차트를
final_report.md / insight_payload.json 아티팩트로 등록한다.

배선 주의:
- 기본 backend factory 에 workspace_storage 가 없어 save_workspace_file 은 쓰지 않는다
  (register_artifact 직접 사용 — ReportAgent 의 알려진 함정 회피).
- supervisor 스왑(tools.py 1줄) 전까지 기존 ReportAgent 와 병존한다.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.insight.evidence import EvidencePack, build_evidence_pack
from DATA_Analyst_Assistant_Agent.agents.insight.loop import run_insight_loop
from DATA_Analyst_Assistant_Agent.agents.insight.schemas import ChartEntry, InsightResult
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, LocalCheck, OrchestrationState, ValidationBlock

_TOOL_NAME = "insight_agent.react_loop"


class InsightAgent:
    name = "insight_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name=_TOOL_NAME)
        pack = build_evidence_pack(state, runtime)
        out_dir = os.getenv("INSIGHT_CHART_DIR") or os.path.join(
            tempfile.gettempdir(), f"insight_charts_{state.run_id}")
        result = run_insight_loop(pack, out_dir=out_dir)

        chart_refs = _register_chart_artifacts(runtime, state, result.charts,
                                               pack.source_artifact_ids, context)
        payload = _build_payload(state, pack, result)
        markdown = _build_markdown(payload)

        report_ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.report,
            content_text=markdown,
            filename="final_report.md",
            created_by_tool=_TOOL_NAME,
            context=context,
            parent_ids=pack.source_artifact_ids,
            metadata={"kind": "final_report", "source_artifact_count": len(pack.source_artifact_ids)},
            preview={"answer": payload["answer"][:300]},
        )
        payload_ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.file,
            content_text=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            filename="insight_payload.json",
            created_by_tool=_TOOL_NAME,
            context=context,
            parent_ids=pack.source_artifact_ids,
            metadata={"kind": "insight_payload"},
            preview={"answer": payload["answer"][:300], "chart_count": len(result.charts),
                     "fallback_used": result.fallback_used},
        )
        return AgentEnvelope(
            agent_name=self.name,
            summary=payload["answer"][:400] or "인사이트 생성 완료",
            artifact_refs=[report_ref, payload_ref, *chart_refs],
            validation=ValidationBlock(local_checks=_self_check(pack, result)),
            fallback_used=result.fallback_used,
        )


def _register_chart_artifacts(runtime, state, charts: list[ChartEntry],
                              parent_ids: list[str], context) -> list[ArtifactRef]:
    """차트 PNG 등록 — EDA 와 같은 이상형(content_bytes)+가드 패턴.

    adapter 가 아직 바이너리를 지원하지 않으면 artifact_id=None 폴백(파이프라인 안 막음).
    local_path 는 payload 에 남아 데모/로컬 소비가 가능하다.
    """
    refs: list[ArtifactRef] = []
    for chart in charts:
        try:
            with open(chart.local_path, "rb") as fh:
                png_bytes = fh.read()
            ref = runtime.adapter.register_artifact(
                state.run_id,
                ArtifactType.chart,
                content_bytes=png_bytes,               # 이상형 — 백엔드가 열면 실제 저장
                filename=chart.filename,
                created_by_tool=_TOOL_NAME,
                context=context,
                parent_ids=parent_ids,
                metadata={"kind": "insight_chart", "title": chart.title, "supports": chart.supports},
            )
            chart.artifact_id = ref.artifact_id
            refs.append(ref)
        except Exception:  # noqa: BLE001  # adapter 미지원/파일 없음 → 폴백
            chart.artifact_id = None
    return refs


def _build_payload(state: OrchestrationState, pack: EvidencePack, result: InsightResult) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "user_question": pack.user_question,
        "route_kind": pack.route_kind,
        "answer": result.answer,
        "key_insights": result.key_insights,
        "action_plan": result.action_plan,
        "limitations": result.limitations,
        "charts": [c.model_dump() for c in result.charts],
        "evidence_sources": pack.source_artifact_ids,
        "evidence_labels": pack.source_labels,          # id → "SQL 결과 테이블" 등 (사람이 읽는 출처)
        "steps": result.steps,                         # 행동 트레이스(provenance)
        "fallback_used": result.fallback_used,
        "rounds": result.rounds,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    """대답이 최상단 — '문서'가 아니라 '답'이 주인공인 리포트."""
    lines = [
        "# Insight Report", "",
        "## Answer",
        payload["answer"] or "(답변 생성 실패)", "",
    ]
    if payload["key_insights"]:
        lines += ["## Key Insights", *[f"- {s}" for s in payload["key_insights"]], ""]
    if payload["charts"]:
        lines += ["## Charts"]
        for c in payload["charts"]:
            # 이미지 임베드 — md 를 차트 폴더 옆에 두면 바로 렌더된다 (파일명 나열만 하던 것 개선)
            rel_dir = os.path.basename(os.path.dirname(c.get("local_path") or "")) or "charts"
            lines += [f"![{c['title'] or c['filename']}]({rel_dir}/{c['filename']})",
                      f"- {c['title'] or c['filename']} (`{c['filename']}`, supports={c['supports']})"]
        lines += [""]
    if payload["action_plan"]:
        lines += ["## Action Plan", *[f"- {s}" for s in payload["action_plan"]], ""]
    if payload["limitations"]:
        lines += ["## Limitations", *[f"- {s}" for s in payload["limitations"]], ""]
    labels = payload.get("evidence_labels") or {}
    lines += ["## User Question", payload["user_question"], "",
              "## Evidence Sources",
              *([f"- {labels.get(a, '상류 아티팩트')} (`{a}`)" for a in payload["evidence_sources"]]
                or ["- (없음)"]), ""]
    if payload["fallback_used"]:
        lines += ["> ⚠️ 답변 검증 실패로 보수적 요약이 제공되었습니다.", ""]
    return "\n".join(lines)


def _self_check(pack: EvidencePack, result: InsightResult) -> list[LocalCheck]:
    has_evidence = bool(pack.df is not None or pack.eda or pack.analysis)
    return [
        LocalCheck(name="evidence_present", passed=has_evidence,
                   severity="error" if not has_evidence else "info",
                   detail="상류 아티팩트(SQL/EDA/분석) 중 하나 이상이 필요하다."),
        LocalCheck(name="answer_generated", passed=bool(result.answer),
                   severity="error" if not result.answer else "info",
                   detail="사용자 질문 직답 문구가 생성되어야 한다."),
        LocalCheck(name="claims_verified", passed=not result.fallback_used,
                   severity="warning" if result.fallback_used else "info",
                   detail="숫자 검증 게이트 통과 여부(폴백이면 보수 답변)."),
        LocalCheck(name="chart_attached", passed=bool(result.charts), severity="info",
                   detail="답을 뒷받침하는 차트 존재 여부(권장)."),
    ]
