"""분기(재분석) 실행 — 그래프 라우팅을 타지 않고, 지정된 단계부터 끝까지 직접 순서대로 실행한다.

그래프의 LLM 라우팅(decide_next_action + validate_candidate의 semantic 추천)이 "일부만
완료된" 비정상 상태에서 예측 불가능하게 다음 단계를 건너뛰는 문제가 실측 확인됐다(SQL만
완료 표시하고 물었더니 EDA를 건너뛰고 분석으로 직행함). 그래서 분기는 의도적으로 그래프를
우회하고, SQL→EDA→분석→인사이트 고정 순서를 코드가 직접 강제한다.

각 단계 실행 후 이벤트(agent.started/completed/failed)와 서머리(generate_node_summary)를
직접 발행해서, 그래프가 만드는 것과 동일한 모양의 이벤트가 프론트로 흘러가게 한다(같은
emit_node_lifecycle_event/ActiveNodeExecution/CompletedNodeExecution을 재사용).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from DATA_Analyst_Assistant_Agent.agents.analysis import AnalysisAgent
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.eda.agent import EDAAgent
from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    AgentStatus,
    AnalysisPlan,
    OrchestrationState,
)
from DATA_Analyst_Assistant_Agent.supervisor.insight.agent import InsightGenerator
from DATA_Analyst_Assistant_Agent.supervisor.lifecycle import emit_node_lifecycle_event
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    ActiveNodeExecution,
    AgentName,
    CompletedNodeExecution,
    FailedNodeExecution,
    StepSummary,
)
from DATA_Analyst_Assistant_Agent.supervisor.summary.generator import generate_node_summary

BranchStage = Literal["sql", "eda", "analysis", "insight"]

_STAGE_ORDER: list[AgentName] = ["sql_agent", "eda_agent", "analysis_agent", "insight"]
_STAGE_ALIASES: dict[BranchStage, AgentName] = {
    "sql": "sql_agent",
    "eda": "eda_agent",
    "analysis": "analysis_agent",
    "insight": "insight",
}


@dataclass
class BranchResult:
    """이번 분기 실행에서 새로 생긴 것만 담는다(재사용한 상류 artifact_ids는 안 섞음)."""

    artifact_ids: dict[str, list[str]] = field(default_factory=dict)
    last_node_id: str = ""
    failed_agent: str | None = None
    failure_reason: str = ""

    @property
    def all_artifact_ids(self) -> list[str]:
        """리포트 생성 등에 바로 넘길 수 있게 전 단계 artifact_id를 평평하게 모은다."""
        ids: list[str] = []
        for stage_ids in self.artifact_ids.values():
            ids.extend(stage_ids)
        return ids


@dataclass(frozen=True)
class BranchPlanContext:
    generated_sql: str = ""
    target_table: str | None = None
    source_tables: list[str] = field(default_factory=list)
    business_grain: str | None = None
    mart_design: dict[str, Any] = field(default_factory=dict)
    analysis_data_contract: dict[str, Any] = field(default_factory=dict)


def branch_from(
    start_stage: BranchStage,
    new_instruction: str,
    *,
    upstream_artifact_ids: dict[str, list[str]],
    original_question: str,
    run_id: str,
    thread_id: str,
    runtime: AgentRuntime,
    backend_adapter: Any,
    parent_node_id: str | None = None,
    target_table: str | None = None,
    route_kind: str = "comprehensive",
) -> BranchResult:
    """start_stage부터 끝(insight)까지 직접 순서대로 실행한다.

    upstream_artifact_ids: start_stage 이전 단계들의 재사용할 artifact_ids
      (예: start_stage="eda"면 {"sql_agent": [...]}만 채워서 넘긴다).
    실패하면 그 지점까지 만든 결과와 실패 사유를 담아 즉시 반환한다(뒤 단계는 안 돎).
    """
    start_agent = _STAGE_ALIASES[start_stage]
    start_index = _STAGE_ORDER.index(start_agent)
    stages_to_run = _STAGE_ORDER[start_index:]

    goal = f"{original_question}\n\n[추가 지시사항] {new_instruction}".strip()
    artifact_ids = dict(upstream_artifact_ids)
    plan_context = _branch_plan_context_from_sql_artifacts(
        artifact_ids.get("sql_agent", []),
        runtime,
        fallback_target_table=target_table,
    )
    result = BranchResult()
    node_sequence = 0
    current_parent = parent_node_id
    fake_state = {"current_run_id": run_id}

    for agent_name in stages_to_run:
        node_sequence += 1
        node_id = f"{run_id}:branch:{node_sequence}"

        state = OrchestrationState(
            run_id=run_id,
            thread_id=thread_id,
            user_query=original_question,
            goal=goal,
            route_kind=route_kind,
            generated_sql=plan_context.generated_sql,
            plan=AnalysisPlan(
                goal=goal,
                route_kind=route_kind,
                generated_sql=plan_context.generated_sql,
                source_sql=plan_context.generated_sql,
                target_table=plan_context.target_table or target_table,
                source_tables=list(plan_context.source_tables),
                business_grain=plan_context.business_grain,
                mart_design=dict(plan_context.mart_design),
                analysis_data_contract=dict(plan_context.analysis_data_contract),
            ),
        )
        state.artifact_ids = dict(artifact_ids)

        active_node = ActiveNodeExecution(
            node_id=node_id,
            agent_name=agent_name,
            parent_node_id=current_parent,
            node_sequence=node_sequence,
            attempt=1,
            status="running",
        )
        emit_node_lifecycle_event(
            backend_adapter,
            fake_state,
            "agent.started",
            active_node,
            f"{agent_name} 작업을 시작했습니다(분기).",
        )

        try:
            envelope = _run_agent(agent_name, state, runtime)
        except Exception as exc:  # noqa: BLE001 - 분기 실행 실패는 그 지점에서 멈추고 사유를 알려야 함
            failed_node = FailedNodeExecution(
                node_id=node_id,
                agent_name=agent_name,
                parent_node_id=current_parent,
                node_sequence=node_sequence,
                attempt=1,
                reason=str(exc),
                reason_code="branch_execution_error",
            )
            emit_node_lifecycle_event(
                backend_adapter,
                fake_state,
                "agent.failed",
                failed_node,
                f"{agent_name} 실행 중 오류가 발생했습니다: {exc}",
            )
            result.failed_agent = agent_name
            result.failure_reason = str(exc)
            result.last_node_id = current_parent or ""
            return result

        stage_artifact_ids = [ref.artifact_id for ref in envelope.artifact_refs]
        artifact_ids[agent_name] = stage_artifact_ids
        result.artifact_ids[agent_name] = stage_artifact_ids

        if envelope.status == AgentStatus.failed:
            failure_reason = envelope.error or envelope.summary or f"{agent_name} returned failed status"
            failed_node = FailedNodeExecution(
                node_id=node_id,
                agent_name=agent_name,
                parent_node_id=current_parent,
                node_sequence=node_sequence,
                attempt=1,
                reason=failure_reason,
                reason_code=envelope.retry_hint.reason_code or "agent_failed",
            )
            emit_node_lifecycle_event(
                backend_adapter,
                fake_state,
                "agent.failed",
                failed_node,
                f"{agent_name} 작업이 실패했습니다(분기): {failure_reason}",
            )
            result.failed_agent = agent_name
            result.failure_reason = failure_reason
            result.last_node_id = current_parent or ""
            return result

        summary_text, summary_artifact_id = _safe_summarize(
            agent_name,
            stage_artifact_ids,
            runtime,
            new_instruction,
            envelope,
        )

        step_summary = StepSummary(
            step="execute_subagent",
            agent=agent_name,
            action=f"call_{agent_name}",
            summary=summary_text,
            artifact_ids=stage_artifact_ids,
            summary_artifact_id=summary_artifact_id,
            next_action="",
        )
        completed_node = CompletedNodeExecution(
            node_id=node_id,
            agent_name=agent_name,
            parent_node_id=current_parent,
            node_sequence=node_sequence,
            attempt=1,
            summary=step_summary,
        )
        emit_node_lifecycle_event(
            backend_adapter,
            fake_state,
            "agent.completed",
            completed_node,
            f"{agent_name} 작업을 완료했습니다(분기).",
        )
        current_parent = node_id

    result.last_node_id = current_parent or ""
    return result


def _branch_plan_context_from_sql_artifacts(
    artifact_ids: list[str],
    runtime: AgentRuntime,
    *,
    fallback_target_table: str | None = None,
) -> BranchPlanContext:
    context = BranchPlanContext(target_table=fallback_target_table)
    for artifact_id in artifact_ids:
        text = _safe_read_artifact_text(runtime, artifact_id)
        if not text:
            continue
        stripped = text.lstrip()
        if stripped[:1] in {"{", "["}:
            context = _merge_sql_json_context(context, _safe_load_json(stripped))
        elif not context.generated_sql and _looks_like_sql(stripped):
            context = _replace_context(context, generated_sql=stripped)
    if not context.analysis_data_contract:
        context = _replace_context(
            context,
            analysis_data_contract=_analysis_contract_from_plan_context(context),
        )
    return context


def _safe_read_artifact_text(runtime: AgentRuntime, artifact_id: str) -> str:
    try:
        return runtime.adapter.read_artifact_text(artifact_id)
    except Exception:  # noqa: BLE001 - branch hydration is best-effort.
        return ""


def _safe_load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 - non-JSON artifacts are ignored.
        return None


def _merge_sql_json_context(context: BranchPlanContext, payload: Any) -> BranchPlanContext:
    if not isinstance(payload, dict):
        return context

    sql_draft = payload.get("sql_draft")
    sql_draft = sql_draft if isinstance(sql_draft, dict) else {}
    mart_design = payload.get("mart_design")
    mart_design = mart_design if isinstance(mart_design, dict) else {}
    plan = payload.get("plan")
    plan = plan if isinstance(plan, dict) else {}
    analysis_contract = payload.get("analysis_data_contract")
    analysis_contract = analysis_contract if isinstance(analysis_contract, dict) else {}

    generated_sql = (
        context.generated_sql
        or str(sql_draft.get("sql") or payload.get("generated_sql") or payload.get("sql") or "")
    )
    target_table = (
        context.target_table
        or _non_empty_string(sql_draft.get("target_table"))
        or _target_table_from_mart_design(mart_design)
        or _non_empty_string(plan.get("target_table"))
    )
    source_tables = (
        context.source_tables
        or _string_list(sql_draft.get("source_tables"))
        or _string_list(mart_design.get("source_tables"))
        or _string_list(plan.get("selected_join_tables"))
        or _string_list(plan.get("candidate_tables"))
    )
    business_grain = (
        context.business_grain
        or _non_empty_string(sql_draft.get("business_grain"))
        or _non_empty_string(mart_design.get("grain"))
        or _non_empty_string(plan.get("business_grain"))
    )
    next_context = _replace_context(
        context,
        generated_sql=generated_sql,
        target_table=target_table,
        source_tables=source_tables,
        business_grain=business_grain,
        mart_design=context.mart_design or mart_design,
        analysis_data_contract=context.analysis_data_contract or analysis_contract,
    )
    if not next_context.analysis_data_contract:
        return _replace_context(
            next_context,
            analysis_data_contract=_analysis_contract_from_plan_context(next_context),
        )
    return next_context


def _replace_context(context: BranchPlanContext, **updates: Any) -> BranchPlanContext:
    values = {
        "generated_sql": context.generated_sql,
        "target_table": context.target_table,
        "source_tables": list(context.source_tables),
        "business_grain": context.business_grain,
        "mart_design": dict(context.mart_design),
        "analysis_data_contract": dict(context.analysis_data_contract),
    }
    values.update(updates)
    return BranchPlanContext(**values)


def _analysis_contract_from_plan_context(context: BranchPlanContext) -> dict[str, Any]:
    mart_design = context.mart_design
    if not mart_design and not (context.generated_sql or context.target_table):
        return {}
    column_plan = [item for item in mart_design.get("column_plan") or [] if isinstance(item, dict)]
    grain_columns = _string_list(mart_design.get("grain_columns"))
    source_tables = context.source_tables or _string_list(mart_design.get("source_tables"))
    return {
        "target_table": context.target_table or _target_table_from_mart_design(mart_design),
        "row_grain": context.business_grain or _non_empty_string(mart_design.get("grain")) or "",
        "grain_columns": grain_columns,
        "metric_support": [
            dict(item) for item in (mart_design.get("metric_support") or [])
            if isinstance(item, dict)
        ],
        "entity_keys": [column for column in grain_columns if not _looks_temporal_column(column)],
        "time_basis": [
            item for item in column_plan
            if _looks_temporal_column(str(item.get("output_column") or ""))
            or any(_looks_temporal_column(str(source)) for source in item.get("source_columns") or [])
        ],
        "derived_columns": column_plan,
        "aggregation_rules": [
            item for item in column_plan
            if str(item.get("aggregation_method") or "none").lower() != "none"
        ],
        "deduplication_keys": _string_list(mart_design.get("deduplication_keys")) or grain_columns,
        "source_tables": source_tables,
        "source_grains": dict(mart_design.get("source_grains") or {}),
        "safe_interpretations": [
            "선언된 row_grain 기준에서만 데이터마트를 해석합니다.",
            "집계 수준의 관계를 개별 주문 또는 판매자 인과로 단정하지 않습니다.",
        ],
        "generated_sql": context.generated_sql,
    }


def _target_table_from_mart_design(mart_design: dict[str, Any]) -> str | None:
    mart_name = _non_empty_string(mart_design.get("mart_name"))
    if not mart_name:
        return None
    schema = _non_empty_string(mart_design.get("target_schema"))
    return f"{schema}.{mart_name}" if schema else mart_name


def _non_empty_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _looks_like_sql(text: str) -> bool:
    return text.upper().startswith(("CREATE ", "SELECT ", "WITH ", "INSERT "))


def _looks_temporal_column(column: str) -> bool:
    normalized = column.casefold()
    return any(token in normalized for token in ("date", "time", "month", "year"))


def _run_agent(agent_name: AgentName, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
    if agent_name == "sql_agent":
        return SQLAgent().run(state, runtime)
    if agent_name == "eda_agent":
        return EDAAgent().run(state, runtime)
    if agent_name == "analysis_agent":
        return AnalysisAgent().run(state, runtime, chart_artifact_loader=runtime.adapter.read_artifact_bytes)
    if agent_name == "insight":
        return InsightGenerator().run(state, runtime)
    raise ValueError(f"알 수 없는 agent_name입니다: {agent_name}")


def _safe_summarize(
    agent_name: str,
    artifact_ids: list[str],
    runtime: AgentRuntime,
    branch_instruction: str,
    envelope: AgentEnvelope,
) -> tuple[str, str | None]:
    """generate_node_summary로 key_finding을 뽑는다. 실패하면 에이전트 자체 요약으로 되돌아간다."""
    try:
        ref = generate_node_summary(artifact_ids, runtime, branch_instruction=branch_instruction)
        payload = json.loads(runtime.adapter.read_artifact_text(ref.artifact_id))
        key_finding = str(payload.get("key_finding") or "").strip()
        return key_finding or envelope.summary, ref.artifact_id
    except Exception:  # noqa: BLE001 - 서머리는 부가 정보, 실패해도 완료 흐름은 계속
        return envelope.summary, None
