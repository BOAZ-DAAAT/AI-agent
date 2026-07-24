from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from data_agent_backend.models.artifacts import ArtifactRef, ArtifactType

from DATA_Analyst_Assistant_Agent.agents.artifact_data import CsvArtifactData, load_analysis_inputs
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, get_llm, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.lib.derived_group import (
    build_derived_group_frame,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AgentEnvelope,
    LocalCheck,
    OrchestrationState,
    RetryHint,
    ValidationFinding,
    ValidationBlock,
)


def register_key_chart_artifacts(runtime, state, chart_paths, parent_ids, context, captions=None):
    """key 차트 PNG를 아티팩트로 등록한다 — **이상적 형태(content_bytes)** 로 호출.

    adapter가 아직 바이너리(content_bytes)를 지원하지 않으면 가드로 잡아 artifact_id=None 폴백한다
    (파이프라인 안 막음). 백엔드가 adapter에 content_bytes를 열면 코드 변경 없이 실제 등록이 작동한다.
    분석 에이전트는 경로 대신 artifact_id로 차트를 로드(멀티모달)한다.
    반환: (entries, refs) — entries=[{"filename","artifact_id","caption"}, ...](payload용),
    refs=등록 성공한 ArtifactRef 목록(AgentEnvelope.artifact_refs용 — 빠지면 state.artifact_ids에
    안 잡혀서 차트가 "만들어졌지만 추적 안 되는" 상태가 된다, #120).
    """
    captions = captions or {}
    entries: list[dict[str, Any]] = []
    refs: list[ArtifactRef] = []
    for path in chart_paths or []:
        filename = os.path.basename(path)
        caption = captions.get(filename, "")
        artifact_id = None
        try:
            with open(path, "rb") as fh:
                png_bytes = fh.read()
            ref = runtime.adapter.register_artifact(
                state.run_id,
                ArtifactType.chart,
                content_bytes=png_bytes,   # 이상형 — 백엔드가 content_bytes 열면 실제 저장
                filename=filename,
                created_by_tool="DATA_Analyst_Assistant_Agent.eda.lang_graph",
                context=context,
                parent_ids=parent_ids,
                metadata={"caption": caption},   # 선정 이유 — 분석 멀티모달 읽기의 설명서(#71 B)
            )
            artifact_id = ref.artifact_id
            refs.append(ref)
        except Exception:  # noqa: BLE001  # adapter 미지원/파일 없음 등 → 폴백
            artifact_id = None
        entries.append({"filename": filename, "artifact_id": artifact_id, "caption": caption})
    return entries, refs


def _relationship_shift_for_row_filter(
    original_df: pd.DataFrame, filtered_df: pd.DataFrame, primary_hypothesis: dict[str, Any],
) -> dict[str, Any] | None:
    """row_filter 분기 전후로 EDA 주가설(target/feature)의 상관계수가 어떻게 바뀌었는지 계산한다.

    Analysis는 필터링된 df만 받아서 원본과 비교할 방법이 없다(원래 있던 행이 이미 빠져있어
    되살릴 수 없음, 2026-07-24). EDA는 필터링하는 바로 이 시점에 원본(original_df)과
    필터본(filtered_df)을 둘 다 메모리에 갖고 있으니, 여기서 딱 한 번 계산해서 넘긴다 —
    Analysis가 필터링된 df를 "원본"이라 잘못 부르거나 스스로 또 필터링하지 않도록.
    """
    def _resolve_column(text: str) -> str | None:
        # primary_hypothesis의 target/feature가 항상 깔끔한 컬럼명은 아니다 — "A 또는 B"처럼
        # 서술형 텍스트로 나올 때가 있다(2026-07-24 실측). 정확히 일치가 아니라, 실제 df
        # 컬럼명이 그 텍스트 안에 부분 문자열로 등장하는지로 느슨하게 찾는다. 숫자 컬럼을
        # 우선한다(상관계수 계산 대상이라 문자열 컬럼은 못 씀).
        cf = text.casefold()
        numeric_cols = set(original_df.select_dtypes(include="number").columns)
        candidates = [c for c in original_df.columns if c.casefold() in cf]
        numeric_candidates = [c for c in candidates if c in numeric_cols]
        return (numeric_candidates or candidates or [None])[0]

    target = _resolve_column(str(primary_hypothesis.get("target") or ""))
    feature = _resolve_column(str(primary_hypothesis.get("feature") or ""))
    if not target or not feature or target == feature:
        return None

    def _corr_pair(frame: pd.DataFrame) -> dict[str, Any] | None:
        x = pd.to_numeric(frame[target], errors="coerce")
        y = pd.to_numeric(frame[feature], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 3:
            return None
        xs, ys = x[valid], y[valid]
        if xs.std() == 0 or ys.std() == 0:
            return None
        return {
            "n": int(valid.sum()),
            "pearson": round(float(xs.corr(ys, method="pearson")), 4),
            "spearman": round(float(xs.corr(ys, method="spearman")), 4),
        }

    before = _corr_pair(original_df)
    after = _corr_pair(filtered_df)
    if not before or not after:
        return None
    return {"target": target, "feature": feature, "before": before, "after": after}


def _extract_branch_instruction(user_query: str) -> str:
    """Return only the follow-up branch instruction, not the original query."""
    markers = ("[추가 지시사항]", "[異붽? 吏?쒖궗??]")
    for marker in markers:
        if marker in user_query:
            return user_query.split(marker, 1)[1].strip()
    return ""


class EDAAgent:
    name = "eda_agent"

    def run(self, state: OrchestrationState, runtime: AgentRuntime) -> AgentEnvelope:
        context = runtime.context(state, node_name=self.name, tool_name="eda_agent.lang_graph")
        # comprehensive(마트) 경로면 analytics 스키마에서 마트를 DB 로 직접 조회하고,
        # simple 경로/조회 실패 시 sql_result CSV 아티팩트로 폴백한다(반환형은 동일).
        csvs = load_analysis_inputs(state, runtime)
        source_ids = [csv.artifact_id for csv in csvs if csv.artifact_id]
        profile = profile_from_csv_artifacts(csvs)

        # 원본 LangGraph EDA 실행 (planner → 분석 노드 → insight/hypothesis → chart_selector)
        eda_result = self._run_eda_graph(csvs, state)
        # JSON payload에 안 들어가게 여기서 바로 꺼낸다(DataFrame은 json.dumps 대상이 아님).
        row_filter_frame = eda_result.pop("_row_filter_frame", None)

        # key 차트 PNG를 아티팩트로 등록(이상형+가드) → 경로 대신 {filename, artifact_id}로 전달
        key_chart_entries, key_chart_refs = register_key_chart_artifacts(
            runtime, state, eda_result.get("key_charts", []), source_ids, context,
            captions=eda_result.get("key_chart_captions", {}))

        # codegen 탈출구가 도메인 밖으로 판정하면 top-level 플래그로 정직하게 노출한다
        # (성공 결과는 statistical_metadata.adhoc_analysis에 편입됨). 분석 에이전트가 라우팅에 씀.
        derived_group_comparison = (eda_result.get("statistical_metadata") or {}).get("derived_group_comparison")
        if derived_group_comparison:
            statistical_metadata = dict(eda_result.get("statistical_metadata", {}) or {})
            statistical_metadata["derived_group_comparison"] = derived_group_comparison
            eda_result["statistical_metadata"] = statistical_metadata
            if "[추가 분기 집계]" not in str(eda_result.get("final_summary", "")):
                eda_result["final_summary"] = _append_derived_summary(
                    eda_result.get("final_summary", ""),
                    derived_group_comparison,
                )
        effective_profile = eda_result.get("profile_override") or profile
        payload = {
            "run_id": state.run_id,
            "source_artifacts": source_ids,
            "profile": effective_profile,
            "user_question": eda_result.get("user_question", state.user_query),
            "analysis_plan": eda_result.get("analysis_plan", {}),
            "insight_result": eda_result.get("insight_result", ""),
            "hypotheses": eda_result.get("hypotheses", ""),
            "primary_hypothesis": eda_result.get("primary_hypothesis", {}),
            "final_summary": eda_result.get("final_summary", ""),
            "analysis_target": eda_result.get("analysis_target", ""),
            "data_level": eda_result.get("data_level", {}),
            "cautions": eda_result.get("cautions", []),
            "analysis_constraints": eda_result.get("analysis_constraints", []),
            "analysis_data_contract": eda_result.get(
                "analysis_data_contract",
                (state.plan.analysis_data_contract if state.plan else {}),
            ),
            "statistical_metadata": eda_result.get("statistical_metadata", {}),
            "key_charts": key_chart_entries,
            "error_log": eda_result.get("error_log", []),
        }
        ref = runtime.adapter.register_artifact(
            state.run_id,
            ArtifactType.data_profile,
            content_text=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            filename="eda_summary.json",
            created_by_tool="DATA_Analyst_Assistant_Agent.eda.lang_graph",
            context=context,
            parent_ids=source_ids,
            metadata={"kind": "eda_summary", "source_artifact_count": len(source_ids)},
            preview={
                "row_count": effective_profile["row_count"],
                "columns": effective_profile["columns"],
                "final_summary": payload["final_summary"],
                "key_charts": payload["key_charts"],
                "quality_status": effective_profile["quality_status"],
            },
        )
        derived_frame_refs: list[ArtifactRef] = []
        if row_filter_frame is not None:
            # Analysis/Insight가 SQL/DB를 처음부터 다시 읽는 대신 이 필터링된 결과를 우선
            # 쓰도록 남긴다(artifact_data.load_eda_derived_frame이 kind로 찾아 읽는다).
            # 원본과 스키마·그레인이 동일해서 다운스트림 계약(analysis_data_contract)이
            # 그대로 유효하다 — 그래서 row_filter만 여기까지 온다(agent.py의 위 필터 참고).
            derived_ref = runtime.adapter.register_artifact(
                state.run_id,
                ArtifactType.sql_result,
                content_text=row_filter_frame.to_csv(index=False),
                filename=f"eda_derived_frame_{state.run_id}.csv",
                created_by_tool="DATA_Analyst_Assistant_Agent.eda.lang_graph",
                context=context,
                parent_ids=source_ids,
                lineage_edge_type="filtered_from",
                metadata={"kind": "eda_derived_frame", "source": "eda_agent.derived_group.row_filter"},
                preview={
                    "row_count": int(len(row_filter_frame)),
                    "columns": list(row_filter_frame.columns),
                },
            )
            derived_frame_refs.append(derived_ref)
        return AgentEnvelope(
            agent_name=self.name,
            summary=payload["final_summary"] or "EDA LangGraph analysis completed.",
            artifact_refs=[ref, *key_chart_refs, *derived_frame_refs],
            validation=ValidationBlock(
                local_checks=run_eda_self_check(source_ids, profile, payload["cautions"]),
                findings=run_eda_validation_findings(payload["cautions"]),
            ),
            retry_hint=_build_retry_hint(eda_result.get("validation_result", {})),
            # 서브에이전트는 핸드오프를 갖지 않는다 — 다음 단계 라우팅은 메인(supervisor)의 몫.
        )

    def _run_eda_graph(self, csvs: list[Any], state: OrchestrationState) -> dict[str, Any]:
        """CSV 아티팩트로 DataFrame을 만들고 원본 EDA LangGraph를 실행한다."""
        from DATA_Analyst_Assistant_Agent.agents.eda.graph import build_app

        frames = [csv.dataframe for csv in csvs if csv.error is None and not csv.dataframe.empty]
        if not frames:
            raise RuntimeError("No non-empty SQL CSV artifact was available for EDA.")
        # mart 경로에선 '마트 생성 완료' 상태 메시지 CSV(1행)가 섞여 온다 — concat 하면
        # col_1 쓰레기차트·data_level 오판·key_col 오염으로 comparison 이 전멸한다(E2E 실측).
        # 스키마가 같은 프레임만 합치고, 아니면 실질 데이터(최대 행) 프레임을 쓴다.
        main = max(frames, key=len)
        same_schema = [f for f in frames if list(f.columns) == list(main.columns)]
        df = pd.concat(same_schema, ignore_index=True) if len(same_schema) > 1 else main
        branch_instruction = _extract_branch_instruction(state.user_query or "")
        derived_question = f"[추가 지시사항] {branch_instruction}" if branch_instruction else ""
        # branch_instruction 이 없으면(일반 실행) build_derived_group_frame 이 곧바로 None을
        # 리턴하므로, 그 흔한 경로에서까지 LLM 클라이언트를 미리 만들지 않는다(테스트/키 없는
        # 환경에서 불필요하게 실패하지 않도록 — get_llm() 은 실제 필요할 때만 평가).
        derived_input = (
            build_derived_group_frame(df, derived_question, llm=get_llm())
            if branch_instruction
            else None
        )
        graph_df = derived_input["dataframe"] if derived_input else df
        graph_mart_design = derived_input.get("mart_design", {}) if derived_input else {}
        graph_question = state.user_query or ""
        if derived_input:
            meta = derived_input["metadata"]
            filter_desc = f'"{meta["filter_expression"]}" 조건' if meta.get("filter_expression") else "필터 없이 전체"
            if meta.get("kind") == "row_filter":
                graph_question = (
                    f"{graph_question}\n\n"
                    "[분기 EDA 입력]\n"
                    f"개체 집계 없이 원본 행 그레인 그대로 {filter_desc}을 적용했습니다. "
                    f"현재 EDA 그래프 입력은 전체 {meta['total_rows']}행 중 "
                    f"{meta['eligible_rows']}행입니다."
                )
            else:
                graph_question = (
                    f"{graph_question}\n\n"
                    "[분기 EDA 입력]\n"
                    f"{meta['entity_col']} 기준으로 임시 집계표를 생성한 뒤 {filter_desc}을 적용했습니다. "
                    f"현재 EDA 그래프 입력은 원본 주문 행이 아니라 "
                    f"{meta['eligible_entities']}개 {meta['entity_col']} 집계 행입니다."
                )

        # 원본 모듈 전역(_df 등)을 대체하는 실행 컨텍스트. df 만 채우고
        # key/measure/time 컬럼은 load_mart 노드가 확정한다.
        set_context(EdaContext(
            df=graph_df,
            key_col=derived_input.get("key_col") if derived_input else None,
            measure_cols=derived_input.get("measure_cols") if derived_input else None,
            target_col=derived_input.get("target_col") if derived_input else None,
            question_type=state.route_kind or "",
            user_question=graph_question,
        ))
        # 앞단이 넘긴 의미 힌트를 EDA로 전달(있으면 줍고 없으면 폴백). 앞단(오케스트레이터)은
        # "매출"/"월"/None 수준이라 대개 비어 옴 → 가설 노드가 priority_metrics로 폴백한다.
        plan = state.plan
        plan_metric = (plan.metric if plan and plan.metric else "") or ""
        plan_dimension = (plan.dimension if plan and plan.dimension else "") or ""
        # GE 정합성 스코핑용 원천 테이블 + grain 교차검증용 선언 grain(있으면 줍고 없으면 폴백).
        plan_source_tables = list(plan.source_tables) if plan and plan.source_tables else []
        plan_business_grain = (plan.business_grain if plan and plan.business_grain else "") or ""
        analysis_data_contract = dict(plan.analysis_data_contract) if plan and plan.analysis_data_contract else {}
        # 수퍼바이저가 직전 시도를 부실 판정했으면(semantic/hard 실패), EDA 자체 재시도 루프
        # (validator.py → validation_feedback)가 읽는 자리에 초기값으로 심어 재사용한다.
        retry_context = plan.retry_context if plan and plan.retry_context else {}
        eda_feedback = (retry_context.get("agent_feedback") or {}).get("eda_agent")
        initial_validation_feedback = ""
        if isinstance(eda_feedback, dict):
            reason = eda_feedback.get("reason") or ""
            missing = eda_feedback.get("missing_evidence") or []
            missing_text = f" 누락된 근거: {', '.join(str(item) for item in missing)}." if missing else ""
            initial_validation_feedback = f"{reason}{missing_text}".strip()
        try:
            from DATA_Analyst_Assistant_Agent.agents.eda.lib.visualize import clear_output_dirs
            clear_output_dirs()  # 이전 런 누적 PNG 정리(run_eda_only도 이 경로를 타므로 규칙 동일)
            app = build_app()
            result = app.invoke(
                {
                    "user_question": graph_question,
                    "target_table": "",
                    "mart_design": graph_mart_design,
                    "question_type": state.route_kind or "",
                    "plan_metric": plan_metric,
                    "plan_dimension": plan_dimension,
                    "plan_source_tables": plan_source_tables,
                    "plan_business_grain": plan_business_grain,
                    "analysis_data_contract": analysis_data_contract,
                    "validation_feedback": initial_validation_feedback,
                    "error_log": [],
                }
            )
        finally:
            reset_context()
        if derived_input:
            # row_filter만 다운스트림(Analysis/Insight)에 CSV로 넘긴다 — 원본과 스키마·그레인이
            # 동일해서 안전하다. entity_comparison(판매자 집계 등)은 그레인이 바뀌어
            # analysis_data_contract와 어긋나므로 EDA 내부용으로만 남겨둔다(2026-07-24).
            if derived_input["metadata"].get("kind") == "row_filter":
                shift = _relationship_shift_for_row_filter(
                    df, graph_df, result.get("primary_hypothesis") or {})
                if shift:
                    derived_input["metadata"]["relationship_shift"] = shift
                result["_row_filter_frame"] = graph_df
            statistical_metadata = dict(result.get("statistical_metadata", {}) or {})
            statistical_metadata["derived_group_comparison"] = derived_input["metadata"]
            result["statistical_metadata"] = statistical_metadata
            result["profile_override"] = profile_from_csv_artifacts([
                CsvArtifactData(artifact_id="derived_group", text="", dataframe=graph_df)
            ])
        return result

# ─────────────────────────────
# 백엔드 핸드오프 헬퍼 (구 profiler.py / self_check.py 통합)
# ─────────────────────────────
def profile_from_csv_artifacts(csvs: list[CsvArtifactData]) -> dict[str, Any]:
    key_issues: list[str] = []
    source_artifacts = [csv.artifact_id for csv in csvs]
    read_errors = {csv.artifact_id: csv.error for csv in csvs if csv.error}

    if not csvs:
        key_issues.append("No SQL result artifact was available for EDA.")
        df = pd.DataFrame()
    else:
        frames = [csv.dataframe for csv in csvs if csv.error is None]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if read_errors:
        key_issues.extend(f"{artifact_id}: {error}" for artifact_id, error in read_errors.items())
    if csvs and len(df.columns) == 0:
        key_issues.append("SQL result CSV did not include columns.")
    if csvs and len(df) == 0:
        key_issues.append("SQL result CSV contains zero data rows.")

    null_counts = {column: int(count) for column, count in df.isna().sum().items()} if len(df.columns) else {}
    unique_counts = {column: int(df[column].nunique(dropna=True)) for column in df.columns}
    numeric_summary = _numeric_summary(df)
    categorical_top_values = _categorical_top_values(df)

    if not csvs:
        quality_status = "unavailable"
        recommended_next_steps = ["run_sql_preview", "clarify_data_source"]
    elif key_issues:
        quality_status = "needs_review"
        recommended_next_steps = ["review_csv_artifact", "rerun_or_refine_sql"]
    else:
        quality_status = "usable"
        recommended_next_steps = ["continue_to_analysis", "document_limitations"]

    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": null_counts,
        "unique_counts": unique_counts,
        "numeric_summary": numeric_summary,
        "categorical_top_values": categorical_top_values,
        "sample_available": len(df) > 0,
        "quality_status": quality_status,
        "key_issues": key_issues,
        "recommended_next_steps": recommended_next_steps,
        "source_artifacts": source_artifacts,
    }


def _numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    summary = numeric_df.describe().to_dict()
    return {
        column: {
            stat: (None if pd.isna(value) else float(value))
            for stat, value in stats.items()
            if stat in {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
        }
        for column, stats in summary.items()
    }


def _categorical_top_values(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    categorical: dict[str, dict[str, int]] = {}
    for column in df.select_dtypes(exclude="number").columns:
        counts = df[column].fillna("<NA>").astype(str).value_counts(dropna=False).head(5)
        categorical[column] = {str(value): int(count) for value, count in counts.items()}
    return categorical


def _append_derived_summary(summary: str, derived: dict[str, Any]) -> str:
    findings = [str(item) for item in derived.get("findings", []) if str(item).strip()]
    if not findings:
        return summary
    addition = " ".join(findings[:3])
    if not summary:
        return addition
    return f"{summary}\n\n[추가 분기 집계] {addition}"


def run_eda_self_check(
    source_artifact_ids: list[str], profile: dict, cautions: list[dict] | None = None
) -> list[LocalCheck]:
    cautions = cautions or []
    validator_caution = next(
        (
            c
            for c in cautions
            if isinstance(c, dict) and c.get("code") == "EDA_SELF_VALIDATION_FAILED"
        ),
        None,
    )
    validator_failure = validator_caution.get("message_ko", "") if validator_caution else None
    validator_retryable = bool((validator_caution or {}).get("details", {}).get("retryable"))
    return [
        LocalCheck(
            name="source_artifact_present",
            passed=bool(source_artifact_ids),
            severity="error" if not source_artifact_ids else "info",
            detail="EDA requires at least one upstream SQL artifact.",
        ),
        LocalCheck(
            name="profile_generated",
            passed=bool(profile),
            severity="error" if not profile else "info",
            detail="EDA profile payload was generated.",
        ),
        LocalCheck(
            name="columns_present",
            passed=bool(profile.get("columns")),
            severity="warning" if not profile.get("columns") else "info",
            detail="Column metadata should be available for downstream analysis.",
        ),
        LocalCheck(
            name="eda_self_validation",
            passed=validator_failure is None,
            severity="error" if validator_retryable else "warning" if validator_failure is not None else "info",
            detail=validator_failure or "EDA internal validator passed.",
        ),
    ]


def run_eda_validation_findings(cautions: list[dict] | None = None) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for caution in cautions or []:
        if not isinstance(caution, dict):
            continue
        if caution.get("code") != "EDA_SELF_VALIDATION_FAILED":
            continue
        details = caution.get("details") if isinstance(caution.get("details"), dict) else {}
        retryable = bool(details.get("retryable"))
        findings.append(
            ValidationFinding(
                code=str(details.get("failure_code") or "eda_self_validation_failed"),
                source="eda_validator",
                severity="error" if retryable else "warning",
                disposition="error" if retryable else "limitation",
                message=str(caution.get("message_ko") or "EDA self validation requires review."),
                retryable=retryable,
                suggested_action="rerun_eda_agent" if retryable else "review_eda_before_use",
                details=details,
            )
        )
    return findings


def _build_retry_hint(validation_result: dict | None) -> RetryHint:
    """validator_node가 남긴 failure_code/retryable을 supervisor의 retry_hint로 변환한다.

    retryable=True인 경우에만 supervisor가 eda_agent를 통째로 1회 더 호출한다
    (supervisor/validation.py의 기존 재시도 엔진 재사용, 그 외 필드는 기본값 유지)."""
    validation_result = validation_result or {}
    if not validation_result.get("retryable"):
        return RetryHint()
    return RetryHint(
        retryable=True,
        reason_code="eda_self_validation_failed_retryable",
        suggested_action="rerun_eda_agent",
        details={
            "failure_code": validation_result.get("failure_code", ""),
            "failure_reason": validation_result.get("reason", ""),
        },
    )
