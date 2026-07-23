from __future__ import annotations

import logging
import mimetypes
import threading
from collections.abc import Callable
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import Response, StreamingResponse

from backend.auth.deps import get_current_user
from backend.session.service import get_owned_session
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.runs import RunRecord, RunStatus

from .event_stream import EVENT_STREAM_POLL_INTERVAL_SECONDS, stream_run_events
from .schemas import (
    AgentNodeReportResponse,
    AgentNodeSummaryResponse,
    AgentReportListItem,
    AgentReportListResponse,
    AgentRunCancelResponse,
    AgentRunBranchRequest,
    AgentRunBranchResponse,
    AgentRunCreateRequest,
    AgentRunDeleteResponse,
    AgentRunResponse,
    AgentRunResumeRequest,
    AgentRunResumeResponse,
)
from .service import (
    BranchPlanError,
    NodeReportGenerationError,
    NodeSummaryNotFoundError,
    RunCancellationConflictError,
    RunDeletionConflictError,
    cancel_agent_run,
    delete_terminal_run_data,
    get_node_summary,
    generate_node_report,
    list_session_events,
    list_session_reports,
    launch_agent_run,
    new_thread_id,
    prepare_branch_plan,
    resume_agent_run,
    run_branch_task,
)


router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])
logger = logging.getLogger(__name__)


def _start_agent_worker(
    target: Callable[..., None],
    *,
    run_id: str,
    **kwargs: Any,
) -> None:
    def run() -> None:
        try:
            target(run_id=run_id, **kwargs)
        except Exception:
            logger.exception("Agent worker failed for run_id=%s", run_id)

    threading.Thread(
        target=run,
        name=f"agent-run-{run_id}",
        daemon=True,
    ).start()


@router.post("/{run_id}/cancel", response_model=AgentRunCancelResponse)
def cancel_run(
    run_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentRunCancelResponse:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    username = str(user["sub"])
    session_id = run.project_id or run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    get_owned_session(session_id, username)

    try:
        result = cancel_agent_run(
            services=services,
            run_id=run_id,
            cancelled_by=username,
        )
    except RunCancellationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return AgentRunCancelResponse(
        run_id=result.run_id,
        status="cancelled",
        discarded_node_id=result.discarded_node_id,
    )


def _branch_root_id(services, run: RunRecord) -> str:
    current = run
    seen = {run.run_id}

    while True:
        parent_run_id = current.metadata.get("branched_from_run_id")
        if not isinstance(parent_run_id, str) or not parent_run_id:
            return current.run_id
        if parent_run_id in seen:
            return current.run_id
        seen.add(parent_run_id)
        try:
            current = services.run_service.get_run(parent_run_id)
        except BackendError:
            return current.run_id


@router.get("/reports", response_model=AgentReportListResponse)
def list_agent_reports(
    request: Request,
    session_id: str = Query(...),
    user: dict = Depends(get_current_user),
) -> AgentReportListResponse:
    get_owned_session(session_id, str(user["sub"]))
    reports = list_session_reports(
        services=request.app.state.services,
        session_id=session_id,
    )
    return AgentReportListResponse(
        reports=[
            AgentReportListItem(
                run_id=item.run_id,
                report_artifact_id=item.report_artifact_id,
                created_at=item.created_at,
                report=item.report,
            )
            for item in reports
        ]
    )


@router.get("/session-events")
def list_agent_session_events(
    request: Request,
    session_id: str = Query(...),
    user: dict = Depends(get_current_user),
) -> list[dict]:
    """세션에서 시작된 모든 run(메인 쿼리마다의 트리 전체)의 이벤트를 한 번에 반환한다.

    플레이그라운드 캔버스가 새로고침 후에도 이전 메인 쿼리들의 트리를 복원할 때 쓴다.
    """
    get_owned_session(session_id, str(user["sub"]))
    events = list_session_events(
        services=request.app.state.services,
        session_id=session_id,
    )
    return [event.model_dump(mode="json") for event in events]


@router.delete("/{run_id}", response_model=AgentRunDeleteResponse)
def delete_agent_run(
    run_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentRunDeleteResponse:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = run.project_id or run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    get_owned_session(session_id, str(user["sub"]))

    try:
        result = delete_terminal_run_data(services=services, run_id=run_id)
    except RunDeletionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentRunDeleteResponse(
        run_id=result.run_id,
        deleted_event_count=result.deleted_event_count,
        deleted_artifact_count=result.deleted_artifact_count,
    )


@router.get(
    "/{run_id}/nodes/{node_id}/summary",
    response_model=AgentNodeSummaryResponse,
)
def read_agent_node_summary(
    run_id: str,
    node_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentNodeSummaryResponse:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = run.project_id or run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    get_owned_session(session_id, str(user["sub"]))

    try:
        result = get_node_summary(services=services, run_id=run_id, node_id=node_id)
    except NodeSummaryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentNodeSummaryResponse(
        run_id=run_id,
        node_id=result.node_id,
        agent_name=result.agent_name,
        summary_artifact_id=result.summary_artifact_id,
        summary=result.summary,
    )


@router.post(
    "/{run_id}/nodes/{node_id}/report",
    response_model=AgentNodeReportResponse,
)
def create_agent_node_report(
    run_id: str,
    node_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentNodeReportResponse:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = run.project_id or run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    get_owned_session(session_id, str(user["sub"]))

    try:
        result = generate_node_report(services=services, run_id=run_id, node_id=node_id)
    except NodeSummaryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NodeReportGenerationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return AgentNodeReportResponse(
        run_id=run_id,
        node_id=result.node_id,
        report_artifact_id=result.report_artifact_id,
        created_at=result.created_at,
        report=result.report,
    )


@router.get("/{run_id}/artifacts/{artifact_id}/content")
def read_agent_run_artifact_content(
    run_id: str,
    artifact_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> Response:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
        artifact = services.artifact_registry.get_artifact(artifact_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = run.project_id or run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    get_owned_session(session_id, str(user["sub"]))

    if artifact.run_id != run_id:
        raise HTTPException(status_code=404, detail="실행에 연결된 artifact가 아닙니다.")

    try:
        path = services.artifact_store.get_path(artifact_id)
        content = path.read_bytes()
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.post("", response_model=AgentRunResponse, status_code=202)
def create_agent_run(
    payload: AgentRunCreateRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentRunResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    username = str(user["sub"])
    session = get_owned_session(payload.session_id, username)
    services = request.app.state.services
    thread_id = new_thread_id()
    run = services.run_service.create_run(
        thread_id=thread_id,
        project_id=session.id,
        metadata={"query": query, "session_id": session.id, "session_db": session.session_db},
    )

    _start_agent_worker(
        launch_agent_run,
        run_id=run.run_id,
        services=services,
        session=session,
        username=username,
        query=query,
        thread_id=thread_id,
    )

    return AgentRunResponse(
        run_id=run.run_id,
        thread_id=thread_id,
        status="created",
        query=query,
        session_id=session.id,
    )


@router.post("/{run_id}/resume", response_model=AgentRunResumeResponse, status_code=202)
def resume_run(
    run_id: str,
    payload: AgentRunResumeRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentRunResumeResponse:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    username = str(user["sub"])
    session_id = run.project_id or run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    session = get_owned_session(session_id, username)

    if not run.thread_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 thread 정보가 없습니다.")

    if payload.type in ("clarification", "analysis_review"):
        # 둘 다 LangGraph interrupt 기반 — waiting_input 상태 + interrupt_type 일치 확인.
        if run.status != RunStatus.waiting_input:
            raise HTTPException(status_code=409, detail="입력을 기다리는 실행만 재개할 수 있습니다.")
        if run.metadata.get("interrupt_type") != payload.type:
            raise HTTPException(status_code=409, detail=f"현재 대기 요청은 {payload.type} 유형이 아닙니다.")
        try:
            services.run_service.claim_waiting_input(run_id, metadata={"resumed_from": payload.type})
        except BackendError as exc:
            if exc.code == "RUN_NOT_WAITING_INPUT":
                raise HTTPException(
                    status_code=409,
                    detail="이미 재개되었거나 더 이상 입력 대기 상태가 아닙니다.",
                ) from exc
            raise
        resume_payload: dict[str, object] = (
            {"answer": payload.answer}
            if payload.type == "clarification"
            else {
                "approval_id": payload.approval_id,
                "selected_option_id": payload.selected_option_id,
                "free_text": payload.free_text,
            }
        )
    else:
        # approval: LangGraph interrupt가 아니라 terminal_state(needs_user_approval)에서
        # 만들어지는 단순 승인 대기 — waiting_input이 아니라 waiting_approval 상태.
        if run.status != RunStatus.waiting_approval:
            raise HTTPException(status_code=409, detail="승인을 기다리는 실행만 재개할 수 있습니다.")
        try:
            services.run_service.claim_waiting_approval(run_id, metadata={"resumed_from": "approval"})
        except BackendError as exc:
            if exc.code == "RUN_NOT_WAITING_APPROVAL":
                raise HTTPException(
                    status_code=409,
                    detail="이미 재개되었거나 더 이상 승인 대기 상태가 아닙니다.",
                ) from exc
            raise
        resume_payload = {"approved": payload.approved}
        if payload.reason:
            resume_payload["reason"] = payload.reason

    _start_agent_worker(
        resume_agent_run,
        run_id=run_id,
        services=services,
        session=session,
        resume_payload=resume_payload,
        thread_id=run.thread_id,
    )
    return AgentRunResumeResponse(
        run_id=run_id,
        thread_id=run.thread_id,
        status="running",
        resume_type=payload.type,
    )


@router.post("/{run_id}/branch", response_model=AgentRunBranchResponse, status_code=202)
def branch_run(
    run_id: str,
    payload: AgentRunBranchRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AgentRunBranchResponse:
    services = request.app.state.services
    try:
        source_run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    username = str(user["sub"])
    session_id = source_run.project_id or source_run.metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 세션 정보가 없습니다.")
    session = get_owned_session(session_id, username)

    if source_run.status != RunStatus.succeeded:
        raise HTTPException(status_code=409, detail="완료된 실행만 분기할 수 있습니다.")
    if not source_run.thread_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 thread 정보가 없습니다.")

    try:
        plan = prepare_branch_plan(services=services, run=source_run, start_stage=payload.start_stage)
    except BranchPlanError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # 원본 run은 이미 succeeded(종료) 상태라 재사용할 수 없다(RunService.update_status가
    # 종료된 run의 상태 변경을 막음) — 분기는 항상 새 run_id를 발급하고, 원본은
    # branched_from_run_id 메타데이터로만 연결한다.
    branch_run = services.run_service.create_run(
        thread_id=source_run.thread_id,
        project_id=session.id,
        metadata={
            "session_id": session.id,
            "query": plan.original_question,
            "branched_from_run_id": run_id,
        },
    )

    _start_agent_worker(
        run_branch_task,
        run_id=branch_run.run_id,
        services=services,
        session=session,
        thread_id=source_run.thread_id,
        start_stage=payload.start_stage,
        instruction=payload.instruction,
        upstream_artifact_ids=plan.upstream_artifact_ids,
        original_question=plan.original_question,
        target_table=plan.target_table,
        parent_node_id=payload.parent_node_id or plan.default_parent_node_id,
    )

    return AgentRunBranchResponse(
        run_id=branch_run.run_id,
        thread_id=source_run.thread_id,
        status="created",
        start_stage=payload.start_stage,
        source_run_id=run_id,
    )


@router.get("/{run_id}/related-events")
def list_related_agent_run_events(run_id: str, request: Request, _user: dict = Depends(get_current_user)) -> list[dict]:
    services = request.app.state.services
    try:
        run = services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    root_id = _branch_root_id(services, run)
    candidate_runs = services.run_service.list_runs(
        thread_id=run.thread_id,
        project_id=run.project_id,
    )
    related_events = []
    for candidate_run in candidate_runs:
        if _branch_root_id(services, candidate_run) != root_id:
            continue
        related_events.extend(services.run_service.list_events(candidate_run.run_id))

    related_events.sort(key=lambda event: (event.created_at or "", event.event_id))
    return [event.model_dump(mode="json") for event in related_events]


@router.get("/{run_id}")
def get_agent_run(run_id: str, request: Request, _user: dict = Depends(get_current_user)) -> dict:
    try:
        run = request.app.state.services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return run.model_dump(mode="json")


@router.get("/{run_id}/events")
def list_agent_run_events(run_id: str, request: Request, _user: dict = Depends(get_current_user)) -> list[dict]:
    try:
        events = request.app.state.services.run_service.list_events(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [event.model_dump(mode="json") for event in events]


@router.get("/{run_id}/events/stream")
def stream_agent_run_events(
    run_id: str,
    request: Request,
    after: str | None = Query(default=None),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    _user: dict = Depends(get_current_user),
) -> StreamingResponse:
    try:
        request.app.state.services.run_service.get_run(run_id)
    except BackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cursor = last_event_id or after
    return StreamingResponse(
        stream_run_events(
            request,
            request.app.state.services.run_service,
            run_id,
            cursor,
            poll_interval=EVENT_STREAM_POLL_INTERVAL_SECONDS,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
