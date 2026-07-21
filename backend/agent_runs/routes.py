from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse

from backend.auth.deps import get_current_user
from backend.session.service import get_owned_session
from data_agent_backend.models.common import BackendError
from data_agent_backend.models.runs import RunStatus

from .event_stream import EVENT_STREAM_POLL_INTERVAL_SECONDS, stream_run_events
from .schemas import (
    AgentNodeSummaryResponse,
    AgentRunBranchRequest,
    AgentRunBranchResponse,
    AgentRunCreateRequest,
    AgentRunResponse,
    AgentRunResumeRequest,
    AgentRunResumeResponse,
)
from .service import (
    BranchPlanError,
    NodeSummaryNotFoundError,
    get_node_summary,
    launch_agent_run,
    new_thread_id,
    prepare_branch_plan,
    resume_agent_run,
    run_branch_task,
)


router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


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


@router.post("", response_model=AgentRunResponse, status_code=202)
def create_agent_run(
    payload: AgentRunCreateRequest,
    background_tasks: BackgroundTasks,
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

    background_tasks.add_task(
        launch_agent_run,
        services=services,
        session=session,
        username=username,
        query=query,
        run_id=run.run_id,
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
    background_tasks: BackgroundTasks,
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

    if run.status != RunStatus.waiting_input:
        raise HTTPException(
            status_code=409,
            detail="설명 입력을 기다리는 실행만 재개할 수 있습니다.",
        )
    if run.metadata.get("interrupt_type") != "clarification":
        raise HTTPException(status_code=409, detail="현재 대기 요청은 clarification 유형이 아닙니다.")
    if not run.thread_id:
        raise HTTPException(status_code=409, detail="실행에 연결된 thread 정보가 없습니다.")

    try:
        services.run_service.claim_waiting_input(
            run_id,
            metadata={"resumed_from": "clarification"},
        )
    except BackendError as exc:
        if exc.code == "RUN_NOT_WAITING_INPUT":
            raise HTTPException(
                status_code=409,
                detail="이미 재개되었거나 더 이상 입력 대기 상태가 아닙니다.",
            ) from exc
        raise

    background_tasks.add_task(
        resume_agent_run,
        services=services,
        session=session,
        answer=payload.answer,
        run_id=run_id,
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
    background_tasks: BackgroundTasks,
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

    background_tasks.add_task(
        run_branch_task,
        services=services,
        session=session,
        run_id=branch_run.run_id,
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
