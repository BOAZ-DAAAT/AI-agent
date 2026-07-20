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

from .event_stream import EVENT_STREAM_POLL_INTERVAL_SECONDS, stream_run_events
from .schemas import AgentRunCreateRequest, AgentRunResponse
from .service import launch_agent_run, new_thread_id


router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


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
